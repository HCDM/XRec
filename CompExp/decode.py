'''
Decode
'''

import argparse
import json

import torch
from torch.utils.data import DataLoader
from nltk.translate import bleu_score

from config import DEVICE, MAX_LENGTH

from main import init, load_dataset
from src.voc import voc
from src.dataset import ExtBuilder
from src.decoder import SearchDecoder, ExtractDecoder
from src.utils import AttrDict


def decode(searcher, batch_data, n_samples=1):
    # expand batch, 1st dim must be batch size
    batch_data = AttrDict({
        k: t.expand(n_samples, *t.size()).flatten(0, 1)
        for k, t in batch_data
    })

    search_result = searcher(batch_data)
    exps = search_result.words
    words_lens = search_result.words_lens

    exps = exps.view(n_samples, -1, exps.size(-1)).transpose_(0, 1).tolist()
    words_lens = words_lens.view(n_samples, -1).transpose_(0, 1).tolist()

    exps = [
        [
            ' '.join([
                voc[w_idx] for w_idx in exp[:l]
                if w_idx != voc.eos_idx
            ])
            for exp, l in zip(exp_samples, ls)  # review
        ]
        for exp_samples, ls in zip(exps, words_lens)  # batch
    ]

    return dict(exps=exps)


def extract(model, batch_data, n_samples=1):
    res = model(batch_data)

    exps = [
        [' '.join(voc[w_idx] for w_idx in exp[:l].tolist())]
        for exp, l in zip(res.words, res.words_lens)
    ]

    probs, ref_attn = res.probs.tolist(), res.ref_weights.tolist()

    item_words = batch_data.item_words.tolist()
    item_words_lens = batch_data.item_words_lens
    item_exp_counts = batch_data.item_exp_mask.sum(1).tolist()

    ref_words = batch_data.ref_words.tolist()
    ref_words_lens = batch_data.ref_words_lens
    ref_exp_counts = batch_data.ref_exp_mask.sum(1).tolist()

    delta_ratings = batch_data.delta_ratings.tolist()

    item_exps, ref_exps = [], []
    i_offset, r_offset = 0, 0

    for i, (ic, rc) in enumerate(zip(item_exp_counts, ref_exp_counts)):
        i_exps = []
        for k in range(ic):
            e_idx = i_offset + k
            e, e_len = item_words[e_idx], item_words_lens[e_idx]
            p = probs[i][k]
            i_exps.append(f'[ {p:.2f} ] ' + ' '.join(voc[w] for w in e[:e_len]))

        r_exps = []
        for k in range(rc):
            e_idx = r_offset + k
            e, e_len = ref_words[e_idx], ref_words_lens[e_idx]
            dr = delta_ratings[e_idx]
            attn = ref_attn[i][k]
            r_exps.append(f'[ {attn:.2f} ] ' + ' '.join(voc[w] for w in e[:e_len]) + f'{dr:8.1f}')

        item_exps.append(i_exps)
        ref_exps.append(r_exps)

        i_offset += ic
        r_offset += rc

    return dict(exps=exps, item_exps=item_exps, ref_exps=ref_exps)


def extgen_one(model, batch_data):
    extractor, generator = model.extractor, model.generator

    ext_res = extract(extractor, batch_data)
    ext_exps = ext_res['exps']

    gen_res = decode(generator, batch_data)

    return {**ext_res, **gen_res, 'ext_exps': ext_exps}


def _calc_bleu(refs, hypo, types=[1, 2, 4]):
    ''' process entity for bleu; run in parallel '''

    type_wights = [
        [1., 0, 0, 0],
        [.5, .5, 0, 0],
        [1 / 3, 1 / 3, 1 / 3, 0],
        [.25, .25, .25, .25]
    ]

    refs = [s.split(' ') for s in refs]
    hypo = hypo.split(' ')

    sf = bleu_score.SmoothingFunction()

    return [
        bleu_score.sentence_bleu(refs, hypo, smoothing_function=sf.method1, weights=type_wights[t-1])
        for t in types
    ]


def decode_dataset(test_data, builder, searcher, ranker=None, output=None, n_samples=1, pp=False, mode=None):
    testloader = DataLoader(test_data, batch_size=128,
                            shuffle=False, collate_fn=builder)

    if output:
        o_file = open(output, 'w')

    rvw_idx = 0
    for _, batch_data in enumerate(testloader):
        batch_data.to(DEVICE)

        if mode == 'extgen':
            results = extgen_one(searcher, batch_data)

        batch_size = len(results['exps'])
        for i in range(batch_size):
            review = test_data[rvw_idx].rvw
            entity = dict(
                user=review.user,
                item=review.item,
                exps=results['exps'][i]
            )

            if 'ext_exps' in results:
                entity.update(
                    ext_exps=results['ext_exps'][i]
                )

            if pp:
                print('User:', entity['user'], 'Item:', entity['item'])
                print('Rating:', review.score)
                print('GT Exp:')

                for exp in review.text:
                    print(exp)

                print('\nGen Exp:')

                exp_bleu = _calc_bleu(review.text, entity['exps'][0])

                for k, exp in enumerate(entity['exps']):
                    print(exp, [round(b, 4) for b in exp_bleu])

                if mode != 'gen':
                    if 'ext_exps' in results:
                        print('\nExtracted:')

                        ext_exp_bleu = _calc_bleu(review.text, entity['ext_exps'][0])

                        for exp in results['ext_exps'][i]:
                            print(exp, [round(b, 4) for b in ext_exp_bleu])

                    print('\nCandidates:')
                    for exp in results['item_exps'][i]:
                        print(exp)

                    print('\nRefs:')
                    for exp in results['ref_exps'][i]:
                        print(exp)

                print('\n')

            if output:
                o_file.write(json.dumps(entity))
                o_file.write('\n')

            rvw_idx += 1

    if output:
        o_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model name to load checkpoints')
    parser.add_argument('-c', '--checkpoint')

    parser.add_argument('-p', '--prettyprint', action='store_true')
    parser.add_argument('-o', '--output', default=None,
                        help='file path to write output json')
    parser.add_argument('-n', '--n_samples', default=-1,
                        type=int, help='number of samples')

    parser.add_argument(
        '-r', '--ranker', help='text ranker to rank generated content')
    parser.add_argument('--ranker_checkpoint', default='best')

    parser.add_argument('--mode', default='ext',
                        choices=['ext', 'gen', 'extgen'])

    parser.add_argument('-s', '--search', default='greedy',
                        choices=['greedy', 'sample'])
    parser.add_argument('-k', '--topk', default=0, type=int)

    args = parser.parse_args()

    model, misc = init(args.model, args.checkpoint)

    model.eval()

    greedy = args.search == 'greedy'
    topk = args.topk

    model_type = misc['model_config'].MODEL_TYPE

    if model_type == 'CompExp':
        model = AttrDict(
            extractor=ExtractDecoder(model.extractor, greedy=True),
            generator=SearchDecoder(model, max_length=MAX_LENGTH, greedy=greedy, topk=topk)
        )
        searcher = model
        mode = 'extgen'
    else:
        searcher = SearchDecoder(model, max_length=MAX_LENGTH, greedy=greedy)
        mode = 'gen'

    test_dataset = load_dataset('test')

    if args.n_samples > 0:
        step = len(test_dataset.reviews) // args.n_samples
        test_dataset.reviews = test_dataset.reviews[::step]

    builder = ExtBuilder(n_item_exps=30, n_ref_exps=10, n_pos_exps=0)
    decode_dataset(test_dataset, builder, searcher, output=args.output, n_samples=args.n_samples, pp=args.prettyprint, mode=mode)


if __name__ == '__main__':
    with torch.no_grad():
        main()
