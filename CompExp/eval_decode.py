import os
import argparse
import json
from collections import defaultdict
from statistics import mean

from nltk.util import ngrams

import config
from main import load_dataset
from src.utils import ParallelBleu

DIR_PATH = os.path.dirname(__file__)


def eval_rmse(entities, dataset):
    ''' RMSE of predicted ratings '''

    mse_sum = 0
    for e in entities:
        review = dataset.get_review(e['user'], e['item'])
        mse_sum += (review.score - e['pred_score']) ** 2

    return (mse_sum / len(entities)) ** 0.5


def eval_bleu(entities, dataset, types=[1, 2, 3, 4], use_idf=False):
    pb = ParallelBleu(4)

    ui_rvws = defaultdict(dict)
    for rvw in dataset.reviews:
        ui_rvws[rvw.user][rvw.item] = rvw

    hyps = [e['exps'][0] for e in entities]
    # hyps = [e['ext_exps'][0] for e in entities]

    refs = [ui_rvws[e['user']][e['item']].text for e in entities]

    bleus = pb(hyps, refs, types=types, use_idf=use_idf)

    return [mean(n_bleus) for n_bleus in zip(*bleus)]


def eval_diversity(entities, test_dataset=None):
    with open(config.NGRAM_IDF_FILE) as f:
        ngram_idf = json.load(f)

    lens, idfs = [], []
    rep_count, total_count = 0, 0
    seq_rep_2_list, uniq_tokens = [], set()

    for e in entities:
        exp = e['exps'][0].split(' ')

        lens.append(len(exp))
        idfs.append(mean(ngram_idf.get(w, 1) for w in exp))

        uniq = set()

        for w in exp:
            if w in uniq:
                rep_count += 1
            uniq.add(w)

        total_count += len(exp)

        grams = list(ngrams(exp, 2))
        if grams:
            seq_rep_2 = 1 - len(set(grams)) / len(grams)
            seq_rep_2_list.append(seq_rep_2)

        uniq_tokens |= set(uniq)

    return mean(lens), mean(idfs), rep_count / total_count, mean(seq_rep_2_list), len(uniq_tokens)


def eval_f_pr(entities, dataset):
    p_sum, r_sum = 0, 0

    length = 0

    with open(config.ATTR_FILE) as f:
        features = set(json.load(f))

    ui_rvws = defaultdict(dict)
    for rvw in dataset.reviews:
        ui_rvws[rvw.user][rvw.item] = rvw

    for e in entities:
        review = ui_rvws[e['user']][e['item']]
        gt_words = set(' '.join(review.text).split(' '))
        gt_features = gt_words.intersection(features)

        if not gt_features:
            continue

        pred_words = set(' '.join(e['exps']).split(' '))

        pred_features = pred_words.intersection(features)

        matches = pred_features.intersection(gt_features)

        p_sum += len(matches) / len(pred_features) if pred_features else 0
        r_sum += len(matches) / len(gt_features)

        length += 1

    return p_sum / length, r_sum / length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='decoded file')
    parser.add_argument('-a', '--all', help='all metrics', action='store_true')

    parser.add_argument('evals', nargs='*')
    args = parser.parse_args()

    with open(args.file) as f:
        print(f'Read decoded entities from {args.file}...')
        lines = f.read().split('\n')
        entities = [json.loads(l) for l in lines if l]
        print(f'Read {len(entities)} decoded entities')

    # bleu(entities)

    test_dataset = load_dataset('test')

    res = eval_bleu(entities, test_dataset, types=[1, 2, 4])
    print('BLEU:', res)
    res = eval_bleu(entities, test_dataset, types=[1, 2, 4], use_idf=True)
    print('IDF-BLEU:', res)

    res = eval_diversity(entities, test_dataset)
    print('Len, IDF, rep/l, seq_rep_2, uniq:', *res)

    res = eval_f_pr(entities, test_dataset)
    print('Feature precision, recall:', *res)


if __name__ == '__main__':
    main()
