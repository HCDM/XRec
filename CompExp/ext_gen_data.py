'''
Use extractor to prepare generation pairs
'''

import os
import argparse
import json

import torch

import config
from main import init, load_dataset
# from src.voc import voc
from src.dataset import WordBuilder

DIR_PATH = os.path.dirname(__file__)

DEVICE = config.DEVICE
GEN_PAIR_CORPUS = config.GEN_PAIR_CORPUS
ATTR_FILE = config.ATTR_FILE

BATCH_SIZE = 2048
SIM_DIM_SIZE = 5e5
TOP_K = 3

with open(os.path.join(DIR_PATH, ATTR_FILE)) as f:
    attrs = set(json.load(f))


avg_attr_ratio = [0, 0]


def is_qualified(sen, ref):
    ''' verify if the pair is qualified (e.g. attributes coverage) '''

    sen_words = set(sen.split(' '))
    ref_words = set(ref.split(' '))

    overlap_words = ref_words & sen_words
    ratio_t = len(overlap_words) / len(sen_words)
    ratio_r = len(overlap_words) / len(ref_words)
    if len(overlap_words) < 6 and (ratio_t < 0.25 or ratio_r < 0.25):
        return False

    sen_attrs = sen_words & attrs
    ref_attrs = ref_words & attrs

    if not sen_attrs or not ref_attrs:
        return False

    overlap_attrs = ref_attrs & sen_attrs
    ratio_t = len(overlap_attrs) / len(sen_attrs)
    ratio_r = len(overlap_attrs) / len(ref_attrs)

    if len(overlap_attrs) < 3 and (ratio_t < 0.5 or ratio_r < 0.5):
        return False

    avg_attr_ratio[0] += ratio_t
    avg_attr_ratio[1] += 1

    return True


def produce_dataset(dataset, model, output=os.path.join(DIR_PATH, GEN_PAIR_CORPUS)):
    builder = WordBuilder()

    delta_rating_sum, count = 0, 0

    with open(output, 'w+') as f:
        for k, (item, rvws) in enumerate(dataset.item_dict.items()):
            item_data = [
                (rvw.user, rvw.score, sen)
                for rvw in rvws
                for sen in rvw.text
            ]

            vct_stack = []

            for i in range(0, len(rvws), BATCH_SIZE):
                samples = rvws[i:i+BATCH_SIZE]
                data = builder(samples).to(DEVICE)

                exp_vcts, _ = model.t_encoder(data.words, lens=data.words_lens)

                vct_stack.append(exp_vcts)

            vcts = torch.cat(vct_stack, dim=0)

            # (batch, 1, dim) - (1, batch, dim) => (batch, batch)
            # batch may to too large for the GPU
            target_vcts = vcts.unsqueeze(1)
            ref_vcts = vcts.unsqueeze(0)
            sim_stack = []
            SIM_BATCH_SIZE = int(SIM_DIM_SIZE // vcts.size(0))
            for i in range(0, vcts.size(0), SIM_BATCH_SIZE):
                sims = (target_vcts[i:i+SIM_BATCH_SIZE] - ref_vcts).pow(2).sum(-1)
                sim_stack.append(sims)

            sims = torch.cat(sim_stack, dim=0)

            top_k = min(sims.size(0), TOP_K + 1)  # +1 for itself
            top_sims, top_indices = sims.topk(top_k, dim=1, largest=False)
            top_sims, top_indices = top_sims.tolist(), top_indices.tolist()

            for j, (j_sims, j_indices) in enumerate(zip(top_sims, top_indices)):
                user, score, sen = item_data[j]

                for sim, idx in zip(j_sims, j_indices):
                    if j == idx:
                        continue
                    ref = item_data[idx][2]

                    if not is_qualified(sen, ref):
                        continue

                    delta_rating_sum += abs(item_data[idx][1] - score)
                    count += 1

                    f.write(json.dumps([user, score, sen, ref, item]) + '\n')

            if k and not k % 500:
                print(f'handled {k} items')

    print('average ratio:', avg_attr_ratio[0] / avg_attr_ratio[1])
    print('average delta rating:', delta_rating_sum / count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model name to load checkpoints')
    parser.add_argument('-c', '--checkpoint')

    parser.add_argument('-o', '--output', default=None,
                        help='file path to write output json')

    args = parser.parse_args()

    model, _ = init(args.model, args.checkpoint)

    model.eval()

    dataset = load_dataset('train')

    produce_dataset(dataset, model, output=args.output)


if __name__ == '__main__':
    with torch.no_grad():
        main()
