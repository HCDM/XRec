import os
import argparse
import json
import random
from collections import defaultdict
from statistics import mean
from multiprocessing import Pool

import torch
from nltk.translate import bleu_score

# import config
from main import load_dataset
from src.features import features, item_features

DIR_PATH = os.path.dirname(__file__)


def eval_rmse(entities, dataset):
  ''' RMSE of predicted ratings '''

  mse_sum = 0
  for e in entities:
    review = dataset.get_review(e['user'], e['item'])
    mse_sum += (review.score - e['pred_score']) ** 2

  return (mse_sum / len(entities)) ** 0.5


def eval_content_rmse(entities, dataset, best_aligned=False):
  ''' RMSE of generated contents towards ground-truth & predicted ratings '''

  gt_mse_sum, pred_mse_sum = 0, 0
  for e in entities:
    review = dataset.get_review(e['user'], e['item'])

    idx = 0 if best_aligned else random.randint(0, len(e['exp_scores']) - 1)
    rvw_score = e['exp_scores'][idx]

    gt_mse_sum += (review.score - rvw_score) ** 2
    pred_mse_sum += (e['pred_score'] - rvw_score) ** 2

  return tuple((mse_sum / len(entities)) ** 0.5 for mse_sum in (gt_mse_sum, pred_mse_sum))


def _calc_bleu(e, review, types, best_aligned=False):
  ''' process entity for bleu; run in parallel '''

  type_wights = [
    [1., 0, 0, 0],
    [.5, .5, 0, 0],
    [1 / 3, 1 / 3, 1 / 3, 0],
    [.25, .25, .25, .25]
  ]

  if not review.text:
    return

  refs = [s.split(' ') for s in review.text]

  hypos = [s.split(' ') for s in e['exps']]
  hypo = hypos[0] if best_aligned else random.choice(hypos)

  sf = bleu_score.SmoothingFunction()

  return [
    bleu_score.sentence_bleu(refs, hypo, smoothing_function=sf.method1, weights=type_wights[t-1])
    for t in types
  ]


def eval_bleu(entities, dataset, types=[1, 2, 3, 4], best_aligned=False):
  bleu_sums = [0.] * len(types)
  length = 0

  with Pool(10) as pool:
    args = (
      (e, dataset.get_review(e['user'], e['item']), types, best_aligned)
      for e in entities
    )

    for b_res in pool.starmap(_calc_bleu, args, chunksize=128):
      if not b_res:
        continue

      for i, score in enumerate(b_res):
        bleu_sums[i] += score

      length += 1

  bleus = tuple(b / length for b in bleu_sums)
  return bleus


def eval_f_pr(entities, dataset):
  p_sum, r_sum, i_p_sum = 0, 0, 0
  pred_i_features = defaultdict(set)

  length = 0

  for e in entities:
    review = dataset.get_review(e['user'], e['item'])
    gt_words = set(' '.join(review.text).split(' '))
    gt_features = gt_words.intersection(features)

    if not gt_features:
      continue

    pred_words = set(' '.join(e['exps']).split(' '))
    pred_features = pred_words.intersection(features)

    matches = pred_features.intersection(gt_features)

    i_features = item_features[review.item]
    i_matches = pred_features.intersection(i_features)
    pred_i_features[review.item] = pred_i_features[review.item].union(i_matches)

    p_sum += len(matches) / len(pred_features) if pred_features else 0
    r_sum += len(matches) / len(gt_features)
    i_p_sum += len(i_matches) / len(pred_features) if pred_features else 0

    length += 1

  coverage = mean(len(s) / len(item_features[k]) for k, s in pred_i_features.items())

  return p_sum / length, r_sum / length, i_p_sum / length, coverage

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--file', help='decoded file')
  parser.add_argument('-a', '--all', help='all metrics', action='store_true')
  parser.add_argument('--best_aligned', help='use the best aligned one which is the first sample generated in decoding', action='store_true')

  parser.add_argument('evals', nargs='*')
  args = parser.parse_args()

  torch.no_grad()

  with open(args.file) as f:
    print(f'Read decoded entities from {args.file}...')
    lines = f.read().split('\n')
    entities = [json.loads(l) for l in lines if l]
    print(f'Read {len(entities)} decoded entities')

  test_dataset = load_dataset('test')

  if args.all:
    metrics = ['rmse', 'content_rmse', 'bleu', 'f_pr']
  else:
    metrics = args.evals

  # Eval metrics
  for ev in metrics:
    if ev == 'rmse':
      mse = eval_rmse(entities, test_dataset)
      print('Rate RMSE: %.4f' % mse)

    elif ev == 'content_rmse':
      gt_rmse, pred_rmse = eval_content_rmse(entities, test_dataset, best_aligned=args.best_aligned)
      print(f'Content RMSE (GT, Pred): {gt_rmse:.4f} {pred_rmse:.4f}')

    elif ev == 'bleu':
      bleu_types = [1, 2, 4]
      bleus = eval_bleu(entities, test_dataset, types=bleu_types, best_aligned=args.best_aligned)
      print(f'Review BLEU {bleu_types}: ', ' '.join(f'{b:.4f}' for b in bleus))

    elif ev == 'f_pr':
      precision, recall, i_precision, coverage = eval_f_pr(entities, test_dataset)
      print('Feature Precision: %.4f' % precision)
      print('Feature Recall: %.4f' % recall)
      print('Item Feature Precision: %.4f' % i_precision)
      print('Item Feature Coverage: %.4f' % coverage)


if __name__ == '__main__':
  main()
