import sys
import os
import math
from collections import Counter
import multiprocessing
from multiprocessing import Pool
import argparse
import stanza

import config

from utils import load_src

GPUS = [0, 2]

SRC_FILE = config.SRC_FILE
FEATURE_FILE = config.FEATURE_FILE


def extract_features(args):
  id, from_idx, to_idx = args

  os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUS[id % len(GPUS)])
  print(f'Worker {id} entity range: [{from_idx}, {to_idx})')

  nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos')

  counts = Counter()

  for idx, entity in enumerate(load_src()):
    if idx < from_idx:
      continue
    if idx >= to_idx:
      break

    if not entity['text']:
      continue

    doc = nlp(entity['text'])

    features = set()
    for sen in doc.sentences:
      words = sen.words
      for word in words:

        if word.xpos in {'NN', 'NNS', 'NNP', 'NNPS'}:
          features.add(word.text)

    counts.update(features)

    if idx % 100000 == 0:
      print(f'Worker {id} processed {idx} reviews')
      for f, c in counts.most_common(500):
        print(f'{f}, {c}')

      sys.stdout.flush()

  return counts


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--workers', type=int, default=5)
  args = parser.parse_args()

  n_works = args.workers

  n_entities = sum(1 for _ in load_src())

  with Pool(n_works) as p:
    per_worker = math.ceil(n_entities / n_works)
    params = [
      (i, per_worker * i, per_worker * (i + 1))
      for i in range(n_works)
    ]

    results = p.map(extract_features, params)

  totals = sum(results, Counter())
  totals = sorted(totals.items(), key=lambda e: e[1], reverse=True)

  with open(FEATURE_FILE, 'w') as file:
    file.write('\n'.join(
      f'{f}, {c}' for f, c in totals
    ))


if __name__ == '__main__':
  multiprocessing.set_start_method('spawn')
  main()
