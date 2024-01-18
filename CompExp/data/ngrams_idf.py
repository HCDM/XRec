import math
import json
from collections import Counter

from nltk.util import ngrams

import config

TRAIN_FILE = config.TRAIN_FILE
VOC_FILE = config.VOC_FILE
NGRAM_IDF_FILE = config.NGRAM_IDF_FILE

N_GRAMS = 1
LOG_BASE = 10
ADD_ONE = True


def main():
    print('read voc:', VOC_FILE)

    with open(VOC_FILE, encoding='utf8') as file:
        voc = set(json.load(file))

    print('read corpus:', TRAIN_FILE)

    ngram_counts = Counter()
    N = 0

    with open(TRAIN_FILE, encoding='utf8') as file:
        for line in file:
            if not line:
                continue

            e = json.loads(line)
            for _, s in e[3]:
                grams = set()  # ngrams per sentence
                s = s.split(' ')

                for n in range(1, N_GRAMS + 1):
                    grams |= set(ngrams(s, n))

                for gram in grams:
                    if all(g in voc for g in gram):
                        ngram_counts[gram] += 1

                N += 1

    res = {' '.join(k): math.log(N / v, LOG_BASE) + (1 if ADD_ONE else 0) for k, v in ngram_counts.items()}

    print('total number of ngrams:', len(ngram_counts))

    with open(NGRAM_IDF_FILE, 'w+') as f:
        json.dump(res, f)


if __name__ == '__main__':
    main()
