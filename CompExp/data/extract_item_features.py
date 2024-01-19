'''
Extract items' features from training data
'''

from collections import defaultdict, Counter
import json
from statistics import mean

import config

ITEM_ATTR_FILE = config.ITEM_ATTR_FILE
TRAIN_FILE = config.TRAIN_FILE
ATTR_FILE = config.ATTR_FILE
ITEM_FILE = config.ITEM_FILE

MIN_THRESHOLD = 1


def main():
    with open(ATTR_FILE) as feature_file:
        features = json.load(feature_file)
        top_features = features[:50]
        features = set(features)

    data = defaultdict(Counter)

    with open(TRAIN_FILE) as train_file:
        lines = train_file.read().split('\n')

        for line in lines:
            item, user, score, rvw = json.loads(line)

            for _, text in rvw:
                for word in text.split(' '):
                    if word in features:
                        data[item][word] += 1

    data = {
        k: [
            i[0] for i in
            sorted(v.items(), key=lambda i: i[1], reverse=True)
            if i[1] >= MIN_THRESHOLD
        ]
        for k, v in data.items()
    }

    with open(ITEM_FILE) as f:
        n_items = len([i for i in f.read().split('\n') if i])

    item_wo_features = [
        i for i in range(n_items) if i not in data or not data[i]
    ]
    print('Num of items with 0 features:', item_wo_features)

    if item_wo_features:
        print('Sample popular features for empty items')

        for i in item_wo_features:
            data[i] = top_features  # top popular features

    print('Max number of features', max([len(v) for v in data.values()]))
    print('Min number of features', min([len(v) for v in data.values()]))
    print('Avg number of features', mean([len(v) for v in data.values()]))

    with open(ITEM_ATTR_FILE, 'w') as if_file:
        json.dump(data, if_file)


if __name__ == '__main__':
    main()
