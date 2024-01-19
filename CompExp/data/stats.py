import json
from collections import Counter
from statistics import mean

import config

DATA_FILE = config.DATA_FILE
NGRAM_IDF_FILE = config.NGRAM_IDF_FILE


def main():
    entity_count, entity_exp_count = 0, 0
    user_counts, item_counts, score_counts, feature_counts = Counter(), Counter(), Counter(), Counter()
    sen_len_counts = Counter()
    sen_idf_sum = 0

    with open(NGRAM_IDF_FILE) as f:
        ngram_idf = json.load(f)

    with open(DATA_FILE, encoding='utf-8') as df:
        for line in df:
            line = line.strip()
            if not line:
                continue

            entity_count += 1

            try:
                iid, uid, score, rvws = json.loads(line)
            except Exception as e:
                print(line, len(line))
                raise e

            item_counts[iid] += 1
            user_counts[uid] += 1
            if type(score) is dict:
                score = score['overall']
            score_counts[score] += 1

            if rvws:
                entity_exp_count += 1

            for features, text in rvws:
                for feature in features:
                    feature_counts[feature] += 1

                words = text.split(' ')
                sen_len_counts[len(words)] += 1
                sen_idf_sum += mean(ngram_idf.get(w, 1) for w in words)

    score_mean = sum(c * v for c, v in score_counts.items()) / entity_count
    score_std = (sum(v * (c - score_mean) ** 2 for c, v in score_counts.items()) / entity_count) ** .5
    avg_sen_len = sum(l * c for l, c in sen_len_counts.items()) / sum(c for c in sen_len_counts.values())
    avg_sen_idf = sen_idf_sum / sum(c for c in sen_len_counts.values())

    print('Num of records:', entity_count)
    print('Num of records with exp:', entity_exp_count)
    print('Num of users:', len(user_counts))
    print('Max items per user:', max(user_counts.values()))
    print('Min items per user:', min(user_counts.values()))
    print('Num of items:', len(item_counts))
    print('Max users per item:', max(item_counts.values()))
    print('Min users per item:', min(item_counts.values()))
    print('Score distribution:', dict(score_counts))
    print('Score mean:', score_mean)
    print('Score std:', score_std)
    print('Num of features:', len(feature_counts))
    print('Max feature count:', max(feature_counts.values()))
    print('Min feature count:', min(feature_counts.values()))
    print('Average sentence length:', avg_sen_len)
    print('Average sentence IDF:', avg_sen_idf)


main()
