from collections import defaultdict

import config
from utils import load_src
from statistics import mean
# import random

SRC_FILE = config.SRC_FILE

ITEM_FILE = config.ITEM_FILE
USER_FILE = config.USER_FILE

MIN_ITEM_COUNT = config.UI_FILTER_CONFIG['min_item_count']
MIN_USER_COUNT = config.UI_FILTER_CONFIG['min_user_count']
# RANDOM_DROP_ITEM = config.UI_FILTER_CONFIG['random_drop_item']


def main():
    item_map = {}

    for idx, review in enumerate(load_src()):
        iid, uid = review['iid'], review['uid']
        if iid not in item_map:
            item_map[iid] = set()
        item_map[iid].add(uid)

    # if RANDOM_DROP_ITEM:
    #     item_map = {
    #         i: v for i, v in item_map.items()
    #         if random.uniform(0, 1) >= RANDOM_DROP_ITEM
    #     }

    # iterate filering until satisfy MIN
    while True:
        user_map = defaultdict(lambda: 0)

        print('Total Item Size:', len(item_map))
        item_map = {
            k: v for k, v in item_map.items()
            if len(v) >= MIN_ITEM_COUNT
        }
        print('Filtered Item Size', len(item_map))

        for user_set in item_map.values():
            for uid in user_set:
                user_map[uid] += 1

        print('Total User Size:', len(user_map))
        user_map = {k: v for k, v in user_map.items() if v >= MIN_USER_COUNT}
        print('Filtered User Size', len(user_map))

        # filter items users
        item_map = {k: {u for u in v if u in user_map}
                    for k, v in item_map.items()}

        min_item_count = min([len(v) for v in item_map.values()])
        print('Min User of Filtered Item:', min_item_count)

        if min_item_count >= MIN_ITEM_COUNT:
            break

    item_map = {k: len(v) for k, v in item_map.items()}

    item_map = {
        k[0]: i for i, k in
        enumerate(sorted(item_map.items(), key=lambda i: i[1], reverse=True))
    }

    # print(sorted(user_map.items(), key=lambda i: i[1], reverse=True)[:100])
    print('Average items per user:', mean(user_map.values()))

    user_map = {
        k[0]: i for i, k in
        enumerate(sorted(user_map.items(), key=lambda i: i[1], reverse=True))
    }

    with open(USER_FILE, 'w') as usrf, open(ITEM_FILE, 'w') as itmf:
        itmf.write('\n'.join(item_map.keys()))
        usrf.write('\n'.join(user_map.keys()))


if __name__ == '__main__':
    main()
