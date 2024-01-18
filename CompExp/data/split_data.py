import os
import math
import random
import json

import config


DATA_FILE = config.DATA_FILE

SPLIT_PATH = config.SPLIT_PATH
TRAIN_FILE = config.TRAIN_FILE
DEV_FILE = config.DEV_FILE
TEST_FILE = config.TEST_FILE

DEV_RATIO = 0.10
TEST_RATIO = 0.10


def main():
    user_group = {}

    with open(DATA_FILE) as data_file:
        lines = data_file.read().split('\n')

        for line in lines:
            if not line:
                continue

            uid = (json.loads(line))[1]
            if uid not in user_group:
                user_group[uid] = []

            user_group[uid].append(line)

    # for i in range(10719):
    #   if i not in user_group:
    #     print('missing user:', i)

    # stats = [len(v) for v in user_group.values()]
    # stats = sorted(stats)

    user_group = {u: v for u, v in user_group.items()}

    if not os.path.exists(SPLIT_PATH):
        os.mkdir(SPLIT_PATH)

    with open(TRAIN_FILE, 'w') as train_file, open(DEV_FILE, 'w') as dev_file, open(TEST_FILE, 'w') as test_file:
        train_lines, dev_lines, test_lines = [], [], []

        for uid, lines in user_group.items():
            random.shuffle(lines)

            leng = len(lines)
            dev_num = math.floor(leng * DEV_RATIO)
            test_num = math.floor(leng * TEST_RATIO)

            dev_lines += lines[:dev_num]
            test_lines += lines[-test_num:]
            train_lines += lines[dev_num:-test_num]

        dev_file.write('\n'.join(dev_lines))
        test_file.write('\n'.join(test_lines))
        train_file.write('\n'.join(train_lines))


if __name__ == '__main__':
    main()
