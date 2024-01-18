''' Purge data with too many unknown token (non-English) '''

import json
import re

import config

DATA_FILE = config.DATA_FILE
VOC_FILE = config.VOC_FILE


def main():
    print('read voc:', VOC_FILE)

    with open(VOC_FILE, encoding='utf8') as file:
        voc = set(json.load(file))

    print('read corpus:', DATA_FILE)

    data = []
    count = 0
    with open(DATA_FILE, encoding='utf8') as file:
        for line in file:
            line = line.rstrip('\n')

            if not line:
                continue

            count += 1

            t1, t2, t3, rvw = json.loads(line)
            filtered_rvw = []
            for fea, text in rvw:
                tokens = text.split(' ')
                n_unk = len([t for t in tokens if t not in voc])

                if n_unk < 4 and n_unk / len(tokens) < 0.33:
                    filtered_rvw.append([fea, text])

            if filtered_rvw:
                data.append([t1, t2, t3, filtered_rvw])

    print('filtered from', count, 'to', len(data))

    with open(DATA_FILE, 'w') as f:
        for d in data:
            f.write(json.dumps(d))
            f.write('\n')


if __name__ == '__main__':
    main()
