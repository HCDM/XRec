import json
import re

import config

DATA_FILE = config.DATA_FILE
VOC_FILE = config.VOC_FILE


def main():
    voc = {}
    print('read corpus:', DATA_FILE)

    with open(DATA_FILE, encoding='utf8') as file:
        for line in file:
            line = line.rstrip('\n')

            if not line:
                continue

            _, _, _, rvw = json.loads(line)
            for _, text in rvw:
                tokens = text.split(' ')

                for token in tokens:
                    if token not in voc:
                        voc[token] = 0
                    voc[token] += 1

    print('size of the voc:', len(voc))

    # handle numbers
    num_tokens, num_v = set(), 0
    for k, v in voc.items():
        # remove numbers appear less than 600
        if re.fullmatch(r'([0-9]+|[0-9]*\.[0-9]+)', k) and v < 600:
            num_tokens.add(k)
            num_v += v

    voc = {k: v for k, v in voc.items() if k not in num_tokens}
    # voc['<num>'] = num_v

    # sort and cur
    voc = sorted(voc.items(), key=lambda i: i[1], reverse=True)

    voc_size = 2 * 10 ** 4
    voc = voc[:voc_size]

    print('trim voc to the size of:', len(voc))
    print('minimum freq in voc:', voc[-1][1])

    voc = [k for k, _ in voc]
    if '<unk>' not in voc:
        print('add <unk>')
        voc.append('<unk>')

    with open(VOC_FILE, 'w') as vf:
        json.dump(voc, vf)


if __name__ == '__main__':
    main()
