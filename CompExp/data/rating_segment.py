from collections import defaultdict, Counter
import json

# from nltk.tokenize import sent_tokenize, word_tokenize

import config

DATA_FILE = config.DATA_FILE

ITERS = 5
TOP_P = 35


def main():
    sen_count = 0
    word_counts = Counter()
    word_cat_counts = defaultdict(Counter)
    cat_counts = Counter()

    with open(DATA_FILE) as f:
        for j, line in enumerate(f):
            _, _, score, text = json.loads(line)
            # if 0 <= score <= 5:
            #     cat = '1-5'
            # elif 5 < score <= 10:
            #     cat = '6-10'
            # elif 10 < score <= 15:
            #     cat = '11-15'
            # elif 15 < score <= 20:
            #     cat = '16-20'

            sentences = [
                s.split(' ')
                for _, s in text
            ]

            for i, words in enumerate(sentences):
                asp = text[i][0][0]
                if asp not in score:
                    continue
                cat = score[asp]

                for w in words:
                    word_counts[w] += 1
                    word_cat_counts[w][cat] += 1

                sen_count += 1
                cat_counts[cat] += 1

            if not j % 50000:
                print(f'handled {j}')
            # if j == 20000:
            #     break

        stop_words = {w for w, _ in word_counts.most_common(200)}
        print('Stop Words:', stop_words)

        cat_words = defaultdict(list)

        for w, counts in word_cat_counts.items():
            if word_counts[w] < 10 or w in stop_words:
                continue

            max_cat, max_chi = None, -1
            for cat, count in counts.items():
                c1 = count
                c2 = word_counts[w] - c1
                c3 = cat_counts[cat] - c1
                c4 = sen_count - cat_counts[cat] - c2

                chi_square = (c1 * c4 - c2 * c3) ** 2 / \
                    ((c1 + c3) * (c2 + c4) * (c1 + c2) * (c3 + c4))

                if chi_square > max_chi:
                    max_cat = cat
                    max_chi = chi_square

            # assign one to maximum one category
            cat_words[max_cat].append((max_chi, w))

        word_cats = defaultdict(list)
        for cat, words in cat_words.items():
            words.sort(reverse=True)

            for _, w in words[:TOP_P]:
                word_cats[w] = cat

            print('top 10 words for', cat)
            for i in range(10):
                print(words[i][1])
            print()

    print('\nResult:')
    print(json.dumps(dict(word_cats)))


if __name__ == '__main__':
    main()
