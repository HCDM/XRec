from collections import defaultdict, Counter
import json

from nltk.tokenize import sent_tokenize, word_tokenize

from utils import load_src

ITERS = 5
TOP_P = 35

seeds = dict(
    service='guide highspeed manager bellman e-mail luggage elevator help drink shelf lugged smile wine serve rude continental calls wi-fi router reservation pc cafe eat bar courteous newspaper buffets computer breakfast carte check-in desk laundry computers massage wired concierge waiter frontdesk connectivity front friendly access greet printer route club broadband polite high-speed hi-speed agressive buffet pcs request lunch gym welcome connection fax property printing emails helpful dinner inform stuff email food gem management checkout internet modem connect fitness garage security restaurant drinks immodium network staff wireless ethernet facility ate lounge lan supply printers office receptionist suitcase speed boarding dial-up print entertainment reception wifi meeting service cord'.split(' '),
    rooms='door carpet house furniture bathroom ready comfortable decor furnish open hear renovation light rooms screen mansion condition bathtub bed sink windows kitchen quiet conditioner sleep conditioning double bedroom shampoo size upgrade huge beds floor window tub shower air soap view suite small wall large room pillow television housekeeping queen apartment space king modern mirror twin overlook spacious louver toilet bath toiletry balcony decorate stay noise chair tower tv book suit facing hairdryer square courtyard channel pillows upgraded'.split(' '),
    location='shop parking bus centre location street shuttle beach transportation close city center airport sight-see touristy subway terminal shops located train bank conference pantheon central station tram museum traffic market near avenue shopping underground wharf walk wall outside position car stop supermarket walking taxi tube garage block restaurant surround minute short distance transport union opera boutique district locate restaurants downtown min boulevard site minutes stay park metro attractions mins blocks bloc convenient route square plaza mall'.split(' '),
    value='ranges half accomodate inclusive deal value paying worth extra $ pricy overprice % atmosphere bargain cost charged discount fee hotwire price vacation tax cash quality penny low cheap bill range expectation choice usd expensive pesos barter priceline honeymoon paid accomodation rate accommodate money pay dollar experience resort cheaper anniversary fraction negotiate dollars star quoted rates haggle accommodation prices rating taxes'.split(' '),
    cleanliness='maintain urine barrier grounds tidy attentive white bottled impeccably teeth pools smoke turquoise immaculately neat spotlessly smoker loungers shade towels linen smell cigarette musty clear clean dirty slide nonsmoking crystal exceptionally towel cleanliness bug well-maintained maintained'.split(' ')
)

# seeds = dict(
#     appearance='appearance color colour red ruby'.split(' '),
#     aroma='aroma aromas smell'.split(' '),
#     taste='taste tastes flavor flavors'.split(' ')
# )


def main():
    word_cats = {w: cat for cat, words in seeds.items() for w in words}

    for i in range(ITERS):
        sen_count = 0
        word_counts = Counter()
        word_cat_counts = defaultdict(Counter)
        cat_counts = Counter()

        for j, e in enumerate(load_src()):
            sentences = sent_tokenize(e['text'])
            sentences = [
                s
                for sen in sentences
                for s in sen.split('\n')  # sent_tokenize wont break multilines
            ]

            for s in sentences:
                # words = {w for w in s.split(' ') if w}
                words = {*word_tokenize(s)}

                s_cat_counts = Counter([
                    word_cats[w] for w in words if w in word_cats
                ])

                if not s_cat_counts:
                    continue

                cats = [cat for cat, c in s_cat_counts.items() if c == max(s_cat_counts.values())]

                for w in words:
                    word_counts[w] += 1

                sen_count += 1

                for cat in cats:
                    cat_counts[cat] += 1
                    for w in words:
                        word_cat_counts[w][cat] += 1

            if not j % 50000:
                print(f'iter {i} handled {j}')
            # if j == 20000:
            #     break

        stop_words = {w for w, _ in Counter({
            w: c for w, c in word_counts.items()
            if w not in word_cats
        }).most_common(200)}
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
