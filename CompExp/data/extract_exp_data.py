import sys
import os
import json
import re
import math
import time
import multiprocessing
from multiprocessing import Pool
import argparse
from collections import Counter
from nltk.tokenize import sent_tokenize
import stanza

import config
from utils import load_src

# stanza.download('en')
GPUS = [0, 1, 2, 3]

OUTPUT_PATH = config.OUTPUT_PATH

DATA_FILE = config.DATA_FILE
ATTR_FILE = config.ATTR_FILE

ITEM_FILE = config.ITEM_FILE
USER_FILE = config.USER_FILE

PARSE_RULE = dict(
    yelp='nlp',
    rb='naive',
    ta='aspects'
)[config.DATASET]

with open(ATTR_FILE) as f:
    attributes = json.load(f)
    attributes = {f: i for i, f in enumerate(attributes)}


def load_filtered_date():
    with open(USER_FILE, 'r') as usrf, open(ITEM_FILE, 'r') as itmf:
        user_map = {v: i for i, v in enumerate(usrf.read().split('\n'))}
        item_map = {v: i for i, v in enumerate(itmf.read().split('\n'))}

        return item_map, user_map


class Parser:
    def __init__(self, rule):
        self.rule = rule

        if rule == 'nlp':
            nlp = stanza.Pipeline(
                'en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True)
        else:
            nlp = stanza.Pipeline('en', processors='tokenize',
                                tokenize_no_ssplit=True)

        if rule == 'aspects':
            with open(config.ASPECT_FILE) as f:
                self.word_2_aspect = json.load(f)

        self.nlp = nlp

    def parse(self, entity_sens):
        entity_sen_lens = [len(sens) for sens in entity_sens]

        raw_text = '\n\n'.join(
            sen
            for sens in entity_sens
            for sen in sens
        )

        while True:
            try:
                doc = self.nlp(raw_text)
                break
            except Exception as e:
                print('parsing error:', e)  # like empty string
                # print(raw_text)
                if str(e).startswith('CUDA out of memory'):
                    time.sleep(5)
                    print('GPU memory not enough, retry')
                    sys.stdout.flush()
                    continue

                return [[] for _ in entity_sens]

        entity_exps, idx = [], 0
        for l in entity_sen_lens:
            exps = []

            for sen in doc.sentences[idx:idx+l]:
                if self.rule == 'nlp':
                    exp = self.nlp_rule_exp(sen)
                elif self.rule == 'naive':
                    exp = self.naive_rule_exp(sen)
                elif self.rule == 'aspects':
                    exp = self.aspect_rule_exp(sen)

                if exp:
                    exps.append(exp)

            entity_exps.append(exps)
            idx += l

        return entity_exps

    def aspect_rule_exp(self, sen):
        ''' check if sentence contains aspect word '''
        words = sen.words
        asp_counts = Counter([
            self.word_2_aspect[word.text] for word in words if word.text in self.word_2_aspect
        ])
        if not asp_counts:
            return None

        asps = [asp for asp, c in asp_counts.items() if c == max(asp_counts.values())]

        s = ' '.join([word.text for word in words])
        return [asps, s]

    def naive_rule_exp(self, sen):
        ''' check if sentence contains feature word '''
        is_exp = False
        attrs = set()

        words = sen.words
        for word in words:
            text = word.text

            if text in attributes:
                attrs.add(text)
                is_exp = True

        if is_exp:
            s = ' '.join([word.text for word in words])
            return [list(attrs), s]

        return None

    def nlp_rule_exp(self, sen):
        is_exp = False
        attrs = set()

        words = sen.words
        for word in words:
            text = word.text

            if text in attributes:
                attrs.add(text)

                # case like: chicken xxx is good
                while word.deprel == 'compound':
                    word = words[word.head - 1]

                if word.deprel == 'nsubj' or word.deprel == 'nsubj:pass':
                    head = words[word.head - 1]
                    if head.xpos in {'JJ', 'JJR', 'JJS', 'RB', 'VBN', 'NN'}:
                        is_exp = True

            # case like: great food / service: very poor
            elif word.xpos in {'JJ', 'JJR', 'JJS'}:
                head = words[word.head - 1]
                if head.deprel == 'root' and head.text in attributes:
                    is_exp = True
                    # print(word.text, head.text, head.deprel)
                    # print([word.text for word in words])

        if is_exp:
            s = ' '.join([word.text for word in words])
            return [list(attrs), s]

        return None


def clean_sentence(s):
    s = s.lower()

    # specific for beer data
    if config.DATASET == 'rb':
        s = re.sub(r'<[^>]+>', ' ', s)
        s = re.sub(r'https?://\S*', ' ', s)
        s = re.sub(r'updated:.*, \d{4}', '', s)
        s = re.sub(r'\([^)]*\)', ' ', s)
    elif config.DATASET == 'ta':
        s = s.replace('see more room tips', '')

    s = re.sub(r'\.\.+', '...', s)
    s = re.sub(r'`', '\'', s)
    s = re.sub(r'^\s*-+', '', s)

    s = re.sub(r'-+', ' - ', s) \
        .replace('*', ' ') \
        .replace('/', ' / ') \
        .replace('~', ' ')

    s = re.sub(r'\s\s+', ' ', s).strip()
    return s


def extract(args):
    id, from_idx, to_idx = args

    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUS[id % len(GPUS)])
    print(f'Worker {id} Idx range: [{from_idx}, {to_idx})')

    parser = Parser(PARSE_RULE)

    TMP_FILE = os.path.join(OUTPUT_PATH, f'data_{id}_{from_idx}_{to_idx}.txt')

    # item_map, user_map = filter_data()
    item_map, user_map = load_filtered_date()

    with open(TMP_FILE, 'w') as of:
        write_buf = []

        # feature_count = {k:0 for k in attributes.keys()}

        for idx, review in enumerate(load_src()):
            if idx < from_idx:
                continue
            if idx >= to_idx:
                break

            if idx % 100000 == 0:
                print(f'Worker {id} processed {idx} reviews')

            if review['iid'] not in item_map or review['uid'] not in user_map:
                continue

            sentences = sent_tokenize(review['text'])
            sentences = [
                clean_sentence(s)
                for sen in sentences
                for s in sen.split('\n')  # sent_tokenize wont break multilines
            ]

            sentences = [s for s in sentences if s and len(
                s) < 100 and len(s) > 3]

            i_idx = item_map[review['iid']]
            u_idx = user_map[review['uid']]
            score = review['rating']
            entity = [i_idx, u_idx, score, sentences]
            write_buf.append(entity)

            if len(write_buf) >= 5:
                exp_list = parser.parse([e[3] for e in write_buf])

                write_buf = [
                    json.dumps([*e[:3], exp])
                    for e, exp in zip(write_buf, exp_list)
                    if exp
                ]

                # print(f'Worker {id} write file')
                of.write('\n'.join(write_buf))
                of.write('\n')
                write_buf = []

            sys.stdout.flush()

        if write_buf:
            exp_list = parser.parse([e[3] for e in write_buf])
            write_buf = [
                json.dumps([*e[:3], exp])
                for e, exp in zip(write_buf, exp_list)
                if exp
            ]

            of.write('\n'.join(write_buf))

    return TMP_FILE


def main():
    n_entities = sum(1 for _ in load_src())
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workers', type=int, default=5)
    args = parser.parse_args()

    n_works = args.workers

    with Pool(n_works) as p:
        per_worker = math.ceil(n_entities / n_works)
        params = [
            (i, per_worker * i, per_worker * (i + 1))
            for i in range(n_works)
        ]
        tmp_files = p.map(extract, params)

    with open(DATA_FILE, 'w') as of:
        for tmp in tmp_files:
            with open(tmp) as tp:
                of.write(tp.read())
                of.write('\n')

            os.remove(tmp)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
