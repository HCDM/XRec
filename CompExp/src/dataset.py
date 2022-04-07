import os
import json
import random
from collections import defaultdict
from statistics import mean

import torch
from torch.utils.data import Dataset
from nltk.translate import bleu_score

import config
from .voc import voc
from .utils import AttrDict, idf_bleu
from .utils.data import binary_mask

DIR_PATH = os.path.dirname(__file__)
ENV = os.environ['ENV'] if 'ENV' in os.environ else None

item_cats = {}
if config.DS != 'ta':
    with open(config.ITEM_CATS_FILE) as vf:
        item_cats = {int(i): c for i, c in json.load(vf).items()}


class Review:
    def __init__(self, user, item, score, text=[]):
        self.user = user
        self.item = item
        self.score = score
        self.text = text


class ReviewDataset(Dataset):
    def __init__(self, reviews, train_set=None):
        self.reviews = reviews

        if not train_set:
            self.user_dict = defaultdict(list)
            self.item_dict = defaultdict(list)
            self.user_item_cats = defaultdict(lambda: defaultdict(list))

            for review in self.reviews:
                self.user_dict[review.user].append(review)
                cat = item_cats[review.item]
                self.user_item_cats[review.user][cat].append(review)

                self.item_dict[review.item].append(review)

        else:
            self.user_dict = train_set.user_dict
            self.item_dict = train_set.item_dict
            self.user_item_cats = train_set.user_item_cats

        # reviews with at least one ref in the same category
        self.reviews = [
            rvw for rvw in self.reviews
            if len(self.user_item_cats[rvw.user][item_cats[rvw.item]]) > 1 and len(self.item_dict[rvw.item]) > 1
        ]

    @classmethod
    def load(cls, filepath, max_length=0, train_set=None):
        # Read the file and split into lines
        with open(filepath, encoding='utf-8') as f:
            lines = f.read().split('\n')

        # for fast development, cut 5000 samples
        if ENV == 'DEV':
            lines = lines[:5000]

        rvws = []
        for line in lines:
            if not line:
                continue

            item, user, score, rvw = json.loads(line)
            user = int(user)
            item = int(item)
            score = float(score)

            text = []

            for fea_opts, sen in rvw:
                words = sen.split(' ')

                if len(words) < 4:
                    continue

                words = words[:max_length-1]

                text.append(' '.join(words))

            if not text:
                continue

            rvw = Review(user, item, score, text=text)
            rvws.append(rvw)

        return ReviewDataset(rvws, train_set=train_set)

    # Return review
    def __getitem__(self, idx):
        rvw = self.reviews[idx]
        return self._rvw_ctx(rvw)

    def _rvw_ctx(self, rvw):
        item_rvws = [
            r for r in self.item_dict[rvw.item] if r != rvw
        ]
        cat = item_cats[rvw.item]
        user_rvws = [
            r for r in self.user_item_cats[rvw.user][cat]
            if r != rvw
        ]

        return AttrDict(
            rvw=rvw,
            item_rvws=item_rvws,
            user_rvws=user_rvws
        )

    # Return the number of elements of the dataset.
    def __len__(self):
        return len(self.reviews)

    def get_score_range(self):
        ''' need ensure dataset cover all scores '''
        return min(r.score for r in self.reviews), max(r.score for r in self.reviews)

    def random_subset(self, n):
        return ReviewDataset(random.sample(self.reviews, n))

    def get_reviews_by_uid(self, uid):
        return self.user_dict[uid]

    def get_reviews_by_iid(self, iid):
        return self.item_dict[iid]

    @property
    def item_ids(self):
        return set(r.item for r in self.reviews)

    @property
    def user_ids(self):
        return set(r.user for r in self.reviews)


class TAReviewDataset(ReviewDataset):
    @classmethod
    def load(cls, filepath, max_length=0, train_set=None):
        ASPS = ['service', 'rooms', 'value', 'location', 'cleanliness']

        # Read the file and split into lines
        with open(filepath, encoding='utf-8') as f:
            lines = f.read().split('\n')

        # for fast development, cut 5000 samples
        if ENV == 'DEV':
            lines = lines[:5000]

        rvws = []
        for line in lines:
            if not line:
                continue

            item, user, scores, rvw = json.loads(line)
            user = int(user)
            item = int(item)

            text = []

            asp_text = defaultdict(list)

            for aspects, sen in rvw:
                words = sen.split(' ')

                if len(words) < 4:
                    continue

                words = words[:max_length-1]
                words = ' '.join(words)

                for asp in aspects:
                    asp_text[asp].append(words)

            for asp, text in asp_text.items():
                if asp in scores:
                    ia = item * len(ASPS) + ASPS.index(asp)
                    score = float(scores[asp])
                    rvw = Review(user, ia, score, text=text)
                    rvws.append(rvw)

                    item_cats[ia] = asp

        return ReviewDataset(rvws, train_set=train_set)


def basic_builder(samples):
    samples = [s.rvw for s in samples]

    users = torch.tensor([s.user for s in samples])
    items = torch.tensor([s.item for s in samples])
    scores = torch.tensor([s.score for s in samples])

    return AttrDict(
        users=users,
        items=items,
        scores=scores
    )


def build_batch_text(text_ary, append_eos=False, need_mask=False):
    eos = [voc.eos_idx] if append_eos else []

    text_batches = [
        voc.words_2_idx(text.split(' ')) + eos if text else []
        for text in text_ary
    ]

    lens = [len(text) for text in text_batches]
    max_len = max(lens)

    lens = torch.LongTensor(lens)

    for text in text_batches:
        while len(text) < max_len:
            text.append(voc.pad_idx)

    words = torch.tensor(text_batches)
    if need_mask:
        mask = binary_mask(text_batches, voc.pad_idx)
        mask = torch.BoolTensor(mask)
        return words, lens, mask

    return words, lens


class ExtBuilder:
    def __init__(self, n_item_exps=10, n_user_exps=0, n_ref_exps=10, n_pos_exps=1, return_rvws=False):
        assert n_item_exps >= n_user_exps + n_pos_exps
        self.n_item_exps = n_item_exps
        self.n_ref_exps = n_ref_exps

        self.n_pos_exps = n_pos_exps
        self.n_user_exps = n_user_exps

        self.return_rvws = return_rvws

    def pair_data(self, samples):
        n_item_exps = self.n_item_exps
        n_user_exps = self.n_user_exps
        n_ref_exps = self.n_ref_exps
        n_pos_exps = self.n_pos_exps

        delta_ratings = []  # (batch, n_ref_exps)
        item_exps = []  # (batch, n_item_exps)

        ref_exps = []  # (batch)

        item_exp_label = []  # (batch)

        for sample in samples:
            rvw = sample.rvw

            refs = [
                (sen, u_rvw.score)
                for u_rvw in sample.user_rvws
                for sen in u_rvw.text
            ]

            if len(refs) > n_ref_exps:
                refs = random.sample(refs, n_ref_exps)

            ref_sens, ref_ratings = (list(l) for l in zip(*refs))
            ref_exps.append(ref_sens)
            delta_ratings.append([rvw.score - s for s in ref_ratings])

            n_item_sens = n_item_exps

            # randomly sample positives
            if len(rvw.text) > n_pos_exps:
                pos_sens = random.sample(rvw.text, n_pos_exps)
            else:
                pos_sens = rvw.text

            # index of the last positive exp
            item_exp_label.append(len(pos_sens))

            n_item_sens -= len(pos_sens)

            # sample negative from user
            u_neg_sens = [
                sen for u_rvw in sample.user_rvws for sen in u_rvw.text]
            if len(u_neg_sens) > n_user_exps:
                u_neg_sens = random.sample(u_neg_sens, n_user_exps)

            n_item_sens -= len(u_neg_sens)

            # sample item candidates
            i_sens = [
                sen
                for i_rvw in sample.item_rvws
                for sen in i_rvw.text
            ]

            if len(i_sens) > n_item_sens:
                i_sens = random.sample(i_sens, n_item_sens)

            item_exps.append(pos_sens + u_neg_sens + i_sens)

        return AttrDict(
            delta_ratings=delta_ratings,
            item_exps=item_exps,
            ref_exps=ref_exps,
            item_exp_label=item_exp_label
        )

    def to_tensor(self, samples, paired_data):
        delta_ratings = paired_data.delta_ratings
        item_exps = paired_data.item_exps
        ref_exps = paired_data.ref_exps
        item_exp_label = paired_data.item_exp_label

        # prepare masks
        max_i_len = max(len(i_exps) for i_exps in item_exps)
        max_r_len = max(len(r_exps) for r_exps in ref_exps)

        item_exp_mask = [
            [1] * len(i_exps) + [0] * (max_i_len - len(i_exps))
            for i_exps in item_exps
        ]  # (batch, <= n_item_exps)
        ref_exp_mask = [
            [1] * len(r_exps) + [0] * (max_r_len - len(r_exps))
            for r_exps in ref_exps
        ]  # (batch, <= n_ref_exps)

        item_exp_label = [
            [1] * n_pos + [0] * (max_i_len - n_pos)
            for n_pos in item_exp_label
        ]  # (batch, <= n_item_exps)

        # flatten exps
        item_exps = [e for es in item_exps for e in es]
        ref_exps = [e for es in ref_exps for e in es]
        delta_ratings = [r for rs in delta_ratings for r in rs]

        # convert to tensors
        item_words, item_words_lens = build_batch_text(item_exps)
        ref_words, ref_words_lens = build_batch_text(ref_exps)

        item_exp_mask = torch.BoolTensor(item_exp_mask)
        ref_exp_mask = torch.BoolTensor(ref_exp_mask)
        delta_ratings = torch.FloatTensor(delta_ratings)

        item_exp_label = torch.BoolTensor(item_exp_label)

        users = torch.LongTensor([sample.rvw.user for sample in samples])
        ratings = torch.LongTensor([sample.rvw.score for sample in samples])

        return AttrDict(
            users=users,
            ratings=ratings,

            delta_ratings=delta_ratings,  # (batch)
            item_words=item_words,  # (<= batch * n_item_exps, seq)
            item_words_lens=item_words_lens,  # (<= batch * n_item_exps)
            item_exp_mask=item_exp_mask,  # (batch, n_item_exps)
            item_exp_label=item_exp_label,  # (batch, n_item_exps)
            ref_words=ref_words,  # (<= batch * n_ref_exps , seq)
            ref_words_lens=ref_words_lens,  # (<= batch * n_ref_exps)
            ref_exp_mask=ref_exp_mask  # (batch, n_ref_exps)
        )

    def __call__(self, samples):
        paired_data = self.pair_data(samples)
        data = self.to_tensor(samples, paired_data)

        if self.return_rvws:
            data.rvws = [s.rvw for s in samples]
            data.item_exps = paired_data.item_exps

        return data


class BleuExtBuilder(ExtBuilder):
    def __init__(self, *args, bleu_type=1, use_idf=False, **kargs):
        super().__init__(*args, **kargs)
        self.sf = bleu_score.SmoothingFunction()

        type_weights = [
            [1.],
            [.5, .5],
            [1 / 3, 1 / 3, 1 / 3],
            [.25, .25, .25, .25]
        ]

        self.weights = type_weights[bleu_type-1]
        self.bleu_func = idf_bleu if use_idf else bleu_score.sentence_bleu

    def to_tensor(self, samples, paired_data):
        data = super().to_tensor(samples, paired_data)

        bleus = [
            [self._calc_bleu(exp, sample.rvw) for exp in exps]
            for sample, exps in zip(samples, paired_data.item_exps)
        ]

        label_idices = [b.index(max(b)) for b in bleus]

        item_exp_label = [[0] * data.item_exp_label.size(1) for _ in samples]
        for idx, labels in zip(label_idices, item_exp_label):
            labels[idx] = 1

        data.item_exp_label = torch.BoolTensor(item_exp_label)

        return data

    def _calc_bleu(self, hypo, review):
        refs = [s.split(' ') for s in review.text]
        hypo = hypo.split(' ')

        return self.bleu_func(refs, hypo, smoothing_function=self.sf.method1, weights=self.weights)


class BleuRankBuilder(BleuExtBuilder):
    def __init__(self, *args, adv=True, **kargs):
        super().__init__(*args, **kargs)
        self.adv = adv

    def to_tensor(self, samples, paired_data):
        data = super(BleuExtBuilder, self).to_tensor(samples, paired_data)

        bleus = [
            [self._calc_bleu(exp, sample.rvw) for exp in exps]
            for sample, exps in zip(samples, paired_data.item_exps)
        ]

        # label_idices = [b.index(max(b)) for b in bleus]
        # for idx, sens in zip(label_idices, paired_data.item_exps):
        #     print(sens[idx])
        # exit()

        if self.adv:
            bleu_means = [mean(b) for b in bleus]
            bleus = [[v - m for v in b] for b, m in zip(bleus, bleu_means)]

        item_exp_len = data.item_exp_label.size(1)
        for b in bleus:
            if len(b) < item_exp_len:
                b += [0] * (item_exp_len - len(b))

        data.item_exp_label = torch.FloatTensor(bleus)

        return data


class WordBuilder:
    def __call__(self, samples):
        exps = [
            sen
            for sample in samples
            for sen in sample.text
        ]

        words, words_lens = build_batch_text(exps, append_eos=True)

        return AttrDict(
            words=words,
            words_lens=words_lens
        )


class RewriteDataset(Dataset):
    def __init__(self, data):
        self.data = data

    @classmethod
    def load(cls, filepath, max_length=0):
        # Read the file and split into lines
        with open(filepath, encoding='utf-8') as f:
            lines = f.read().split('\n')

        # for fast development, cut 5000 samples
        if ENV == 'DEV':
            lines = random.sample(lines, 5000)

        data = []
        for line in lines:
            if not line:
                continue

            user, score, exp, ref, item = json.loads(line)[:5]
            user = int(user)
            item = item
            score = float(score)

            text = []

            for sen in (exp, ref):
                words = sen.split(' ')

                if len(words) < 4:
                    break

                words = words[:max_length-1]

                text.append(' '.join(words))

            if len(text) < 2:
                continue

            exp, ref = text
            data.append((user, score, exp, ref, item))

        return cls(data)

    # Return record
    def __getitem__(self, idx):
        return self.data[idx]

    # Return the number of elements of the dataset.
    def __len__(self):
        return len(self.data)


class CompExpGenBuilder:
    def __init__(self, rvw_data, n_ref_exps=10):
        self.rvw_data = rvw_data
        self.n_ref_exps = n_ref_exps

    def __call__(self, samples):
        n_ref_exps = self.n_ref_exps

        ref_exps, delta_ratings = [], []
        target_exps, inp_exps = [], []
        for user, rating, target_exp, inp_exp, item in samples:
            target_exps.append(target_exp)
            inp_exps.append(inp_exp)

            cat = item_cats[item]

            user_rvws = [
                r for r in self.rvw_data.user_item_cats[user][cat]
            ]

            refs = [
                (sen, u_rvw.score)
                for u_rvw in user_rvws
                for sen in u_rvw.text
            ]

            if len(refs) > n_ref_exps:
                refs = random.sample(refs, n_ref_exps)

            ref_sens, ref_ratings = (list(l) for l in zip(*refs))
            ref_exps.append(ref_sens)
            delta_ratings.append([rating - s for s in ref_ratings])

        max_r_len = max(len(r_exps) for r_exps in ref_exps)
        ref_exp_mask = [
            [1] * len(r_exps) + [0] * (max_r_len - len(r_exps))
            for r_exps in ref_exps
        ]
        # flatten exps
        ref_exps = [e for es in ref_exps for e in es]
        delta_ratings = [r for rs in delta_ratings for r in rs]

        ref_words, ref_words_lens = build_batch_text(ref_exps)
        ref_exp_mask = torch.BoolTensor(ref_exp_mask)
        delta_ratings = torch.FloatTensor(delta_ratings)

        item_words, item_words_lens = build_batch_text(inp_exps)
        item_exp_mask = [
            [1]
            for _ in inp_exps
        ]  # (batch, 1)
        item_exp_mask = torch.BoolTensor(item_exp_mask)

        words, words_lens, words_mask = build_batch_text(
            target_exps, append_eos=True, need_mask=True)

        return AttrDict(
            delta_ratings=delta_ratings,  # (batch)
            item_words=item_words,  # (<= batch * n_item_exps, seq)
            item_words_lens=item_words_lens,  # (<= batch * n_item_exps)
            item_exp_mask=item_exp_mask,  # (batch, n_item_exps)

            ref_words=ref_words,  # (<= batch * n_ref_exps , seq)
            ref_words_lens=ref_words_lens,  # (<= batch * n_ref_exps)
            ref_exp_mask=ref_exp_mask,  # (batch, n_ref_exps)

            words=words,
            words_lens=words_lens,
            words_mask=words_mask
        )
