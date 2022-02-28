import os
import json
import itertools
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

from .voc import voc
from .features import item_features
from .utils import AttrDict

ENV = os.environ['ENV'] if 'ENV' in os.environ else None


class Review:
    def __init__(self, user, item, score, text=[], frs=None):
        self.user = user
        self.item = item
        self.score = score
        self.text = text
        self.frs = frs


class ReviewDataset(Dataset):
    def __init__(self, reviews):
        self.reviews = reviews

        self.user_dict = defaultdict(dict)
        self.item_dict = defaultdict(dict)
        for review in self.reviews:
            if review.item not in self.user_dict[review.user]:
                self.user_dict[review.user][review.item] = review

            if review.user not in self.item_dict[review.item]:
                self.item_dict[review.item][review.user] = review

    @classmethod
    def load(cls, filepath, max_length):
        def parse(line):
            item, user, score, rvw = json.loads(line)
            user = int(user)
            item = int(item)
            score = float(score)

            text = []
            frs = []  # feature ratio

            for fea_opts, sen in rvw:
                words = sen.split(' ')

                if len(words) < 4:
                    continue

                words = words[:max_length-1]

                # remove text with too many unk token
                word_idxs = voc.words_2_idx(words)
                if len([w for w in word_idxs if w == voc.unk]) >= 5:
                    continue

                text.append(' '.join(words))

                frs.append(len(fea_opts) / len(words))

            return Review(user, item, score, text=text, frs=frs)

        # Read the file and split into lines
        with open(filepath, encoding='utf-8') as f:
            lines = f.read().split('\n')

            # for fast development, just pick 5000 samples
            if ENV == 'DEV':
                lines = lines[:5000]

        # Map every line into review
        rvws = [parse(line) for line in lines if line]

        return ReviewDataset(rvws)

    # Return review
    def __getitem__(self, idx):
        return self.reviews[idx]

    # Return the number of elements of the dataset.
    def __len__(self):
        return len(self.reviews)

    def get_score_range(self):
        ''' need ensure dataset cover all scores '''
        return min(r.score for r in self.reviews), max(r.score for r in self.reviews)

    def rvw_subset(self):
        return ReviewDataset([r for r in self.reviews if r.text])

    def random_subset(self, n):
        return ReviewDataset(random.sample(self.reviews, n))

    def get_review(self, uid, iid):
        if iid in self.user_dict[uid]:
            return self.user_dict[uid][iid]
        else:
            return None

    def get_reviews_by_uid(self, uid):
        return list(self.user_dict[uid].values())

    def get_reviews_by_iid(self, iid):
        return list(self.item_dict[iid].values())

    @property
    def item_ids(self):
        return set(r.item for r in self.reviews)

    @property
    def user_ids(self):
        return set(r.user for r in self.reviews)


def binary_mask(matrix, *maskvalues):
    maskvalues = set(maskvalues)

    mask = [
        [
            0 if index in maskvalues else 1
            for index in row
        ]
        for row in matrix
    ]

    return mask


def basic_builder(samples):
    users = torch.tensor([s.user for s in samples])
    items = torch.tensor([s.item for s in samples])
    scores = torch.tensor([s.score for s in samples])

    return AttrDict(
        users=users,
        items=items,
        scores=scores
    )


def ui_builder(samples):
    users = torch.tensor([s.user for s in samples])
    items = torch.tensor([s.item for s in samples])

    return AttrDict(
        users=users,
        items=items
    )


def feature_builder(samples):
    max_n_fea = 100

    i_features = [
        voc.words_2_idx(item_features[r.item][:max_n_fea])
        for r in samples
    ]
    max_n_f = max(len(f) for f in i_features)
    i_features = [
        f + [voc.pad_idx] * (max_n_f - len(f))
        for f in i_features
    ]
    if_mask = binary_mask(i_features, voc.pad_idx)
    if_mask = torch.BoolTensor(if_mask)
    i_features = torch.tensor(i_features)

    return AttrDict(
        i_features=i_features,
        if_mask=if_mask
    )


def score_builder(samples):
    scores = torch.tensor([s.score for s in samples])

    return AttrDict(scores=scores)


def content_builder(samples):
    # batch of array of words
    rvw_batches = [
        random.choice(r.text) if r.text else ''
        for r in samples
    ]

    rvw_batches = [
        voc.words_2_idx(sen.split(' ')) + [voc.eos_idx] if sen else []
        for sen in rvw_batches
    ]

    pad_seqs = list(itertools.zip_longest(*rvw_batches, fillvalue=voc.pad_idx))

    mask = binary_mask(pad_seqs, voc.pad_idx)
    mask = torch.BoolTensor(mask)

    words = torch.tensor(pad_seqs)

    return AttrDict(words=words, mask=mask)


class ReviewBuilder:
    '''
    Inputs:
      samples: array of reviews
    Outputs:
      users: (batch)
      items: (batch)
      scores: (batch)
      words: (seq, batch)
      mask: (seq, batch)
    '''

    def __init__(
        self,
        need_scores=True,
        need_features=True,
        need_content=True,
        no_empty=False
    ):
        # TODO when batch size is too small, there is nothing to return
        self.no_empty = no_empty

        self.builders = []
        if need_scores:
            self.builders.append(score_builder)
        if need_features:
            self.builders.append(feature_builder)
        if need_content:
            self.builders.append(content_builder)

    def __call__(self, samples):
        if self.no_empty:
            samples = [s for s in samples if s.text]

            if not samples:
                raise Exception('no samples has review to build')

        data = ui_builder(samples)
        for builder in self.builders:
            data.update(builder(samples))

        return data


def grp_wrap_collate(collate_fn, dataset, grp_config):
    grp_size = grp_config['grp_size']
    n_min_rated = grp_config['n_min_rated']
    assert n_min_rated <= grp_size

    max_item_id = max(dataset.item_ids)

    def get_group(rvw):
        grp = [rvw]
        if n_min_rated > 1:
            # remove itself
            pool = [r for r in dataset.get_reviews_by_uid(
                rvw.user) if r != rvw]

            if len(pool) < n_min_rated - 1:
                raise Exception('Not enough rated items in the user pool')
            else:
                grp += random.sample(pool, k=n_min_rated - 1)

        while len(grp) < grp_size:
            iid = random.randint(0, max_item_id)
            counter_rvw = dataset.get_review(rvw.user, iid)

            if counter_rvw:
                grp.append(counter_rvw)
            else:
                # negative sampling
                grp.append(Review(rvw.user, iid, 0))

        return grp

    def wrapper(samples):
        '''
        Inputs:
          grp_samples: [batch, grp_size] nested array
        Outputs:
          data_dict: AttrDict
            attributes of the wrapped collate function
            grp_size: int
        '''

        grp_samples = sum([get_group(sample) for sample in samples], [])

        batch_data = collate_fn(grp_samples)
        batch_data.grp_size = grp_size

        return batch_data

    return wrapper


class ReviewGroupDataLoader(DataLoader):
    '''
    Review Dataset Loader supporting group by user
      dataset: ReviewDataset
      collate_fn: string
        'basic': collate (users, items, scores)
        'review': collate (users, items, scores, words, mask)
      grp_config:
        grp_size: number of reviews per group, default 0 for no group
        n_min_rated: minimum number of rated items, default None for all rated
    '''

    def __init__(self, dataset, collate_fn=None, grp_config=None, no_empty=False, **kargs):
        if grp_config is not None:
            # need group reviews
            collate_fn = grp_wrap_collate(collate_fn, dataset, grp_config)

        if no_empty:
            dataset.remove_empty()

        super().__init__(dataset, collate_fn=collate_fn, **kargs)
