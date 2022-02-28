import os
import math
# import random
import json
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from nltk.translate import bleu_score

from .data import Review, basic_builder, ReviewBuilder
from .loss import mask_nll_loss
from .utils import ndcg
from .voc import voc
from .features import features, item_features
import config

DIR_PATH = os.path.dirname(__file__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 256


def load_ndcg(ndcg_path):
    if not ndcg_path:
        ndcg_path = config.NDCG_TEST_FILE

    with open(ndcg_path) as f:
        ndcg_user_items = json.load(f)
        ndcg_user_items = {int(uid): items for uid,
                           items in ndcg_user_items.items()}
    return ndcg_user_items


def test_review_ndcg(model, dev_data=None, ndcg_user_items=None, ndcg_path=None, searcher=None, pred_rates=False):
    if ndcg_user_items is None:
        ndcg_user_items = load_ndcg(ndcg_path)

    ndcg_sum = 0

    user_items = list(ndcg_user_items.items())

    collate_fn = ReviewBuilder()

    length = 0

    batch_size = 8  # avoid too large batch, each user has 1x reviews
    for i in range(0, len(user_items), batch_size):
        batch_user_items = user_items[i:i+batch_size]

        grp_reviews = []
        grp_size = 0
        for uid, items in batch_user_items:
            grp = [
                r for r in [
                    dev_data.get_review(uid, iid) for iid in items
                ] if r
            ]

            if all(bool(r.text) for r in grp) or searcher:
                grp_reviews += grp
                grp_size += 1

        if not searcher:
            if not grp_size:
                continue

            batch_data = collate_fn(grp_reviews).to(DEVICE)
            scores, words, mask = batch_data.scores, batch_data.words, batch_data.mask

            output, _ = model(words, word_mask=mask)

        else:
            batch_data = collate_fn(grp_reviews).to(DEVICE)
            scores = batch_data.scores

            search_result = searcher(batch_data)

            output, _ = model(search_result.words,
                              rvw_lens=search_result.rvw_lens)

            if pred_rates:
                scores = searcher.model.rate(batch_data)

        output = output.view(grp_size, -1)
        scores = scores.view(grp_size, -1)

        _, indices = output.sort(descending=True)
        ranked_scores = scores.gather(-1, indices)

        ndcg_sum += ndcg(ranked_scores).sum().item()
        length += grp_size

    return ndcg_sum / length


def test_rate_ndcg(test_data, model, ndcg_user_items=None, ndcg_path=None, k=[], builder=basic_builder, batch_size=BATCH_SIZE):
    if ndcg_user_items is None:
        ndcg_user_items = load_ndcg(ndcg_path)

    ndcg_sum = 0
    pure_ndcg_sum = 0

    length = 0
    user_items = list(ndcg_user_items.items())

    if k:
        ndcg_sum = [0] * len(k)
        pure_ndcg_sum = [0] * len(k)

    for i in range(0, len(user_items), batch_size):
        batch_user_items = user_items[i:i+batch_size]
        reviews = []
        n_grps = 0

        for uid, items in batch_user_items:
            grp_reviews = [
                test_data.get_review(uid, iid) or Review(uid, iid, -1)
                for iid in items
            ]
            if any(r.score > 0 for r in grp_reviews):  # cannnot rank all 0
                reviews += grp_reviews
                n_grps += 1

        batch_data = builder(reviews).to(DEVICE)
        scores = batch_data.scores

        output = model.rate(batch_data).view(n_grps, -1)
        _, indices = output.sort(descending=True)
        ranked_scores = scores.view(n_grps, -1).gather(-1, indices)

        pure_ranked_scores = ranked_scores[ranked_scores != -
                                           1].view(n_grps, -1)
        ranked_scores[ranked_scores == -1] = 0

        if k:
            for i, k_ in enumerate([5, 10, 15, 20]):
                ndcg_sum[i] += ndcg(ranked_scores, k_).sum().item()
            for i, k_ in enumerate(k):
                pure_ndcg_sum[i] += ndcg(pure_ranked_scores, k_).sum().item()
        else:
            ndcg_sum += ndcg(ranked_scores).sum().item()
            pure_ndcg_sum += ndcg(pure_ranked_scores).sum().item()

        length += n_grps

    if k:
        return [n / length for n in ndcg_sum], [n / length for n in pure_ndcg_sum]
    else:
        return ndcg_sum / length, pure_ndcg_sum / length


def test_rate_rmse(test_data, model, builder=basic_builder, batch_size=BATCH_SIZE):
    testloader = DataLoader(test_data, batch_size=batch_size,
                            shuffle=False, collate_fn=builder)

    total_loss = 0
    cat_total_loss, cat_total_count = Counter(), Counter()
    for batch_data in testloader:
        batch_data.to(DEVICE)

        rate_output = model.rate(batch_data)

        min_rating, max_rating = test_data.get_score_range()
        rate_output[rate_output > max_rating] = max_rating
        rate_output[rate_output < min_rating] = min_rating

        total_loss += mse_loss(rate_output,
                               batch_data.scores, reduction='sum').item()

        for r in range(1, 6):
            cat_total_loss[r] += rate_output[batch_data.scores ==
                                             r].sum().item()
            cat_total_count[r] += (batch_data.scores == r).sum().item()

    rmse = math.sqrt(total_loss / len(test_data))

    # print([cat_total_loss[i] / cat_total_count[i] for i in range(1, 6)])
    return rmse


def test_rate_mae(test_data, model, builder=basic_builder, batch_size=BATCH_SIZE):
    testloader = DataLoader(test_data, batch_size=batch_size,
                            shuffle=False, collate_fn=builder)

    total_loss = 0
    for batch_data in testloader:
        batch_data.to(DEVICE)

        rate_output = model.rate(batch_data)

        min_rating, max_rating = test_data.get_score_range()
        rate_output[rate_output > max_rating] = max_rating
        rate_output[rate_output < min_rating] = min_rating

        total_loss += (rate_output - batch_data.scores).abs().sum().item()

    mae = total_loss / len(test_data)

    return mae


def test_review_mse(test_data, model, searcher=None, pred_rates=False, distribution=False):
    collate_fn = ReviewBuilder()

    if not searcher:
        test_data = test_data.rvw_subset()

    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    text_rate_counts = Counter()
    ui_rate_counts = Counter()

    length = 0
    total_loss = 0
    for batch_data in testloader:
        batch_data.to(DEVICE)
        scores = batch_data.scores

        if not searcher:
            rate_output, _ = model(batch_data.words, word_mask=batch_data.mask)
        else:
            search_result = searcher(batch_data)

            rate_output, _ = model(search_result.words,
                                   rvw_lens=search_result.rvw_lens)

            if pred_rates:
                scores = searcher.model.rate(batch_data)

        total_loss += mse_loss(rate_output, scores, reduction='sum').item()
        length += scores.size(0)

        # prediction distribution
        text_rate_counts.update(rate_output.view(-1).round().tolist())
        ui_rate_counts.update(scores.view(-1).round().tolist())

    if distribution:
        print('text_rate_counts')
        print(text_rate_counts)

        print('ui_rate_counts')
        print(ui_rate_counts)

    rmse = math.sqrt(total_loss / length)

    return rmse


def test_review_perplexity(test_data, model, nosens=False):
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=ReviewBuilder())

    sum_nll, sum_words = 0, 0
    for batch_data in testloader:
        batch_data.to(DEVICE)

        words, mask = batch_data.words, batch_data.mask
        if words is None:
            continue

        sos_var = torch.full((1, words.size(-1)), voc.sos_idx,
                             dtype=torch.long, device=DEVICE)
        inp = torch.cat([sos_var, words[:-1]])

        review_output = model.review(batch_data, inp)
        mean_nll = mask_nll_loss(review_output.output, words, mask).item()
        n_words = mask.sum().item()

        sum_nll += mean_nll * n_words
        sum_words += n_words

    ppl = math.exp(sum_nll / sum_words)
    return ppl


def _decode_samples(model, batch_data, n_samples=3):
    samples = []
    for _ in range(n_samples):
        search_output = model(batch_data)

        # switch batch & sequence
        words = search_output.words.transpose(0, 1).tolist()
        rvw_lens = search_output.rvw_lens

        samples.append([
            [
                voc[w_idx]
                for w_idx in b_words[:l]
                if w_idx not in [voc.eos_idx, voc.pad_idx]
            ]
            for b_words, l in zip(words, rvw_lens)
        ])

    batch_size = len(batch_data.users)
    samples = [
        sum([
            samples[i][j] for i in range(n_samples)
        ], [])
        for j in range(batch_size)
    ]

    return samples


def test_review_bleu(test_data, model, voc, types=[2, 4]):
    test_data = test_data.rvw_subset()
    collate_fn = ReviewBuilder()

    length = 0

    type_wights = [
        [1., 0, 0, 0],
        [.5, .5, 0, 0],
        [1 / 3, 1 / 3, 1 / 3, 0],
        [.25, .25, .25, .25]
    ]

    totals = [0.] * len(types)

    sf = bleu_score.SmoothingFunction()

    for i in range(0, len(test_data.reviews), BATCH_SIZE):
        reviews = [r for r in test_data.reviews[i:i+BATCH_SIZE] if r.text]
        batch_data = collate_fn(reviews)
        batch_data.to(DEVICE)
        samples = _decode_samples(model, batch_data, n_samples=1)

        for review, sample in zip(reviews, samples):
            # print(sen)
            # print(sampled_sen)
            refs = [sen.split(' ') for sen in review.text]

            for j, t in enumerate(types):
                weights = type_wights[t-1]
                totals[j] += bleu_score.sentence_bleu(
                    refs, sample, smoothing_function=sf.method1, weights=weights)

            length += 1

    totals = [total / length for total in totals]

    return (*totals, length)


def test_feature_pr(test_data, model):
    collate_fn = ReviewBuilder()

    # pred_count, gt_count, match_count = 0, 0, 0
    p_sum, r_sum, i_p_sum = 0, 0, 0
    # i_match_count = 0
    pred_i_features = defaultdict(set)

    length = 0

    for i in range(0, len(test_data.reviews), BATCH_SIZE):
        reviews = [r for r in test_data.reviews[i:i+BATCH_SIZE] if r.text]

        batch_data = collate_fn(reviews)
        batch_data.to(DEVICE)
        samples = _decode_samples(model, batch_data)

        for review, sample in zip(reviews, samples):
            gt_words = set(' '.join(review.text).split(' '))
            gt_features = gt_words.intersection(features)

            if not gt_features:
                continue

            pred_words = set(sample)
            pred_features = pred_words.intersection(features)

            matches = pred_features.intersection(gt_features)

            i_features = item_features[review.item]
            i_matches = pred_features.intersection(i_features)
            pred_i_features[review.item] = pred_i_features[review.item].union(
                i_matches)

            p_sum += len(matches) / len(pred_features)
            r_sum += len(matches) / len(gt_features)
            i_p_sum += len(i_matches) / len(pred_features)

            length += 1

    print('Avg features coverage per item')
    print(sum(len(s) / len(item_features[k])
              for k, s in pred_i_features.items()) / len(pred_i_features))

    print('Avg item feature precision')
    print(i_p_sum / length)

    return p_sum / length, r_sum / length
