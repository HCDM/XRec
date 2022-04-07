import math
import json
from collections import Counter
from statistics import mean

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import nll_loss
from nltk.util import ngrams
from torch.nn.functional import mse_loss

from config import DEVICE, NGRAM_IDF_FILE
from .dataset import ExtBuilder, build_batch_text, basic_builder
from .voc import voc
from .loss import mask_nll_loss
from .utils import AttrDict, ParallelBleu

BATCH_SIZE = 512


def test_rate_rmse(test_data, model):
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=basic_builder)

    total_loss = 0
    preds = {}
    for batch_data in testloader:
        batch_data.to(DEVICE)

        rate_output = model.rate(batch_data).rate_output

        # min_rating, max_rating = test_data.get_score_range()
        # rate_output[rate_output > max_rating] = max_rating
        # rate_output[rate_output < min_rating] = min_rating
        for u, i, s in zip(batch_data.users.tolist(), batch_data.items.tolist(), rate_output.tolist()):
            preds[f'{u} {i}'] = s

        total_loss += mse_loss(rate_output,
                               batch_data.scores, reduction='sum').item()

    rmse = math.sqrt(total_loss / len(test_data))
    with open('./pred_scores.json', 'w') as f:
        json.dump(preds, f)

    return rmse

def test_ext_perplexity(test_data, model):
    ext_builder = ExtBuilder(n_item_exps=10, n_ref_exps=10, n_pos_exps=1)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=ext_builder)

    sum_nll, counts = 0, 0
    for batch_data in testloader:
        batch_data.to(DEVICE)

        labels = batch_data.item_exp_label
        probs = model(batch_data).probs

        log_probs = (probs + 1e-10).log()
        labels = torch.zeros(log_probs.size(0), dtype=torch.long, device=log_probs.device)
        nll = nll_loss(log_probs, labels, reduction='sum').item()

        sum_nll += nll
        counts += labels.size(0)

    ppl = math.exp(sum_nll / counts)
    return ppl


def test_ext_mrr(test_data, model):
    n_item_exps = 30

    ext_builder = ExtBuilder(n_item_exps=n_item_exps, n_ref_exps=10, n_pos_exps=10)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=ext_builder)

    sum_rr, counts = 0, 0
    rank_counts = Counter()
    for batch_data in testloader:
        batch_data.to(DEVICE)

        labels = batch_data.item_exp_label.tolist()
        probs = model(batch_data).probs

        _, indices = probs.sort(descending=True)
        indices = indices.tolist()

        for idx_list, label in zip(indices, labels):
            i = 0
            while not label[idx_list[i]]:  # until find the 1st pos
                i += 1
            sum_rr += 1 / (i + 1)
            rank_counts[i + 1] += 1

        counts += len(labels)

    rank_dist = [rank_counts[i] / counts for i in range(1, n_item_exps + 1)]
    mrr = sum_rr / counts
    return mrr, rank_dist


def test_ext_var(test_data, model):
    n_item_exps = 10

    ext_builder = ExtBuilder(n_item_exps=n_item_exps, n_ref_exps=10, n_pos_exps=0)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=ext_builder)

    sum_var, count = 0, 0
    for batch_data in testloader:
        batch_data.to(DEVICE)

        probs = model(batch_data).probs

        sum_var += probs.var(-1).sum().item()
        count += probs.size(0)

    return sum_var / count


def test_exp_perplexity(test_data, builder, model):
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=builder)

    sum_nll, sum_words = 0, 0
    for batch_data in testloader:
        batch_data.to(DEVICE)

        words, mask = batch_data.words, batch_data.words_mask

        sos_var = torch.full((words.size(0), 1), voc.sos_idx,
                             dtype=torch.long, device=DEVICE)
        inp = torch.cat([sos_var, words[..., :-1]], dim=1)

        output = model(inp, batch_data).output
        mean_nll = mask_nll_loss(output, words, mask).item()
        n_words = mask.sum().item()

        sum_nll += mean_nll * n_words
        sum_words += n_words

    ppl = math.exp(sum_nll / sum_words)
    return ppl


def test_bleu(test_data, collate_fn, model, types=[2, 4], use_idf=False):
    pb = ParallelBleu(8)

    totals = [0.] * len(types)
    length = 0

    for i in range(0, len(test_data.reviews), BATCH_SIZE):
        # samples = test_data[i:i+BATCH_SIZE]
        samples = [
            test_data[s]
            for s in range(i, min(i + BATCH_SIZE, len(test_data)))
        ]
        batch_data = collate_fn(samples).to(DEVICE)
        # batch_data.ratings = torch.randint_like(batch_data.ratings, 0, 21, device=DEVICE)

        result = model(batch_data)

        exps = [
            [voc[w_idx] for w_idx in exp[:l].tolist() if w_idx != voc.eos_idx]
            for exp, l in zip(result.words, result.words_lens)
        ]

        refs = [sample.rvw.text for sample in samples]

        bleus = pb(exps, refs, types=types, use_idf=use_idf)

        for sample_bleus in bleus:
            for j, b in enumerate(sample_bleus):
                totals[j] += b

        length += len(samples)

    bleu = [total / length for total in totals]

    return bleu


def test_ext_bleu_upper_bound(test_data, types=[2, 4]):
    collate_fn = ExtBuilder(n_item_exps=30, n_ref_exps=10, n_pos_exps=0)

    pb = ParallelBleu(8)
    length = 0

    totals = [0.] * len(types)

    for i in range(0, len(test_data.reviews), BATCH_SIZE):
        # samples = test_data[i:i+BATCH_SIZE]
        samples = [
            test_data[s]
            for s in range(i, min(i + BATCH_SIZE, len(test_data)))
        ]
        paired_data = collate_fn.pair_data(samples)

        for sample, exps in zip(samples, paired_data.item_exps):
            refs = [sample.rvw.text for _ in exps]
            bleus = pb(exps, refs, types=types)

            for j, _ in enumerate(types):
                totals[j] += max(
                    sample_bleus[j]
                    for sample_bleus in bleus
                )

        length += len(samples)

    bleu = [total / length for total in totals]

    return bleu


def test_gen_ext_bleu_upper_bound(test_data, model, types=[2, 4]):
    collate_fn = ExtBuilder(n_item_exps=30, n_ref_exps=10, n_pos_exps=0)

    length = 0

    pb = ParallelBleu(8)
    totals = [0.] * len(types)

    for i in range(0, len(test_data.reviews), BATCH_SIZE):
        # samples = test_data[i:i+BATCH_SIZE]
        samples = [
            test_data[s]
            for s in range(i, min(i + BATCH_SIZE, len(test_data)))
        ]
        paired_data = collate_fn.pair_data(samples)

        ext_exps = []
        for sample, exps in zip(samples, paired_data.item_exps):
            refs = [sample.rvw.text for _ in exps]
            bleus = pb(exps, refs, types=types[:1])

            _, ext_exp = max(
                (sample_bleus[0], exp)
                for sample_bleus, exp in zip(bleus, exps)
            )
            ext_exps.append(' '.join(ext_exp))

        inp_words, inp_words_lens = build_batch_text(ext_exps)
        batch_data = AttrDict(
            users=torch.LongTensor([s.rvw.user for s in samples]),
            ratings=torch.LongTensor([s.rvw.score for s in samples]),
            src_words=inp_words,
            src_words_lens=inp_words_lens,
        ).to(DEVICE)

        result = model(batch_data)

        exps = [
            [voc[w_idx] for w_idx in exp[:l].tolist() if w_idx != voc.eos_idx]
            for exp, l in zip(result.words, result.words_lens)
        ]

        refs = [sample.rvw.text for sample in samples]
        bleus = pb(exps, refs, types=types)
        for sample_bleus in bleus:
            for j, b in enumerate(sample_bleus):
                totals[j] += b

        length += len(samples)

    bleu = [total / length for total in totals]

    return bleu


def test_length(test_data, collate_fn, searcher):
    with open(NGRAM_IDF_FILE) as f:
        ngram_idf = json.load(f)

    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    lens, idfs = [], []
    for batch_data in testloader:
        batch_data.to(DEVICE)

        result = searcher(batch_data)
        exps = [
            [voc[w_idx] for w_idx in exp[:l].tolist() if w_idx != voc.eos_idx]
            for exp, l in zip(result.words, result.words_lens)
        ]

        for exp in exps:
            lens.append(len(exp))
            idfs.append(mean(ngram_idf.get(w, 1) for w in exp))

    return mean(lens), mean(idfs)


def test_diversity(test_data, collate_fn, searcher):
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

    rep_count, total_count = 0, 0
    seq_rep_2_list, uniq_tokens = [], set()

    for batch_data in testloader:
        batch_data.to(DEVICE)

        result = searcher(batch_data)
        exps = [
            [voc[w_idx] for w_idx in exp[:l].tolist() if w_idx != voc.eos_idx]
            for exp, l in zip(result.words, result.words_lens)
        ]

        for exp in exps:
            uniq = set()

            for w in exp:
                if w in uniq:
                    rep_count += 1
                uniq.add(w)

            total_count += len(exp)

            grams = list(ngrams(exp, 2))
            if grams:
                seq_rep_2 = 1 - len(set(grams)) / len(grams)
                seq_rep_2_list.append(seq_rep_2)

            uniq_tokens |= set(uniq)

    return rep_count / total_count, mean(seq_rep_2_list), len(uniq_tokens)


def test_self_bleu(test_data, collate_fn, model, types=[4], N=3000):
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=collate_fn)

    pb = ParallelBleu(8)

    exps = []
    totals = [0.] * len(types)

    for batch_data in testloader:
        batch_data = batch_data.to(DEVICE)

        result = model(batch_data)

        for exp, l in zip(result.words, result.words_lens):
            exps.append([voc[w_idx] for w_idx in exp[:l].tolist() if w_idx != voc.eos_idx])

        if len(exps) >= N:
            break

    exps = exps[:N]
    refs = [exps[:i] + exps[i+1:] for i, exp in enumerate(exps)]

    bleus = pb(exps, refs, types=types, use_idf=False)
    print(len(bleus))
    for sample_bleus in bleus:
        for j, b in enumerate(sample_bleus):
            totals[j] += b

    bleu = [total / N for total in totals]

    return bleu
