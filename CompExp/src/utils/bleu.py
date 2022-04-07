import math
import json
from collections import Counter, defaultdict
from statistics import mean
from multiprocessing import Pool
from collections.abc import Iterable

from nltk.translate import bleu_score
# from nltk.compat import Fraction
from nltk.util import ngrams

import config

_sf = bleu_score.SmoothingFunction()

_ngram_idf = defaultdict(lambda: 1)


def _get_ngram_idf():
    if not _ngram_idf:
        with open(config.NGRAM_IDF_FILE) as f:
            _ngram_idf.update({
                ngram: idf
                # int(round(idf, 5) * 10 ** 5)  # convert idf to int to be used for Fraction
                for ngram, idf in json.load(f).items()
            })

    return _ngram_idf


def _bleu_handler(args):
    hypo, ref_list, weights, use_idf = args

    if type(hypo) == str:
        hypo = hypo.split(' ')
    if type(ref_list[0]) == str:
        ref_list = [r.split(' ') for r in ref_list]
    blue_func = idf_bleu if use_idf else bleu_score.sentence_bleu

    return [
        blue_func(ref_list, hypo, smoothing_function=_sf.method1, weights=w)
        for w in weights
    ]


def _ext_recall_handler(args):
    ext, hypo, ref_list, weights, use_idf = args

    if type(ext) == str:
        ext = ext.split(' ')
    if type(hypo) == str:
        hypo = hypo.split(' ')
    if type(ref_list[0]) == str:
        ref_list = [r.split(' ') for r in ref_list]

    return [
        idf_ext_recall(ref_list, hypo, ext, weights=w)
        for w in weights
    ]


class ParallelBleu:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.pool = Pool(pool_size)

        self.type_weights = [
            [1.],
            [.5, .5],
            [1 / 3, 1 / 3, 1 / 3],
            [.25, .25, .25, .25]
        ]

    def ext_recall(self, exts, hypos, refs, types=None, weights=None, use_idf=False):
        if weights and type(weights[0]) != list:
            weights = [weights]
        if not weights and types:
            weights = [self.type_weights[t-1] for t in types]

        args = [
            (e, h, rs, weights, use_idf)
            for e, h, rs in zip(exts, hypos, refs)
        ]

        return self.pool.map(_ext_recall_handler, args)

    def __call__(self, hypos, refs, types=None, weights=None, use_idf=False):
        if weights and type(weights[0]) != list:
            weights = [weights]
        if not weights and types:
            weights = [self.type_weights[t-1] for t in types]

        args = [
            (h, rs, weights, use_idf)
            for h, rs in zip(hypos, refs)
        ]

        return self.pool.map(_bleu_handler, args)


def idf_bleu(
    references,
    hypothesis,
    weights=((1., 1.), (0.25, 0.25, 0.25, 0.25)),
    smoothing_function=None,
    auto_reweigh=False,
):
    brevity_weights = None
    if isinstance(weights[0], Iterable):
        brevity_weights, weights = weights

    ngram_idf = _get_ngram_idf()

    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_mean_length = 0, 0
    hyp_idf, ref_mean_idf = 0, 0

    # For each order of ngram, calculate the numerator and
    # denominator for the corpus-level modified precision.
    for i, _ in enumerate(weights, start=1):
        numerator, denominator = modified_precision(references, hypothesis, i)
        p_numerators[i] += numerator
        p_denominators[i] += denominator

    # Calculate the hypothesis length and the closest reference length.
    # Adds them to the corpus-level hypothesis and reference counts.
    hyp_len = len(hypothesis)
    hyp_lengths += hyp_len
    # ref_lengths += bleu_score.closest_ref_length(references, hyp_len)

    # hyp_idf += mean(ngram_idf[g] for g in hypothesis) if hyp_len else 0
    # ref_mean_idf += mean(mean(
    #     ngram_idf[g] for g in ref
    # ) for ref in references)

    hyp_idf += sum(ngram_idf[g] for g in set(hypothesis)) / hyp_len if hyp_len else 0
    ref_mean_idf += mean(sum(
        ngram_idf[g] for g in set(ref)
    ) / len(ref) for ref in references)

    ref_mean_length += mean(len(ref) for ref in references)

    # Calculate corpus-level brevity penalty.
    # bp = bleu_score.brevity_penalty(ref_lengths, hyp_lengths)
    bp = brevity_penalty(ref_mean_length, hyp_len)
    idf_bp = brevity_penalty(ref_mean_idf, hyp_idf)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [
        (p_numerators[i], p_denominators[i])
        for i, _ in enumerate(weights, start=1)
    ]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    # if not smoothing_function:
    #     smoothing_function = bleu_score.SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    # p_n = smoothing_function(
    #     p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
    # )

    # custom smoothing
    epsilon = 0.1
    p_n = [
        (numerator + epsilon if numerator == 0 else numerator) / denominator
        for numerator, denominator in p_n
    ]

    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    if brevity_weights:
        idf_bp **= brevity_weights[0]
        bp **= brevity_weights[1]
    s = idf_bp * bp * math.exp(math.fsum(s))
    return s


def brevity_penalty(ref_mean_val, hyp_val):
    if hyp_val > ref_mean_val:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_val == 0:
        return 0
    else:
        return math.exp(1 - ref_mean_val / hyp_val)


def modified_precision(references, hypothesis, n):
    ngram_idf = _get_ngram_idf()

    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()

    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = math.fsum(
        count * max(ngram_idf[gram] for gram in ngram)
        for ngram, count in clipped_counts.items()
    )
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, math.fsum(
        count * max(ngram_idf[gram] for gram in ngram)
        for ngram, count in counts.items()
    ))

    return (numerator, denominator)


def idf_ext_recall(
    references,
    hypothesis,
    extraction,
    weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None,
    auto_reweigh=False,
):
    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.

    # For each order of ngram, calculate the numerator and
    # denominator for the corpus-level modified precision.
    for i, _ in enumerate(weights, start=1):
        numerator, denominator = clipped_recall(references, hypothesis, extraction, i)
        p_numerators[i] += numerator
        p_denominators[i] += denominator

    # Collects the various precision values for the different ngram orders.
    p_n = [
        (p_numerators[i], p_denominators[i])
        for i, _ in enumerate(weights, start=1)
    ]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # custom smoothing
    epsilon = 0.1
    p_n = [
        (numerator + epsilon if numerator == 0 else numerator) / denominator
        for numerator, denominator in p_n
    ]

    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s = math.exp(math.fsum(s))
    return s


def clipped_recall(references, hypothesis, extraction, n):
    ngram_idf = _get_ngram_idf()

    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    ext_counts = Counter(ngrams(extraction, n)) if len(extraction) >= n else Counter()

    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for ngram in ext_counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_ext_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in ext_counts.items()
    }

    clipped_counts = {
        ngram: min(count, clipped_ext_counts[ngram]) for ngram, count in counts.items() if ngram in clipped_ext_counts
    }

    numerator = math.fsum(
        count * max(ngram_idf[gram] for gram in ngram)
        for ngram, count in clipped_counts.items()
    )
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, math.fsum(
        count * max(ngram_idf[gram] for gram in ngram)
        for ngram, count in clipped_ext_counts.items()
    ))

    return (numerator, denominator)
