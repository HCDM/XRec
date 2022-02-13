import torch
from torch import device
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import Counter
import bottleneck as bn
import collections
import math
from nltk.translate import bleu_score
from rouge_score import rouge_scorer
from sklearn import metrics
from sklearn.metrics import ndcg_score
import random


def get_ndcg_score_pred(pred_feature_scores, target_featureids, user2featuretf, item2featuretf):
    user_featureids = set(user2featuretf.keys())
    item_featureids = set(item2featuretf.keys())
    useritem_featureids_union = user_featureids | item_featureids
    pred_scores = np.zeros((1, len(useritem_featureids_union)))
    true_scores = np.zeros((1, len(useritem_featureids_union)))
    cnt_new_gt_feature = 0
    for featureid in target_featureids:
        if featureid not in useritem_featureids_union:
            cnt_new_gt_feature += 1
    for idx, featureid in enumerate(list(useritem_featureids_union)):
        if featureid in pred_feature_scores:
            pred_scores[0][idx] = pred_feature_scores[featureid]
        if featureid in target_featureids:
            true_scores[0][idx] = 1.0
    return ndcg_score(true_scores, pred_scores), cnt_new_gt_feature


def get_ndcg_score_random(target_featureids, user2featuretf, item2featuretf):
    user_featureids = set(user2featuretf.keys())
    item_featureids = set(item2featuretf.keys())
    useritem_featureids_union = user_featureids | item_featureids
    useritem_featureids_inter = user_featureids & item_featureids
    pred_scores = np.zeros((1, len(useritem_featureids_union)))
    true_scores = np.zeros((1, len(useritem_featureids_union)))
    cnt_new_gt_feature = 0
    for featureid in target_featureids:
        if featureid not in useritem_featureids_union:
            cnt_new_gt_feature += 1
    for idx, featureid in enumerate(list(useritem_featureids_union)):
        if featureid in useritem_featureids_inter:
            pred_scores[0][idx] = random.random()
        if featureid in target_featureids:
            true_scores[0][idx] = 1.0
    return ndcg_score(true_scores, pred_scores), cnt_new_gt_feature


def get_auc_score_pred(pred_feature_scores, target_featureids, user2featuretf, item2featuretf):
    user_featureids = set(user2featuretf.keys())
    item_featureids = set(item2featuretf.keys())
    useritem_featureids_union = user_featureids | item_featureids
    pred_scores = np.zeros(len(useritem_featureids_union))
    true_scores = np.zeros(len(useritem_featureids_union))
    for idx, featureid in enumerate(list(useritem_featureids_union)):
        if featureid in pred_feature_scores:
            pred_scores[idx] = pred_feature_scores[featureid]
        if featureid in target_featureids:
            true_scores[idx] = 1.0
    if sum(true_scores) == 0:
        return 0
    fpr, tpr, thresholds = metrics.roc_curve(true_scores, pred_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def get_auc_score_random(target_featureids, user2featuretf, item2featuretf):
    user_featureids = set(user2featuretf.keys())
    item_featureids = set(item2featuretf.keys())
    useritem_featureids_union = user_featureids | item_featureids
    useritem_featureids_inter = user_featureids & item_featureids
    pred_scores = np.zeros(len(useritem_featureids_union))
    true_scores = np.zeros(len(useritem_featureids_union))
    for idx, featureid in enumerate(list(useritem_featureids_union)):
        if featureid in useritem_featureids_inter:
            pred_scores[idx] = random.random()
        if featureid in target_featureids:
            true_scores[idx] = 1.0
    if sum(true_scores) == 0:
        return 0
    fpr, tpr, thresholds = metrics.roc_curve(true_scores, pred_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def get_recall_precision_f1(preds, targets, topk=26):
    fpr, tpr, thresholds = metrics.roc_curve(targets.squeeze(), preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    _, topk_preds = torch.topk(preds, topk, dim=0)

    topk_preds = F.one_hot(topk_preds, num_classes=targets.size(0))

    topk_preds = torch.sum(topk_preds, dim=0)
    targets = targets.squeeze()

    T = (topk_preds == targets)
    P = (topk_preds == 1)

    TP = sum(T*P).item()

    precision = TP/topk

    recall = TP/(sum(targets).item())

    # f1 = 2*precision*recall/(precision+recall)

    if precision+recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0.0

    return precision, recall, f1, auc, topk_preds


def get_recall_precision_f1_sent(select_featureids, target_featureids, total_feature_num=1000):
    """ Compute the feature P/R/F1 of the features in the selected/predicted sentences
    :param: select_featureids: list of featureids of the selected predict sentences (after origin topk/3-gram/bleu)
    :param: target_featureids: list of target featureids. Target can be proxy or ground-truth.
    :param: total_feature_num: total number of features in this dataset.
    """
    # Construct the target label. 1-dim vector. The index of featureid in target will be 1, otherwise 0.
    target_labels = np.zeros(total_feature_num)
    for target_featureid in target_featureids:
        target_featureid_int = int(target_featureid)
        target_labels[target_featureid_int] = 1.0
    # Construct the 1-hot select scores. 1-dim vector. The index of featureid being select will be 1, otherwise 0.
    select_scores = np.zeros(total_feature_num)
    for select_featureid in select_featureids:
        select_featureid_int = int(select_featureid)
        select_scores[select_featureid_int] = 1.0
    # Compute AUC. It's meaning-less here to compute AUC score since we don't have the predict scores
    fpr, tpr, thresholds = metrics.roc_curve(target_labels, select_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    T = (select_scores == target_labels)
    P = (select_scores == 1.0)

    TP = sum(T*P).item()

    precision = TP/(sum(select_scores))

    recall = TP/(sum(target_labels))

    # f1 = 2*precision*recall/(precision+recall)

    if precision+recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0.0

    return precision, recall, f1, auc


def get_recall_precision_f1_random(preds, targets, topk=26):
    random_preds = torch.randn_like(preds)

    fpr, tpr, thresholds = metrics.roc_curve(targets.squeeze(), random_preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    _, topk_preds = torch.topk(random_preds, topk, dim=0)

    topk_preds = F.one_hot(topk_preds, num_classes=targets.size(0))

    topk_preds = torch.sum(topk_preds, dim=0)       # 1-dim tensor. position of value 1 is the predicted
    targets = targets.squeeze()

    T = (topk_preds == targets)
    P = (topk_preds == 1)

    TP = sum(T*P).item()

    precision = TP/topk

    recall = TP/(sum(targets).item())

    # f1 = 2*precision*recall/(precision+recall)

    if precision+recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0.0

    return precision, recall, f1, auc, topk_preds


def get_recall_precision_f1_popular(preds, targets, feature_tf, total_feature_num=1000):
    """
    "param: pred: list of predict featureids
    :param: targets: list of target featureids
    :param: feature_tf: dict, key is featureid and value is feature-tf
    """
    # Construct the target label. 1-dim vector. The index of featureid in target will be 1, otherwise 0
    target_labels = np.zeros(total_feature_num)
    for target_featureid in targets:
        target_featureid_int = int(target_featureid)
        target_labels[target_featureid_int] = 1
    # Construct the 1-hot predict scores. 1-dim vector. The index of featureid in predict will be 1, otherwise 0
    pred_scores = np.zeros(total_feature_num)
    for pred_featureid in preds:
        pred_featureid_int = int(pred_featureid)
        pred_scores[pred_featureid_int] = 1.0
    # Construct the predict scores. 1-dim vector.
    pred_scores_tf = np.zeros(total_feature_num)
    for cdd_featureid in feature_tf.keys():
        cdd_featureid_int = int(cdd_featureid)
        pred_scores_tf[cdd_featureid_int] = feature_tf[cdd_featureid]
    # Compute the AUC score
    # fpr, tpr, thresholds = metrics.roc_curve(target_labels, pred_scores, pos_label=1)
    fpr, tpr, thresholds = metrics.roc_curve(target_labels, pred_scores_tf, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    T = (pred_scores == target_labels)
    P = (pred_scores == 1)

    TP = sum(T*P).item()

    precision = TP/(sum(pred_scores))

    recall = TP/(sum(target_labels))

    # f1 = 2*precision*recall/(precision+recall)

    if precision+recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0.0

    return precision, recall, f1, auc


def get_recall_precision_f1_gt(preds_scores, targets, featureids, topk=26, total_feature_num=1000):
    """
    "param: preds_scores: predict scores
    :param: targets: list of target featureids
    :param: feature_tf: dict, key is featureid and value is feature-tf
    """
    # Construct the target label. 1-dim vector. The index of featureid in target will be 1, otherwise 0
    target_labels = np.zeros(total_feature_num)
    for target_featureid in targets:
        target_featureid_int = int(target_featureid)
        target_labels[target_featureid_int] = 1
    # Construct the 1-hot predict scores. 1-dim vector. The index of featureid in predict will be 1, otherwise 0
    preds_scores_model = np.zeros(total_feature_num)
    for i in range(len(featureids)):
        preds_scores_model[int(featureids[i])] = preds_scores[i]
    # Compute the AUC score
    # fpr, tpr, thresholds = metrics.roc_curve(target_labels, pred_scores, pos_label=1)
    fpr, tpr, thresholds = metrics.roc_curve(target_labels, preds_scores_model, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # get the topk-feature of the model's predicts
    topk_logits, topk_preds = torch.topk(preds_scores, topk, dim=0)
    # print("topk_preds: ", topk_preds)
    # topk_preds_featureid_index = featureids[topk_preds]
    topk_preds_featureid_index = [featureids[idx.item()] for idx in topk_preds]
    # Construct the target label. 1-dim vector. The index of featureid in target will be 1, otherwise 0
    topk_pred_labels = np.zeros(total_feature_num)
    for pred_featureid in topk_preds_featureid_index:
        topk_pred_featureid_int = int(pred_featureid)
        topk_pred_labels[topk_pred_featureid_int] = 1

    T = (topk_pred_labels == target_labels)
    P = (topk_pred_labels == 1)

    TP = sum(T*P).item()

    precision = TP/(sum(topk_pred_labels))

    recall = TP/(sum(target_labels))

    # f1 = 2*precision*recall/(precision+recall)

    if precision+recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0.0

    return precision, recall, f1, auc, topk_preds_featureid_index, topk_logits


def get_recall_precision_f1_gt_random(preds_scores, targets, featureids, topk=26, total_feature_num=1000):
    """
    "param: preds_scores: predict scores
    :param: targets: list of target featureids
    :param: feature_tf: dict, key is featureid and value is feature-tf
    """
    # Construct the target label. 1-dim vector. The index of featureid in target will be 1, otherwise 0
    target_labels = np.zeros(total_feature_num)
    for target_featureid in targets:
        target_featureid_int = int(target_featureid)
        target_labels[target_featureid_int] = 1

    # Construct the random predict scores. 1-dim vector.
    random_preds_scores = torch.randn_like(preds_scores)
    preds_scores_model = np.zeros(total_feature_num)
    for i in range(len(featureids)):
        preds_scores_model[int(featureids[i])] = random_preds_scores[i]

    # Compute the AUC score
    fpr, tpr, thresholds = metrics.roc_curve(target_labels, preds_scores_model, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # get the topk-feature of the model's predicts
    _, topk_preds = torch.topk(random_preds_scores, topk, dim=0)
    # print("topk_preds: ", topk_preds)
    # topk_preds_featureid_index = featureids[topk_preds]
    topk_preds_featureid_index = [featureids[idx.item()] for idx in topk_preds]
    # Construct the target label. 1-dim vector. The index of featureid in target will be 1, otherwise 0
    topk_pred_labels = np.zeros(total_feature_num)
    for pred_featureid in topk_preds_featureid_index:
        topk_pred_featureid_int = int(pred_featureid)
        topk_pred_labels[topk_pred_featureid_int] = 1

    T = (topk_pred_labels == target_labels)
    P = (topk_pred_labels == 1)

    TP = sum(T*P).item()

    precision = TP/(sum(topk_pred_labels))

    recall = TP/(sum(target_labels))

    # f1 = 2*precision*recall/(precision+recall)

    if precision+recall != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = 0.0

    return precision, recall, f1, auc, topk_preds_featureid_index


def get_feature_recall_precision(pred, ref):
    '''
    :param pred: list of features which appears in the predicted sentences
    :param ref: list of features that are in the reference sentences

    :return recall: the recall score of the pred features
    :return precision: the precision score of the pred features
    '''
    recall = 0.0
    precision = 0.0
    recall_num = 0
    precision_num = 0
    pred_count = Counter(pred)
    ref_count = Counter(ref)
    for key, value in ref_count.items():
        if key in pred_count:
            recall_num += min(value, pred_count[key])
    for key, value in pred_count.items():
        if key in ref_count:
            precision_num += min(value, ref_count[key])
    recall = recall_num / len(ref)
    precision = precision_num / len(pred)
    return recall, precision


def get_feature_recall_precision_rouge(pred, ref):
    ''' using rouge-score to compute feature precision/recall
    :param pred: list of features which appears in the predicted sentences
    :param ref: list of features that are in the reference sentences

    :return recall: the recall score of the pred features
    :return precision: the precision score of the pred features
    '''
    pred_concat = " ".join(pred)
    ref_concat = " ".join(pred)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(ref_concat, pred_concat)
    return scores['rouge1'].recall, scores['rouge1'].precision


def get_example_recall_precision(pred, target, k=1):
    recall = 0.0
    precision = 0.0

    pred = list(pred.numpy())
    # target = list(target.numpy())

    true_pos = set(target) & set(pred)
    true_pos_num = len(true_pos)

    target_num = len(target)
    recall = true_pos_num*1.0/target_num

    precision = true_pos_num*1.0/k

    return recall, precision


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            # print(ngram)
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, hypothese_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        hypothese_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        BLEU score
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    reference_length = 0
    hypothesis_length = 0

    for (references, hypothesis) in zip(reference_corpus, hypothese_corpus):
        reference_length += min(len(r) for r in references)
        hypothesis_length += len(hypothesis)

        merged_ref_ngram_counts = Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)

        hyp_ngram_counts = _get_ngrams(hypothesis, max_order)
        overlap = hyp_ngram_counts & merged_ref_ngram_counts

        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]

        for order in range(1, max_order+1):
            possible_matches = len(hypothesis)-order+1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i]+1.0)/(possible_matches_by_order[i]+1.0))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i])/possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0/max_order)*math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(hypothesis_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1-1.0/ratio)

    bleu = geo_mean*bp

    return bleu


# def compute_bleu(references, hypotheses, max_order=4, smooth=False):
#     matches_by_order = [0]*max_order
#     possible_matches_by_order = [0]*max_order

#     reference_length = 0
#     hypothesis_length = 0

#     for (reference, hypothesis) in zip(references, hypotheses):
#         reference_length += min(len(r) for r in references)
#         hypothesis_length += len(hypothesis)

#         merged_ref_ngram_counts = collections.Counter()
#         for reference in references:
#             merged_ref_ngram_counts |= _get_ngrams(reference, max_order)

#         hyp_ngram_counts = _get_ngrams(hypothesis, max_order)
#         overlap = hyp_ngram_counts & merged_ref_ngram_counts

#         for ngram in overlap:
#             matches_by_order[len(ngram)-1] += overlap[ngram]

#         for order in range(1, max_order+1):
#             possible_matches = len(hypothesis)-order+1
#             if possible_matches > 0:
#                 possible_matches_by_order[order-1] += possible_matches

#     precisions = [0]*max_order
#     for i in range(0, max_order):
#         if smooth:
#             precisions[i] = ((matches_by_order[i]+1.0)/(possible_matches_by_order[i]+1.0))
#         else:
#             if possible_matches_by_order[i] > 0:
#                 precisions[i] = (float(matches_by_order[i])/possible_matches_by_order[i])
#             else:
#                 precisions[i] = 0.0

#     if min(precisions) > 0:
#         p_log_sum = sum((1.0/max_order)*math.log(p) for p in precisions)
#         geo_mean = math.exp(p_log_sum)
#     else:
#         geo_mean = 0

#     ratio = float(hypothesis_length) / reference_length

#     if ratio > 1.0:
#         bp = 1.0
#     else:
#         bp = math.exp(1-1.0/ratio)

#     bleu = geo_mean*bp

#     return bleu


def get_sentence_bleu(references, hypotheses, types=[1, 2, 3, 4]):
    """ This is used to compute sentence-level bleu
    param: references: list of reference sentences, each reference sentence is a list of tokens
    param: hypoyheses: hypotheses sentences, this is a list of tokenized tokens
    return:
        bleu-1, bleu-2, bleu-3, bleu-4
    """
    type_weights = [[1.0, 0., 0., 0],
                    [0.5, 0.5, 0.0, 0.0],
                    [1.0/3, 1.0/3, 1.0/3, 0.0],
                    [0.25, 0.25, 0.25, 0.25]]
    sf = bleu_score.SmoothingFunction()
    bleu_1_score = bleu_score.sentence_bleu(
        references, hypotheses, smoothing_function=sf.method1, weights=type_weights[0])
    bleu_2_score = bleu_score.sentence_bleu(
        references, hypotheses, smoothing_function=sf.method1, weights=type_weights[1])
    bleu_3_score = bleu_score.sentence_bleu(
        references, hypotheses, smoothing_function=sf.method1, weights=type_weights[2])
    bleu_4_score = bleu_score.sentence_bleu(
        references, hypotheses, smoothing_function=sf.method1, weights=type_weights[3])
    return bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score


def get_bleu(references, hypotheses, types=[1, 2, 3, 4]):
    type_weights = [[1.0, 0., 0., 0],
                    [0.5, 0.5, 0.0, 0.0],
                    [1.0/3, 1.0/3, 1.0/3, 0.0],
                    [0.25, 0.25, 0.25, 0.25]]

    totals = [0.0] * len(types)

    sf = bleu_score.SmoothingFunction()

    num = 0

    for (reference, hypothesis) in zip(references, hypotheses):

        for j, t in enumerate(types):
            weights = type_weights[t-1]
            totals[j] += bleu_score.sentence_bleu(reference, hypothesis, smoothing_function=sf.method1, weights=weights)

        num += 1.0

    totals = [total/num for total in totals]

    return totals
