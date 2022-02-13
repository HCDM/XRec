import os
import sys
import json
import pickle
import random
import datetime
import time
import argparse

# Multi-processing
# import ray
from multiprocessing import Pool

# scientific and machine learning toolkits
import math
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import torch

# NLP metrics and Feature Prediction metrics
from rouge import Rouge
from nltk.translate import bleu_score
from metric import compute_bleu, get_bleu, get_sentence_bleu
from metric import get_example_recall_precision, get_feature_recall_precision, get_recall_precision_f1, get_recall_precision_f1_random

# ILP, we use Gurobi
import gurobipy as gp
from gurobipy import GRB

sys.path.append(os.getcwd())

label_format = 'soft_label'
use_ILP = True                                  # using ILP to select sentences

save_hyps_refs_pool = True
compute_rouge_score = True
compute_bleu_score = True

ILP_top_relevance_score_thres = 100


class EVAL_ILP(object):
    def __init__(self, args):
        super().__init__()

        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_eval_output_path = args.eval_output_path
        self.m_data_dir = args.data_dir
        self.m_dataset = args.data_set
        self.m_dataset_name = args.data_name
        self.select_s_topk = args.select_topk_s
        self.m_print_frequency = args.print_freq
        self.m_num_threads = args.num_threads
        self.m_bias_size = args.bias_lines
        self.m_select_top = args.select_top
        self.m_select_top_lines_num = args.select_lines
        self.m_alpha = args.alpha
        self.m_filter_method = args.filter

        print("Data directory: {}".format(self.m_data_dir))
        print("Evaluation results are saved under dir: {}".format(self.m_eval_output_path))
        print("Dataset: {0} \t Label: {1}".format(self.m_dataset, label_format))
        self.model_pred_DIR = '../data_postprocess/{}'.format(self.m_dataset)
        self.model_pred_DIR = os.path.join(self.model_pred_DIR, self.m_model_file.split('.')[0])
        print("Prediction files are saved under the directory: {}".format(self.model_pred_DIR))
        self.model_pred_file = os.path.join(self.model_pred_DIR, 'model_pred_multiline.json')
        self.sid2swords_file = os.path.join(self.model_pred_DIR, 'sid2swords.pickle')
        self.sid2sentid_file = os.path.join(self.model_pred_DIR, 'sid2sentid.pickle')
        with open(self.sid2swords_file, 'rb') as handle:
            print("Load file: {}".format(self.sid2swords_file))
            self.m_sid2swords = pickle.load(handle)
        with open(self.sid2sentid_file, 'rb') as handle:
            print("Load file: {}".format(self.sid2sentid_file))
            self.m_sid2sentid = pickle.load(handle)
        # Post-processing methods
        print("--"*10+"post-processing method"+"--"*10)
        print("Using ILP for post-processing.")
        # Pool size
        if ILP_top_relevance_score_thres is not None:
            print("Only use the top {} predicted sentences for each user-item pair.".format(
                ILP_top_relevance_score_thres
            ))
        else:
            print("Use all cdd sentences for each user-item pair.")
        if self.m_filter_method is None:
            print("Candidate sentences from the union of user-side and item-side sentences.")
        elif self.m_filter_method == "item":
            print("Candidate sentences only from item-side sentences.")
        elif self.m_filter_method == "item_feature":
            print("Candidate sentences from item-side sentences and user-side sentences which only contain item-side features.")
        else:
            print("Filter method: {} not supported, exit.".format(self.m_filter_method))
            exit()
        # Baselines
        print("--"*10+"sentence predict score"+"--"*10)
        print("hypothesis selected based on original score and filtering methods.")
        # need to load some mappings
        print("--"*10+"load preliminary mappings"+"--"*10)
        id2feature_file = os.path.join(self.m_data_dir, 'train/feature/id2feature.json')
        feature2id_file = os.path.join(self.m_data_dir, 'train/feature/feature2id.json')
        trainset_id2sent_file = os.path.join(self.m_data_dir, 'train/sentence/id2sentence.json')
        testset_id2sent_file = os.path.join(self.m_data_dir, 'test/sentence/id2sentence.json')
        testset_useritem_cdd_withproxy_file = os.path.join(self.m_data_dir, 'test/useritem2sentids_withproxy.json')
        trainset_user2sentid_file = os.path.join(self.m_data_dir, 'train/user/user2sentids.json')
        trainset_item2sentid_file = os.path.join(self.m_data_dir, 'train/item/item2sentids.json')
        trainset_senttfidf_embed_file = os.path.join(self.m_data_dir, 'train/sentence/tfidf_sparse_clean.npz')
        # Load the combined train/test set
        trainset_combined_file = os.path.join(self.m_data_dir, 'train_combined.json')
        testset_combined_file = os.path.join(self.m_data_dir, 'test_combined.json')
        # Load test-set after item-side-feature filtering
        testset_item_feature_file = os.path.join(self.m_data_dir, 'test/useritem2sentids_item_feat_test_multilines.json')
        with open(id2feature_file, 'r') as f:
            print("Load file: {}".format(id2feature_file))
            self.d_id2feature = json.load(f)
        with open(feature2id_file, 'r') as f:
            print("Load file: {}".format(feature2id_file))
            self.d_feature2id = json.load(f)
        with open(trainset_id2sent_file, 'r') as f:
            print("Load file: {}".format(trainset_id2sent_file))
            self.d_trainset_id2sent = json.load(f)
        with open(testset_id2sent_file, 'r') as f:
            print("Load file: {}".format(testset_id2sent_file))
            self.d_testset_id2sent = json.load(f)
        with open(testset_useritem_cdd_withproxy_file, 'r') as f:
            print("Load file: {}".format(testset_useritem_cdd_withproxy_file))
            self.d_testset_useritem_cdd_withproxy = json.load(f)
        # Load trainset user to sentence id dict
        with open(trainset_user2sentid_file, 'r') as f:
            print("Load file: {}".format(trainset_user2sentid_file))
            self.d_trainset_user2sentid = json.load(f)
        # Load trainset item to sentence id dict
        with open(trainset_item2sentid_file, 'r') as f:
            print("Load file: {}".format(trainset_item2sentid_file))
            self.d_trainset_item2sentid = json.load(f)
        # Load the sentence tf-idf sparse matrix
        print("Load file: {}".format(trainset_senttfidf_embed_file))
        self.train_sent_tfidf_sparse = sp.load_npz(trainset_senttfidf_embed_file)
        print("Shape of the tf-idf matrix: {}".format(self.train_sent_tfidf_sparse.shape))
        # Get trainset sid2featuretf dict
        # Load train/test combined review for standard evaluation
        self.d_trainset_combined = dict()
        with open(trainset_combined_file, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                userid = line_data['user']
                itemid = line_data['item']
                review_text = line_data['review']
                if userid not in self.d_trainset_combined:
                    self.d_trainset_combined[userid] = dict()
                    self.d_trainset_combined[userid][itemid] = review_text
                else:
                    assert itemid not in self.d_trainset_combined[userid]
                    self.d_trainset_combined[userid][itemid] = review_text
        self.d_testset_combined = dict()
        with open(testset_combined_file, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                userid = line_data['user']
                itemid = line_data['item']
                review_text = line_data['review']
                if userid not in self.d_testset_combined:
                    self.d_testset_combined[userid] = dict()
                    self.d_testset_combined[userid][itemid] = review_text
                else:
                    assert itemid not in self.d_testset_combined[userid]
                    self.d_testset_combined[userid][itemid] = review_text
        self.d_testset_item_feature_filtered = dict()
        with open(testset_item_feature_file, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                userid = str(line_data['user_id'])
                itemid = str(line_data['item_id'])
                cdd_sentids = line_data['candidate']
                assert isinstance(cdd_sentids[0], str)
                if userid not in self.d_testset_item_feature_filtered:
                    self.d_testset_item_feature_filtered[userid] = dict()
                    self.d_testset_item_feature_filtered[userid][itemid] = cdd_sentids
                else:
                    assert itemid not in self.d_testset_item_feature_filtered[userid]
                    self.d_testset_item_feature_filtered[userid][itemid] = cdd_sentids

    def f_eval(self):
        """
        1. Save Predict/Selected sentences and Reference sentences to compute BLEU using the perl script.
        2. Add mojority vote based baselines.
        3. Seperate code chunks into functions.
        """
        print('--'*10)
        s_topk = self.select_s_topk
        print("Number of topk selected sentences: {}".format(s_topk))
        print("Alpha: {}".format(self.m_alpha))

        self.line_data_whole = []
        # Load the model's predictions from file line-by-line
        with open(self.model_pred_file, 'r') as f:
            print("Read file: {} line-by-line".format(self.model_pred_file))
            for line in f:
                line_data = json.loads(line)
                self.line_data_whole.append(line_data)
        whole_size = len(self.line_data_whole)
        print("Total number of lines: {}".format(whole_size))

        if self.m_select_top:
            whole_size = self.m_select_top_lines_num
            print("For brevity, only select the top {} lines".format(whole_size))

        # Add a bias at the start
        bias_size = self.m_bias_size
        print("We start at line: {}".format(bias_size))

        # Start Multi-processing
        print("Pool num: {}".format(self.m_num_threads))
        # Position 0 is the thread idx
        idx_list_pool = [[i] for i in range(self.m_num_threads)]
        start_idx = 0
        end_idx = 0
        window_size = int(math.ceil(whole_size / self.m_num_threads))
        print("Window size: {}".format(window_size))
        for i in range(self.m_num_threads):
            start_idx = i * window_size + bias_size
            end_idx = (i+1) * window_size + bias_size
            if end_idx > (whole_size + bias_size):
                end_idx = whole_size + bias_size
            idx_list_pool[i].extend(list(range(start_idx, end_idx)))
            print("[Thread {}] start from line {}, end at line {}.".format(
                idx_list_pool[i][0], idx_list_pool[i][1], idx_list_pool[i][-1]
                ))
        with Pool(self.m_num_threads) as p:
            results_pool = p.map(self.Compute_ILP_Pool, idx_list_pool)

        # save results into file
        refs_file = os.path.join(self.m_eval_output_path, 'reference.txt')
        hyps_file = os.path.join(self.m_eval_output_path, 'hypothesis.txt')
        refs_json_file = os.path.join(self.m_eval_output_path, 'refs.json')
        hyps_json_file = os.path.join(self.m_eval_output_path, 'hyps.json')
        cnt_line = 0
        with open(refs_file, 'w') as f_r, open(refs_json_file, 'w') as f_rj:
            with open(hyps_file, 'w') as f_h, open(hyps_json_file, 'w') as f_hj:
                for result in results_pool:
                    for result_line in result:
                        line_idx = result_line[0]
                        user_id = result_line[1]
                        item_id = result_line[2]
                        refs_text = result_line[3]
                        hyps_text = result_line[4]
                        hyps_sids = result_line[5]
                        hyps_cdd_sids = result_line[6]
                        try:
                            assert line_idx == cnt_line
                        except AssertionError:
                            print("whole line idx {} and pool line idx {} not aligned!".format(
                                cnt_line, line_idx
                            ))
                        # write reference raw text
                        f_r.write(refs_text)
                        f_r.write("\n")
                        # write reference raw text with user/item id
                        ref_json_dict = {'user': user_id, 'item': item_id, 'text': refs_text}
                        json.dump(ref_json_dict, f_rj)
                        f_rj.write("\n")
                        # write hypothesis raw text
                        f_h.write(hyps_text)
                        f_h.write("\n")
                        # write hypothesis raw text with user/item id
                        hyp_json_dict = {
                            'user': user_id, 'item': item_id, 'text': hyps_text,
                            'sids': hyps_sids, 'cdd_sids': hyps_cdd_sids
                        }
                        json.dump(hyp_json_dict, f_hj)
                        f_hj.write("\n")
                        cnt_line += 1
        print("Finish Writing {} lines of references and hypothesis into file.".format(cnt_line))
        print("hypothesis: {}".format(hyps_file))

    def Compute_ILP_Pool(self, idx_list):
        """
        :param: idx_list:    a list of line idx. NOTE: position 0 is the thread idx
        """
        rouge_1_f_list, rouge_1_p_list, rouge_1_r_list = [], [], []
        rouge_2_f_list, rouge_2_p_list, rouge_2_r_list = [], [], []
        rouge_l_f_list, rouge_l_p_list, rouge_l_r_list = [], [], []
        bleu_list, bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], [], []
        rouge = Rouge()
        num_empty_hyps = 0
        num_too_long_hyps = 0
        result_pool = []
        thread_idx = idx_list[0]
        print("[Thread {}] From line {} to line {} (totally {} lines).".format(
            thread_idx, idx_list[1], idx_list[-1], len(idx_list[1:])))
        # Loop through the idx list in this Pool
        for i, idx in enumerate(idx_list[1:]):
            line_data_i = self.line_data_whole[idx]
            user_id = line_data_i['user']
            item_id = line_data_i['item']
            assert isinstance(user_id, str)
            assert isinstance(item_id, str)
            cdd_sids_i = line_data_i['cdd_sids']
            cdd_sids2logits_i = line_data_i['cdd_sids2logits']
            # Filter candidate sids
            if self.m_filter_method is None:
                pass
            elif self.m_filter_method == 'item':
                cdd_sids_i_filter = []
                cdd_sids2logits_i_filter = {}
                item_side_sentids = set(self.d_trainset_item2sentid[item_id])
                for sid in cdd_sids_i:
                    if str(sid) in item_side_sentids:
                        cdd_sids_i_filter.append(sid)
                        cdd_sids2logits_i_filter[str(sid)] = cdd_sids2logits_i[str(sid)]
                assert len(cdd_sids2logits_i_filter) == len(cdd_sids_i_filter)
                assert len(cdd_sids_i_filter) > 0
            elif self.m_filter_method == 'item_feature':
                cdd_sids_i_filter = []
                cdd_sids2logits_i_filter = {}
                item_feature_sentids = set(self.d_testset_item_feature_filtered[user_id][item_id])
                for sid in cdd_sids_i:
                    if str(sid) in item_feature_sentids:
                        cdd_sids_i_filter.append(sid)
                        cdd_sids2logits_i_filter[str(sid)] = cdd_sids2logits_i[str(sid)]
                assert len(cdd_sids2logits_i_filter) == len(cdd_sids_i_filter)
                assert len(cdd_sids_i_filter) > 0
            else:
                print("Filter method: {} not supported, exit.".format(self.m_filter_method))
                exit()

            # ILP Solver
            # s_topk_logits: logits of the select sentences, list
            # s_pred_sids:   sids of the select sentences,   list
            # s_cdd_sent_sids: cdd sent sids (top-pool may apply) from which ILP select sids, list
            if self.m_filter_method is None:
                s_topk_logits, s_pred_sids, s_cdd_sent_sids, _ = self.ILP_sent_prediction(
                    user_id, item_id, cdd_sids_i, cdd_sids2logits_i,
                    topk=self.select_s_topk, alpha=self.m_alpha,
                    thres=ILP_top_relevance_score_thres,
                    line_idx=idx, thread_idx=thread_idx
                )
            else:
                s_topk_logits, s_pred_sids, s_cdd_sent_sids, _ = self.ILP_sent_prediction(
                    user_id, item_id, cdd_sids_i_filter, cdd_sids2logits_i_filter,
                    topk=self.select_s_topk, alpha=self.m_alpha,
                    thres=ILP_top_relevance_score_thres,
                    line_idx=idx, thread_idx=thread_idx
                )
            # Combine the Hypothesis sentences
            hyps_i_list = []
            for sid_k in s_pred_sids:
                hyps_i_list.append(self.m_sid2swords[sid_k])
            hyps_i = " ".join(hyps_i_list)
            # Get the true combined reference text in data
            true_combined_ref = self.d_testset_combined[user_id][item_id]

            # save the results to list
            # idx: whole eval-set's line idx (start from 0)
            result_pool.append(
                [idx, user_id, item_id, true_combined_ref, hyps_i, s_pred_sids, s_cdd_sent_sids]
            )

            if (i+1) % self.m_print_frequency == 0:
                print("[Thread {}] Finish {} lines".format(thread_idx, i+1))

            if save_hyps_refs_pool:
                # Compute ROUGE/BLEU score
                # Save refs and selected hyps into file
                refs_file = os.path.join(
                    self.m_eval_output_path, 'reference_{}.txt'.format(thread_idx)
                )
                hyps_file = os.path.join(
                    self.m_eval_output_path, 'hypothesis_{}.txt'.format(thread_idx)
                )
                refs_json_file = os.path.join(
                    self.m_eval_output_path, 'refs_{}.json'.format(thread_idx)
                )
                hyps_json_file = os.path.join(
                    self.m_eval_output_path, 'hyps_{}.json'.format(thread_idx)
                )
                # write reference raw text
                with open(refs_file, 'a') as f_ref:
                    # f_ref.write(refs_j)
                    f_ref.write(true_combined_ref)
                    f_ref.write("\n")
                # write reference raw text with user/item id
                with open(refs_json_file, 'a') as f_ref_json:
                    cur_ref_json = {
                        'user': user_id, 'item': item_id, 'text': true_combined_ref
                    }
                    json.dump(cur_ref_json, f_ref_json)
                    f_ref_json.write("\n")
                # write hypothesis raw text
                with open(hyps_file, 'a') as f_hyp:
                    f_hyp.write(hyps_i)
                    f_hyp.write("\n")
                # write hypothesis raw text with user/item id
                with open(hyps_json_file, 'a') as f_hyp_json:
                    cur_hyp_json = {
                        'user': user_id, 'item': item_id, 'text': hyps_i
                    }
                    json.dump(cur_hyp_json, f_hyp_json)
                    f_hyp_json.write("\n")

            if compute_rouge_score:
                try:
                    scores_i = rouge.get_scores(hyps_i, true_combined_ref, avg=True)
                except Exception:
                    if hyps_i == '':
                        hyps_i = '<unk>'
                        scores_i = rouge.get_scores(hyps_i, true_combined_ref, avg=True)
                        num_empty_hyps += 1
                    else:
                        # hyps may be too long, then we truncate it to be half
                        hyps_i_trunc = " ".join(hyps_i_list[0:int(self.select_s_topk/2)])
                        scores_i = rouge.get_scores(hyps_i_trunc, true_combined_ref, avg=True)
                        num_too_long_hyps += 1

                rouge_1_f_list.append(scores_i["rouge-1"]["f"])
                rouge_1_r_list.append(scores_i["rouge-1"]["r"])
                rouge_1_p_list.append(scores_i["rouge-1"]["p"])

                rouge_2_f_list.append(scores_i["rouge-2"]["f"])
                rouge_2_r_list.append(scores_i["rouge-2"]["r"])
                rouge_2_p_list.append(scores_i["rouge-2"]["p"])

                rouge_l_f_list.append(scores_i["rouge-l"]["f"])
                rouge_l_r_list.append(scores_i["rouge-l"]["r"])
                rouge_l_p_list.append(scores_i["rouge-l"]["p"])

            if compute_bleu_score:
                bleu_scores_i = compute_bleu([[true_combined_ref.split()]], [hyps_i.split()])
                bleu_list.append(bleu_scores_i)

                bleu_1_i, bleu_2_i, bleu_3_i, bleu_4_i = get_sentence_bleu(
                    [true_combined_ref.split()], hyps_i.split())

                bleu_1_list.append(bleu_1_i)
                bleu_2_list.append(bleu_2_i)
                bleu_3_list.append(bleu_3_i)
                bleu_4_list.append(bleu_4_i)

        print("[Thread {0}] Number of empty hypothesis: {1}, Number of too long hypothesis: {2}".format(
                thread_idx, num_empty_hyps, num_too_long_hyps))

        if compute_rouge_score:
            print("[Thread %d] rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f" % (
                thread_idx,
                np.mean(rouge_1_f_list),
                np.mean(rouge_1_p_list),
                np.mean(rouge_1_r_list),
                np.mean(rouge_2_f_list),
                np.mean(rouge_2_p_list),
                np.mean(rouge_2_r_list),
                np.mean(rouge_l_f_list),
                np.mean(rouge_l_p_list),
                np.mean(rouge_l_r_list)))

        if compute_bleu_score:
            print("[Thread %d] bleu:%.4f, bleu-1:%.4f, bleu-2:%.4f, bleu-3:%.4f, bleu-4:%.4f" % (
                thread_idx,
                np.mean(bleu_list),
                np.mean(bleu_1_list),
                np.mean(bleu_2_list),
                np.mean(bleu_3_list),
                np.mean(bleu_4_list)))

        print("[Thread {}] Finish!".format(thread_idx))

        return result_pool

    def ILP_sent_prediction(self, user_id, item_id, cdd_sids, cdd_sids2logits, topk=3, alpha=1.0, thres=None, line_idx=0, thread_idx=0):
        """
        :param: user_id,                user id.
        :param: item_id,                item id.
        :param: cdd_sids,               sentence's sid.
        :param: cdd_sids2logits,        sentence's sid to logit.
        :param: topk,                   number of select sentences.
        :param: alpha,                  trade-off parameter between 2 costs.
        :param: thres,                  only use thres number of top predicted sentences.
                                        None: not use top filtering.
        :return
        """
        ILP_compute_scores = 0.0

        # Get the s_logits for each sid
        assert len(cdd_sids) == len(cdd_sids2logits)
        cdd_slogits = [cdd_sids2logits[str(sid)] for sid in cdd_sids]
        cdd_slogits = torch.tensor(cdd_slogits)
        sorted_s_logits, sorted_idx = cdd_slogits.sort(descending=True)
        cdd_sent_sids = []
        cdd_sent_sentids = []
        cdd_sent_sentids_int = []
        cdd_sent_logits = []
        log_file = os.path.join(
            self.m_eval_output_path, 'log_{}.txt'.format(thread_idx)
        )

        if thres is None:
            # Select sentences from all cdd sentences
            for j in range(len(cdd_sids)):
                sid_j = cdd_sids[j]
                sentid_j = self.m_sid2sentid[sid_j]
                sentid_j_int = int(sentid_j)
                sent_pred_score_j = cdd_slogits[j].item()
                cdd_sent_sids.append(sid_j)
                cdd_sent_sentids.append(sentid_j)
                cdd_sent_sentids_int.append(sentid_j_int)
                cdd_sent_logits.append(sent_pred_score_j)
        else:
            for j in sorted_idx:
                sid_j = cdd_sids[j.item()]
                sentid_j = self.m_sid2sentid[sid_j]
                sentid_j_int = int(sentid_j)
                sent_pred_score_j = cdd_slogits[j.item()].item()
                if sent_pred_score_j <= 0.0:
                    break
                else:
                    if len(cdd_sent_logits) > 0:
                        try:
                            assert cdd_sent_logits[-1] >= sent_pred_score_j
                        except AssertionError:
                            exit()
                cdd_sent_sids.append(sid_j)
                cdd_sent_sentids.append(sentid_j)
                cdd_sent_sentids_int.append(sentid_j_int)
                cdd_sent_logits.append(sent_pred_score_j)
                if len(cdd_sent_sids) == thres:
                    break

        # Check how many selected cdd sentences after top-truncating
        try:
            assert len(cdd_sent_sids) >= topk
        except AssertionError:
            topk = len(cdd_sent_sids)
        # Get the cosine similarity matrix for these cdd sentences
        cosine_sim_maxtrix = cosine_similarity(self.train_sent_tfidf_sparse[cdd_sent_sentids_int])
        cosine_sim_upper = np.triu(cosine_sim_maxtrix, 1)
        cdd_sent_pred_scores = np.array(cdd_sent_logits)
        pool_size = len(cdd_sent_sids)
        # Create a new model for ILP
        ILP_m = gp.Model("graph2x_ilp_{}".format(thread_idx))
        ILP_m.Params.LogToConsole = 0
        # ILP_m.setParam(GRB.Param.TimeLimit, 1000.0)
        # Create variables
        X = ILP_m.addMVar(shape=pool_size, vtype=GRB.BINARY, name="X")
        Y = ILP_m.addMVar(shape=(pool_size, pool_size), vtype=GRB.BINARY, name="Y")
        # Construct Objective
        ILP_m.setObjective(
            (cdd_sent_pred_scores @ X) - alpha * sum(
                Y[i_m] @ cosine_sim_upper[i_m] for i_m in range(pool_size)),
            GRB.MAXIMIZE
        )
        # ILP_m.setObjective(
        #     (cdd_sent_pred_scores @ X) - alpha * sum(
        #         Y[i_m][j_m] * cosine_sim_upper[i_m][j_m] for i_m in range(pool_size) for j_m in range(i_m+1, pool_size)),
        #     GRB.MAXIMIZE
        # )
        # Add the sum constrain of X
        ones_i = np.ones(len(cdd_sent_sids))
        ILP_m.addConstr(ones_i @ X == topk, name="c0")
        # Add the inequality constraints
        # usage: https://www.gurobi.com/documentation/9.1/refman/py_model_addconstrs.html
        ILP_m.addConstrs(
            ((X[i_m] + X[j_m]) <= (Y[i_m][j_m] + 1) for i_m in range(pool_size) for j_m in range(i_m+1, pool_size)), name='c1'
        )
        # Add the sum constraint of Y
        E_num = topk * (topk - 1) / 2

        ILP_m.addConstr(
            sum(sum(Y[i_m][i_m+1:]) for i_m in range(pool_size)) == E_num, name="c2"
        )
        # ILP_m.addConstr(
        #     sum(Y[i_m][j_m] for i_m in range(pool_size) for j_m in range(i_m+1, pool_size)) == E_num, name="c2"
        # )

        # Optimize model
        ILP_m.optimize()
        # Get the obj value
        ILP_compute_scores = ILP_m.objVal
        # Get the X variables' value
        select_sent_idx_i = np.where(X.X == 1.0)[0].tolist()
        # Check the Y variables value
        try:
            for i_m in range(pool_size):
                for j_m in range(i_m+1, pool_size):
                    if X.X[i_m] == 1.0 and X.X[j_m] == 1.0:
                        assert Y.X[i_m][j_m] == 1.0
                    else:
                        assert Y.X[i_m][j_m] == 0.0
        except AssertionError:
            # Add log of Y not aligned with X
            print("At line {}, Y not aligned with X".format(line_idx))
            with open(log_file, 'a') as f_log:
                f_log.write("X and Y in ILP not align error at line: {0} (user_id: {1} item_id: {2})\n".format(
                    line_idx, user_id, item_id))

        # Get the select sids and sentids
        select_sids_i = [cdd_sent_sids[idx] for idx in select_sent_idx_i]
        select_sentids_i = [cdd_sent_sentids[idx] for idx in select_sent_idx_i]
        select_sent_logits_i = [cdd_sent_logits[idx] for idx in select_sent_idx_i]
        # Clean up the model
        ILP_m.dispose()
        # end_i = time.process_time()
        # print("{} for 1 user-item review ({} cdd sents)".format(end_i-start_i, num_sent_i))
        # NOTE: if we can not solve the ILP, we use greedy (this should occur in a very rare situation)
        if len(select_sent_idx_i) == 0:
            with open(log_file, 'a') as f_log:
                f_log.write("Using Greedy Result at line: {0} (user_id: {1} item_id: {2})\n".format(
                    line_idx, user_id, item_id))
            return self.ILP_sent_prediction_greedy(
                user_id=user_id, item_id=item_id,
                cdd_sids=cdd_sids, cdd_sids2logits=cdd_sids2logits,
                topk=topk, alpha=alpha, thres=thres,
                line_idx=line_idx, thread_idx=thread_idx
            )

        return select_sent_logits_i, select_sids_i, cdd_sent_sids, ILP_compute_scores

    def ILP_sent_prediction_greedy(self, user_id, item_id, cdd_sids, cdd_sids2logits, topk=3, alpha=1.0, thres=None, line_idx=0, thread_idx=0):
        """
        :param: user_id,                user id.
        :param: item_id,                item id.
        :param: cdd_sids,               sentence's sid.
        :param: cdd_sids2logits,        sentence's sid to logit.
        :param: topk,                   number of select sentences.
        :param: alpha,                  trade-off parameter between 2 costs.
        :param: thres,                  only use thres number of top predicted sentences.
                                        None: not use top filtering.
        :return
        """
        ILP_compute_scores = 0.0

        # Get the s_logits for each sid
        assert len(cdd_sids) == len(cdd_sids2logits)
        cdd_slogits = [cdd_sids2logits[str(sid)] for sid in cdd_sids]
        cdd_slogits = torch.tensor(cdd_slogits)
        sorted_s_logits, sorted_idx = cdd_slogits.sort(descending=True)
        cdd_sent_sids = []
        cdd_sent_sentids = []
        cdd_sent_sentids_int = []
        cdd_sent_logits = []
        log_file = os.path.join(
            self.m_eval_output_path, 'log_{}.txt'.format(thread_idx)
        )

        if thres is None:
            # Select sentences from all cdd sentences
            for j in range(len(cdd_sids)):
                sid_j = cdd_sids[j]
                sentid_j = self.m_sid2sentid[sid_j]
                sentid_j_int = int(sentid_j)
                sent_pred_score_j = cdd_slogits[j].item()
                cdd_sent_sids.append(sid_j)
                cdd_sent_sentids.append(sentid_j)
                cdd_sent_sentids_int.append(sentid_j_int)
                cdd_sent_logits.append(sent_pred_score_j)
        else:
            for j in sorted_idx:
                sid_j = cdd_sids[j.item()]
                sentid_j = self.m_sid2sentid[sid_j]
                sentid_j_int = int(sentid_j)
                sent_pred_score_j = cdd_slogits[j.item()].item()
                if sent_pred_score_j <= 0.0:
                    break
                else:
                    if len(cdd_sent_logits) > 0:
                        try:
                            assert cdd_sent_logits[-1] >= sent_pred_score_j
                        except AssertionError:
                            exit()
                cdd_sent_sids.append(sid_j)
                cdd_sent_sentids.append(sentid_j)
                cdd_sent_sentids_int.append(sentid_j_int)
                cdd_sent_logits.append(sent_pred_score_j)
                if len(cdd_sent_sids) == thres:
                    break

        # Check how many selected cdd sentences after top-truncating
        try:
            assert len(cdd_sent_sids) >= topk
        except AssertionError:
            topk = len(cdd_sent_sids)
        # Get the cosine similarity matrix for these cdd sentences
        cdd_sent_pred_scores = np.array(cdd_sent_logits)
        pool_size = len(cdd_sent_sids)
        # Create a new model for ILP
        ILP_m = gp.Model("graph2x_ilp_greedy_{}".format(thread_idx))
        ILP_m.Params.LogToConsole = 0
        # ILP_m.setParam(GRB.Param.TimeLimit, 1000.0)
        # Create variables
        X = ILP_m.addMVar(shape=pool_size, vtype=GRB.BINARY, name="X")
        # Construct Objective
        ILP_m.setObjective(
            (cdd_sent_pred_scores @ X),
            GRB.MAXIMIZE
        )
        # Add the sum constrain of X
        ones_i = np.ones(len(cdd_sent_sids))
        ILP_m.addConstr(ones_i @ X == topk, name="c0")

        # Optimize model
        ILP_m.optimize()
        # Get the obj value
        ILP_compute_scores = ILP_m.objVal
        # Get the X variables' value
        select_sent_idx_i = np.where(X.X == 1.0)[0].tolist()

        # Get the select sids and sentids
        select_sids_i = [cdd_sent_sids[idx] for idx in select_sent_idx_i]
        select_sentids_i = [cdd_sent_sentids[idx] for idx in select_sent_idx_i]
        select_sent_logits_i = [cdd_sent_logits[idx] for idx in select_sent_idx_i]
        # Clean up the model
        ILP_m.dispose()
        # end_i = time.process_time()
        # print("{} for 1 user-item review ({} cdd sents)".format(end_i-start_i, num_sent_i))

        return select_sent_logits_i, select_sids_i, cdd_sent_sids, ILP_compute_scores

    def ILP_quad_sent_prediction(self, cdd_sids, cdd_sids2logits, topk=3, alpha=1.0, thres=None):
        """
        :param: s_logits,   sentence's predict scores.        shape: (batch_size, max_sent_num)
        :param: sids,       sentence's sid.                   shape: (batch_size, max_sent_num)
        :param: s_masks,    0 for masks. 1 for true sids.     shape: (batch_size, max_sent_num)
        :param: topk,       number of select sentences.
        :param: alpha,      trade-off parameter between 2 costs.
        :param: thres,      only use thres number of top predicted sentences.
                            None: not use top filtering.
        :return
        """
        ILP_compute_scores = 0.0

        # Get the s_logits for each sid
        assert len(cdd_sids) == len(cdd_sids2logits)
        cdd_slogits = [cdd_sids2logits[str(sid)] for sid in cdd_sids]
        cdd_slogits = torch.tensor(cdd_slogits)
        sorted_s_logits, sorted_idx = cdd_slogits.sort(descending=True)
        cdd_sent_sids = []
        cdd_sent_sentids = []
        cdd_sent_sentids_int = []
        cdd_sent_logits = []

        if thres is None:
            # Select sentences from all cdd sentences
            for j in range(len(cdd_sids)):
                sid_j = cdd_sids[j]
                sentid_j = self.m_sid2sentid[sid_j]
                sentid_j_int = int(sentid_j)
                sent_pred_score_j = cdd_slogits[j].item()
                cdd_sent_sids.append(sid_j)
                cdd_sent_sentids.append(sentid_j)
                cdd_sent_sentids_int.append(sentid_j_int)
                cdd_sent_logits.append(sent_pred_score_j)
        else:
            for j in sorted_idx:
                sid_j = cdd_sids[j.item()]
                sentid_j = self.m_sid2sentid[sid_j]
                sentid_j_int = int(sentid_j)
                sent_pred_score_j = cdd_slogits[j.item()].item()
                if sent_pred_score_j <= 0.0:
                    break
                else:
                    if len(cdd_sent_logits) > 0:
                        try:
                            assert cdd_sent_logits[-1] >= sent_pred_score_j
                        except AssertionError:
                            exit()
                cdd_sent_sids.append(sid_j)
                cdd_sent_sentids.append(sentid_j)
                cdd_sent_sentids_int.append(sentid_j_int)
                cdd_sent_logits.append(sent_pred_score_j)
                if len(cdd_sent_sids) == thres:
                    break

        # Check how many selected cdd sentences after top-truncating
        try:
            assert len(cdd_sent_sids) >= topk
        except AssertionError:
            topk = len(cdd_sent_sids)
        # Get the cosine similarity matrix for these cdd sentences
        cosine_sim_maxtrix = cosine_similarity(self.train_sent_tfidf_sparse[cdd_sent_sentids_int])
        cosine_sim_upper = np.triu(cosine_sim_maxtrix, 1)
        cdd_sent_pred_scores = np.array(cdd_sent_logits)
        pool_size = len(cdd_sent_sids)
        # Create a new model for ILP
        ILP_m = gp.Model("graph2x_ilp")
        ILP_m.Params.LogToConsole = 0
        # ILP_m.setParam(GRB.Param.TimeLimit, 1000.0)
        # Create variables
        X = ILP_m.addMVar(shape=pool_size, vtype=GRB.BINARY, name="X")
        # Construct Objective
        ILP_m.setObjective(
            (cdd_sent_pred_scores @ X) - alpha * (X @ cosine_sim_upper @ X),
            GRB.MAXIMIZE
        )
        # Add the sum constrain of X
        ones_i = np.ones(len(cdd_sent_sids))
        ILP_m.addConstr(ones_i @ X == topk, name="c0")
        # Optimize model
        ILP_m.optimize()
        # Get the obj value
        ILP_compute_scores = ILP_m.objVal
        # Get the X variables' value
        select_sent_idx_i = np.where(X.X == 1.0)[0].tolist()

        # Get the select sids and sentids
        select_sids_i = [cdd_sent_sids[idx] for idx in select_sent_idx_i]
        select_sentids_i = [cdd_sent_sentids[idx] for idx in select_sent_idx_i]
        select_sent_logits_i = [cdd_sent_logits[idx] for idx in select_sent_idx_i]
        # Clean up the model
        ILP_m.dispose()
        # end_i = time.process_time()
        # print("{} for 1 user-item review ({} cdd sents)".format(end_i-start_i, num_sent_i))

        return select_sent_logits_i, select_sids_i, ILP_compute_scores

    def compute_cosine_sim(self, cdd_sentids):
        """ Compute pairwise cosine similarity for sentences in cdd_sentids.
            The result should be a upper triangle matrix (the diagnol is all-zero).
        """
        cosine_sim_maxtrix = cosine_similarity(self.train_sent_tfidf_sparse[cdd_sentids])
        cosine_sim_upper = np.triu(cosine_sim_maxtrix, 1)
        return cosine_sim_upper

    def save_predict_sentences(self, true_userid, true_itemid, refs_sent, hyps_sent, topk_logits, pred_sids):
        # top-predicted/selected sentences
        predict_log_file = os.path.join(
            self.m_eval_output_path, 'eval_logging_{0}_{1}.txt'.format(self.m_dataset_name, label_format))
        with open(predict_log_file, 'a') as f:
            f.write("user id: {}\n".format(true_userid))
            f.write("item id: {}\n".format(true_itemid))
            f.write("hyps: {}\n".format(hyps_sent))
            f.write("refs: {}\n".format(refs_sent))
            f.write("probas: {}\n".format(topk_logits))
            # if use_trigram_blocking:
            #     f.write("rank: {}\n".format(ngram_block_pred_rank[j]))
            # elif use_bleu_filtering:
            #     f.write("rank: {}\n".format(bleu_filter_pred_rank[j]))
            f.write("========================================\n")

    def save_model_predict(self, graph_batch, batch_size, s_logits, sids, s_masks, target_sids):
        userid_batch = graph_batch.u_rawid
        itemid_batch = graph_batch.i_rawid
        for j in range(batch_size):
            userid_j = userid_batch[j].item()
            itemid_j = itemid_batch[j].item()
            # get the true user/item id
            true_userid_j = self.m_uid2user[userid_j]
            true_itemid_j = self.m_iid2item[itemid_j]
            assert s_logits[j].size(0) == sids[j].size(0)
            assert s_logits[j].size(0) == s_masks[j].size(0)
            num_sents_j = int(sum(s_masks[j]).item())
            # get predict sids and relevant logits
            cdd_sent_sids_j = []
            target_sent_sids_j = []
            cdd_sent_sids2logits_j = {}
            for ij in range(num_sents_j):
                sid_ij = sids[j][ij].item()
                assert sid_ij == int(sid_ij)
                sid_ij = int(sid_ij)
                cdd_sent_sids_j.append(sid_ij)
                assert sid_ij not in cdd_sent_sids2logits_j
                cdd_sent_sids2logits_j[sid_ij] = s_logits[j][ij].item()
            for sid_ij in target_sids[j]:
                target_sent_sids_j.append(sid_ij.item())
            # triple_data_list = []
            # for pos in range(len(s_logits[j])):
            #     triple_data_list.append(
            #         [s_logits[j][pos].item(), sids[j][pos].item(), s_masks[j][pos].item()])
            # predict_data_j['predict_data'] = triple_data_list
            # get this user-item's predict data
            predict_data_j = {
                'user': true_userid_j,
                'item': true_itemid_j,
                'cdd_sids': cdd_sent_sids_j,
                'target_sids': target_sent_sids_j,
                'cdd_sids2logits': cdd_sent_sids2logits_j
            }
            with open(self.model_pred_file, 'a') as f:
                json.dump(predict_data_j, f)
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### data
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='ratebeer')
    parser.add_argument('--data_file', type=str, default='data.pickle')
    parser.add_argument('--graph_dir', type=str, default='../output_graph/')
    parser.add_argument('--data_set', type=str, default='medium_500_pure')

    parser.add_argument('--vocab_file', type=str, default='vocab.json')
    parser.add_argument('--model_file', type=str, default="model_best.pt")
    parser.add_argument('--model_name', type=str, default="graph_sentence_extractor")
    parser.add_argument('--model_path', type=str, default="../checkpoint/")
    parser.add_argument('--eval_output_path', type=str, default="../result/")

    ### hyper-param
    parser.add_argument('--select_topk_s', type=int, default=5)
    parser.add_argument('--select_topk_f', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=1.0)

    ### others
    parser.add_argument('--parallel', action="store_true", default=False)
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--bias_lines', type=int, default=0)
    parser.add_argument('--select_top', action="store_true", default=False)
    parser.add_argument('--select_lines', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=100)

    args = parser.parse_args()

    print("Start ILP post-processing evaluation ...")
    eval_obj = EVAL_ILP(args)
    eval_obj.f_eval()
