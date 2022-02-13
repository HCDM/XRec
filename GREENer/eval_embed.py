import numpy as np
from numpy.core.numeric import indices
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
import json
from torch import nonzero
import torch.nn.functional as F
import torch.nn as nn
import datetime
import statistics
from metric import get_example_recall_precision, compute_bleu, get_bleu, get_feature_recall_precision, get_recall_precision_f1, get_sentence_bleu, get_recall_precision_f1_random
from rouge import Rouge
from nltk.translate import bleu_score
import pickle
import random

dataset_name = 'medium_500_pure'
label_format = 'soft_label'
# method to extract predicted sentences
use_blocking = False        # whether using 3-gram blocking or not
use_filtering = False       # whether using bleu score filtering or not
bleu_filter_value = 0.25

MAX_batch_output = 433


class EVAL_EMBED(object):
    def __init__(self, vocab_obj, args, device):
        super().__init__()
        self.m_batch_size = args.batch_size
        self.m_mean_loss = 0

        self.m_sid2swords = vocab_obj.m_sid2swords
        self.m_feature2fid = vocab_obj.m_feature2fid
        self.m_item2iid = vocab_obj.m_item2iid
        self.m_user2uid = vocab_obj.m_user2uid
        self.m_sent2sid = vocab_obj.m_sent2sid
        self.m_train_sent_num = vocab_obj.m_train_sent_num
        # feature / sentence init embeddings
        self.m_fid2fembed = vocab_obj.m_fid2fembed
        self.m_sid2sembed = vocab_obj.m_sid2sembed

        self.m_device = device
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_eval_output_path = args.eval_output_path
        self.select_s_topk = args.select_topk_s

        # get item id to item mapping
        self.m_iid2item = {self.m_item2iid[k]: k for k in self.m_item2iid}
        # get user id to user mapping
        self.m_uid2user = {self.m_user2uid[k]: k for k in self.m_user2uid}
        # get fid to feature(id) mapping
        self.m_fid2feature = {self.m_feature2fid[k]: k for k in self.m_feature2fid}
        # get sid to sent_id mapping
        self.m_sid2sentid = {self.m_sent2sid[k]: k for k in self.m_sent2sid}

        print("Evaluation results are saved under dir: {}".format(self.m_eval_output_path))
        print("Dataset: {0} \t Label: {1}".format(dataset_name, label_format))
        if use_blocking:
            print("Using tri-gram blocking.")
        elif use_filtering:
            print("Using bleu-based filtering.")
        else:
            print("Use the original scores.")

        # need to load some mappings
        id2feature_file = '../../Dataset/ratebeer/{}/train/feature/id2feature.json'.format(dataset_name)
        feature2id_file = '../../Dataset/ratebeer/{}/train/feature/feature2id.json'.format(dataset_name)
        trainset_id2sent_file = '../../Dataset/ratebeer/{}/train/sentence/id2sentence.json'.format(dataset_name)
        testset_id2sent_file = '../../Dataset/ratebeer/{}/test/sentence/id2sentence.json'.format(dataset_name)
        testset_useritem_cdd_withproxy_file = '../../Dataset/ratebeer/{}/test/useritem2sentids_withproxy.json'.format(dataset_name)
        trainset_user2sentid_file = '../../Dataset/ratebeer/{}/train/user/user2sentids.json'.format(dataset_name)
        trainset_item2sentid_file = '../../Dataset/ratebeer/{}/train/item/item2sentids.json'.format(dataset_name)
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

    def f_init_eval(self, network, model_file=None, reload_model=False):
        if reload_model:
            print("reload model")
            if not model_file:
                model_file = "model_best.pt"
            model_name = os.path.join(self.m_model_path, model_file)
            print("model name", model_name)
            check_point = torch.load(model_name)
            network.load_state_dict(check_point['model'])

        self.m_network = network

    def f_eval(self, train_data, eval_data):
        print('--'*10)
        num_nodes_per_graph = []
        num_edges_per_graph = []
        num_feature_nodes_per_graph = []
        num_sent_nodes_per_graph = []

        s_topk = self.select_s_topk
        s_topk_candidate = 20
        cnt_useritem_pair = 0
        cnt_useritem_batch = 0
        save_logging_cnt = 0

        # save files
        if use_blocking:
            cos_sim_results_file = os.path.join(
                self.m_eval_output_path, 'cos_sim_3gram_{}batch.json'.format(MAX_batch_output))
        elif use_filtering:
            cos_sim_results_file = os.path.join(
                self.m_eval_output_path, 'cos_sim_bleu_{}batch.json'.format(MAX_batch_output))
        else:
            cos_sim_results_file = os.path.join(
                self.m_eval_output_path, 'cos_sim_origin_{}batch.json'.format(MAX_batch_output))

        print("Saving cosine similarity results in: {}".format(cos_sim_results_file))

        self.m_network.eval()
        with torch.no_grad():
            print("Number of training data: {}".format(len(train_data)))
            print("Number of evaluation data: {}".format(len(eval_data)))
            print("Number of topk selected sentences: {}".format(s_topk))
            # Perform Evaluation on eval_data / train_data
            # for graph_batch in eval_data:
            for graph_batch in train_data:
                if cnt_useritem_batch % 100 == 0:
                    print("... eval ... ", cnt_useritem_batch)

                graph_batch = graph_batch.to(self.m_device)
                # logits: batch_size*max_sen_num
                (s_logits, sids, s_masks, target_sids,
                    f_logits, fids, f_masks, target_f_labels,
                    hidden_f_batch, graph_batch_x, mask_graph_batch_x) = self.m_network.eval_forward(graph_batch, get_embedding=True)

                graph_batch_x = graph_batch_x.cpu()
                mask_graph_batch_x = mask_graph_batch_x.cpu()
                # print(s_logits.shape)
                # print(sids.shape)
                # print(s_masks.shape)
                # print(sids)
                # print(s_masks)
                # mask_sids = sids*s_masks
                # print(mask_sids)
                # print(mask_sids.shape)
                # print(int(sum(s_masks[0].cpu()).item()))
                # print(sum(s_masks[1].cpu()).item())

                batch_size = s_logits.size(0)
                userid = graph_batch.u_rawid
                itemid = graph_batch.i_rawid

                # sentence prediction
                if use_blocking:
                    # 3-gram blocking
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.trigram_blocking_sent_prediction(
                        s_logits, sids, s_masks, batch_size, topk=s_topk, topk_cdd=s_topk_candidate
                    )
                elif use_filtering:
                    # bleu filtering
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.bleu_filtering_sent_prediction(
                        s_logits, sids, s_masks, batch_size, topk=s_topk, topk_cdd=s_topk_candidate, bleu_bound=bleu_filter_value
                    )
                else:
                    # original score
                    s_topk_logits, s_pred_sids, s_top_cdd_logits, s_top_cdd_pred_sids, s_bottom_cdd_logits, s_bottom_cdd_pred_sids = self.origin_blocking_sent_prediction(
                        s_logits, sids, s_masks, topk=s_topk, topk_cdd=s_topk_candidate
                    )

                # Decide the batch_save_flag. To get shorted results, we only print the first several batches' results
                cnt_useritem_batch += 1
                if cnt_useritem_batch <= MAX_batch_output:
                    batch_save_flag = True
                else:
                    batch_save_flag = False
                # Whether to break or continue(i.e. pass) when the batch_save_flag is false
                if batch_save_flag:
                    save_logging_cnt += 1
                else:
                    # pass or break. pass will continue evaluating full batch testing set, break will only
                    # evaluate the first several batches of the testing set.
                    # pass
                    break

                for j in range(batch_size):
                    userid_j = userid[j].item()
                    itemid_j = itemid[j].item()
                    # get number of nodes
                    num_nodes_per_graph.append(graph_batch[j].num_nodes)
                    # get number of edges
                    num_edges_per_graph.append(graph_batch[j].num_edges)
                    # get the true user/item id
                    true_userid_j = self.m_uid2user[userid_j]
                    true_itemid_j = self.m_iid2item[itemid_j]
                    # get the sids
                    sid_j = sids[j].cpu()
                    s_num_j = int(sum(s_masks[j].cpu()).item())
                    assert s_num_j == graph_batch[j]['s_num'].item()
                    mask_sid_j = sid_j[:s_num_j]
                    num_sent_nodes_per_graph.append(s_num_j)
                    # get the fids
                    fid_j = fids[j].cpu()
                    f_num_j = target_f_labels[j].cpu().size(0)
                    assert f_num_j == graph_batch[j]['f_num'].item()
                    mask_fid_j = fid_j[:f_num_j]
                    num_feature_nodes_per_graph.append(f_num_j)
                    # get the s_rawid and s_nid tensor
                    s_rawid_j = graph_batch[j].s_rawid.cpu()
                    s_nid_j = graph_batch[j].s_nid.cpu()

                    # # get a embedding
                    # f_emb_1 = self.m_fid2fembed[fid_j[0].item()]
                    # print(type(f_emb_1), len(f_emb_1))
                    # s_emb_1 = self.m_sid2sembed[sid_j[0].item()]
                    # print(type(s_emb_1), len(s_emb_1))
                    # exit()

                    # get the mapping of sid to nid
                    sid2nid = self.get_sid2nid(s_rawid_j, s_nid_j)
                    # get item-side sids from the union of user and item side cdd sids
                    item_side_sids = self.get_item_side_sids(s_rawid_j, true_userid_j, true_itemid_j)

                    # get the top-2 predicted sentences sid
                    pred_sid_pos_0 = s_pred_sids[j][0].item()
                    pred_sid_pos_1 = s_pred_sids[j][1].item()
                    # get the corresponding sentence content
                    pred_sent_pos_0 = self.m_sid2swords[pred_sid_pos_0]
                    pred_sent_pos_1 = self.m_sid2swords[pred_sid_pos_1]
                    # get the corresponding nid
                    pred_s_nid_pos_0 = sid2nid[pred_sid_pos_0]
                    pred_s_nid_pos_1 = sid2nid[pred_sid_pos_1]
                    # get the corresponding node embdding
                    pred_s_n_embed_pos_0 = graph_batch_x[j][pred_s_nid_pos_0]
                    pred_s_n_embed_pos_1 = graph_batch_x[j][pred_s_nid_pos_1]
                    # compute the similarity between this 2 node embeddings
                    cos_sim_pred_s_top2 = self.compute_emb_similarity(
                        pred_s_n_embed_pos_0, pred_s_n_embed_pos_1
                    )
                    # compute the weighted between
                    weighted_s_n_embed_pos_0 = self.compute_weighted_item_side_s_embed(
                        item_side_sids, pred_sid_pos_0, sid2nid, graph_batch_x[j]
                    )
                    weighted_s_n_embed_pos_1 = self.compute_weighted_item_side_s_embed(
                        item_side_sids, pred_sid_pos_1, sid2nid, graph_batch_x[j]
                    )
                    # compute the similarity between this 2 weighted node embeddings
                    cos_sim_item_s_weighted_s_top2 = self.compute_emb_similarity(
                        weighted_s_n_embed_pos_0, weighted_s_n_embed_pos_1
                    )
                    with open(cos_sim_results_file, 'a') as f_cos_sim:
                        cur_cos_sim_json = {
                            'user': true_userid_j,
                            'item': true_itemid_j,
                            'predict_top_1': pred_sent_pos_0,
                            'predict_top_2': pred_sent_pos_1,
                            'cosine_sim': cos_sim_pred_s_top2.item(),
                            'cosine_sim_after_weight': cos_sim_item_s_weighted_s_top2.item()
                        }
                        json.dump(cur_cos_sim_json, f_cos_sim)
                        f_cos_sim.write('\n')

        print("Totally {} graphs ...".format(len(num_nodes_per_graph)))
        print("Average number of nodes per graph: %.4f" % (np.mean(num_nodes_per_graph)))
        print("Average number of edges per graph: %.4f" % (np.mean(num_edges_per_graph)))
        print("Average number of feature nodes per graph: %.4f" % (np.mean(num_feature_nodes_per_graph)))
        print("Average number of sentence nodes per graph: %.4f" % (np.mean(num_sent_nodes_per_graph)))

    def compute_emb_similarity(self, embed_proj_0, embed_proj_1):
        """ Compute Cosine-similarity between feature embedding and sentence embedding
        :param: embed_proj_0: 256-dim vector. (e.g. feature embedding after feed through feature project layer)
        :param: embed_proj_1: 256-dim vector. (e.g. sent embedding after feed through sent project layer)
        return:
            cosine similarity score
        """
        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=0)
            cos_sim = cos(embed_proj_0, embed_proj_1)

        return cos_sim

    def compute_emb_similarity_batch(self, embed_proj_0, embed_proj_1):
        """ Compute Cosine-similarity between feature embedding and sentence embedding
        :param: embed_proj_0: (batch_size, 256) vector. (e.g. feature embedding after feed through feature project layer)
        :param: embed_proj_1: (batch_size, 256) vector. (e.g. sent embedding after feed through sent project layer)
        return:
            cosine similarity score
        """
        with torch.no_grad():
            batch_cos = nn.CosineSimilarity(dim=1)
            batch_cos_sim = batch_cos(embed_proj_0, embed_proj_1)

        return batch_cos_sim

    def compute_attn_weight(self, embed_proj_0, embed_proj_1):
        """
        :param: embed_proj_0,   (256,) 1-dim tensor
        :param: embed_proj_1,   (batch_size, 256) 2-dim tensor
        return:
            attention weight, 1-dim tensor
        """
        # TODO: Change torch.dot to torch.mm for better computing efficency
        # compute the weight. similar to compute attention weight
        # 1. compute inner product
        # 2. compute softmax
        embedding_dim = embed_proj_0.size(0)
        batch_size = embed_proj_1.size(0)
        assert embedding_dim == embed_proj_1.size(1)
        with torch.no_grad():
            # attn_weight = torch.zeros(batch_size)
            # # 1. compute inner product
            # for i in range(batch_size):
            #     attn_weight[i] = torch.dot(embed_proj_0, embed_proj_1[i])
            attn_weight = torch.mm(embed_proj_0.unsqueeze(0), embed_proj_1.T).squeeze()
            assert attn_weight.size(0) == batch_size
            # 2. compute softmax
            m = nn.Softmax(dim=0)
            attn_weight_soft = m(attn_weight)
        return attn_weight_soft

    def compute_weighted_item_side_s_embed(self, item_sids, query_sid, sid2nid, graph_embeds):
        """ Compute a weighted item-side s_node embeddings of current query s_node.
        :param: item_sids:      item-side sids
        :param: query_sid:      a query sid which is used to compute a weighted embedding on it
        :param: sid2nid:        sid to nid mapping
        :param: graph_embeds:   graph_batch.x
        :return:
            a weighted sum of item-side node embeddings
        """
        # get the query sid's nid
        query_s_nid = sid2nid[query_sid]
        # get the item side sids' nids
        item_s_nids = [sid2nid[sid] for sid in item_sids]
        batch_size = len(item_s_nids)
        # get the query sid's corresponding node embedding, and form batch
        query_s_n_embed = graph_embeds[query_s_nid]
        # get the item side sids' corresponding node embedding, and form batch
        item_s_n_embed_batch = [graph_embeds[nid].unsqueeze(0) for nid in item_s_nids]
        item_s_n_embed_batch = torch.cat(item_s_n_embed_batch, dim=0)

        # # compute the batch cosine similarity, tensor, shape: [len(item_s_nids)]
        # batch_cos_sim = self.compute_emb_similarity_batch(query_s_n_embed_batch, item_s_n_embed_batch)
        # compute the attention weight (inner product + softmax)
        attn_weight = self.compute_attn_weight(query_s_n_embed, item_s_n_embed_batch)

        # compute a weighted s_node embedding
        embed_shape = query_s_n_embed.size(0)
        weight_query_s_embed = torch.zeros(embed_shape)
        for i in range(batch_size):
            weight_query_s_embed = weight_query_s_embed.add(item_s_n_embed_batch[i], alpha=attn_weight[i])

        return weight_query_s_embed

    def get_fid2nid(self, f_rawid_tensor, f_nid_tensor):
        """ Mapping the f_rawid with f_nid
        """
        assert f_rawid_tensor.size(0) == f_nid_tensor.size(0)
        fid2nid = dict()
        for idx in range(f_rawid_tensor.size(0)):
            assert f_rawid_tensor[idx].item() not in fid2nid
            fid2nid[f_rawid_tensor[idx].item()] = f_nid_tensor[idx].item()
        return fid2nid

    def get_sid2nid(self, s_rawid_tensor, s_nid_tensor):
        """ Mapping the s_rawid with s_nid
        """
        assert s_rawid_tensor.size(0) == s_nid_tensor.size(0)
        sid2nid = dict()
        for idx in range(s_rawid_tensor.size(0)):
            assert s_rawid_tensor[idx].item() not in sid2nid
            sid2nid[s_rawid_tensor[idx].item()] = s_nid_tensor[idx].item()
        return sid2nid

    def get_sid_user_item_source(self, pred_sids, user_id, item_id):
        """ Given the predicted/selected sids, find each sid's source, i.e. user-side or item-side sentence.
        :param: pred_sids:  predicted sids, tensor
        :param: user_id:    userid on the dataset, str
        :param: item_id:    itemid on the dataset, str
        return: user_item_source: the user/item side of the sids, a list
        """
        user_item_source = []
        for sid in pred_sids:
            sid_i = sid.item()
            ''' mapping back this sid to sentid (used in the dataset)
            since this is on trainset, we don't need minus the number of sentences
            in the trianset to get the true sentid
            '''
            sentid_i = self.m_sid2sentid[sid_i]
            # check whether this sentid occurs in the user-side or item-side
            if sentid_i in self.d_trainset_user2sentid[user_id]:
                if sentid_i in self.d_trainset_item2sentid[item_id]:
                    user_item_source.append("both user and item side")
                else:
                    user_item_source.append("user side")
            else:
                if sentid_i in self.d_trainset_item2sentid[item_id]:
                    user_item_source.append("item side")
                else:
                    raise Exception("Error: User:{0}\tItem:{1}\tSentid:{2} NOT ON USER AND ITEM SIDE!".format(
                        user_id, item_id, sentid_i
                    ))
        return user_item_source

    def get_item_side_sids(self, cdd_sids, user_id, item_id):
        """ Given the cdd sids (on trainset), find item-side sids
        :param: cdd_sids:   candidate sids
        :param: user_id:    userid in the dataset, str
        :param: item_id:    itemid in the dataset, str
        return:
            item_side_sids: item-side sids, list
        """
        item_side_sids = []
        for sid in cdd_sids:
            sid_i = sid.item()
            sentid_i = self.m_sid2sentid[sid_i]
            # check whether this sentid occurs in the user-side or item-side
            if sentid_i in self.d_trainset_item2sentid[item_id]:
                item_side_sids.append(sid_i)
            else:
                if sentid_i in self.d_trainset_user2sentid[user_id]:
                    pass
                else:
                    raise Exception("Error: User:{0}\tItem:{1}\tSentid:{2} NOT ON USER AND ITEM SIDE!".format(
                        user_id, item_id, sentid_i
                    ))
        return item_side_sids

    def ngram_blocking(self, sents, p_sent, n_win, k):
        """ ngram blocking
        :param sents:   batch of lists of candidate sentence, each candidate sentence is a string. shape: [batch_size, sent_num]
        :param p_sent:  torch tensor. batch of predicted/relevance scores of each candidate sentence. shape: (batch_sizem, sent_num)
        :param n_win:   ngram window size, i.e. which n-gram we are using. n_win can be 2,3,4,...
        :param k:       we are selecting the top-k sentences

        :return:        selected index of sids
        """
        batch_size = p_sent.size(0)
        batch_select_idx = []
        batch_select_proba = []
        batch_select_rank = []
        assert len(sents) == len(p_sent)
        assert len(sents) == batch_size
        assert len(sents[0]) == len(p_sent[0])
        # print(sents)
        # print("batch size (sents): {}".format(len(sents)))
        for i in range(len(sents)):
            # print(len(sents[i]))
            assert len(sents[i]) == len(sents[0])
            assert len(sents[i]) == len(p_sent[i])
        # print(p_sent)
        # print(p_sent.shape)
        for batch_idx in range(batch_size):
            ngram_list = []
            _, sorted_idx = p_sent[batch_idx].sort(descending=True)
            select_idx = []
            select_proba = []
            select_rank = []
            idx_rank = 0
            for idx in sorted_idx:
                idx_rank += 1
                try:
                    cur_sent = sents[batch_idx][idx]
                except KeyError:
                    print("Error! i: {0} \t idx: {1}".format(batch_idx, idx))
                cur_tokens = cur_sent.split()
                overlap_flag = False
                cur_sent_ngrams = []
                for i in range(len(cur_tokens)-n_win+1):
                    this_ngram = " ".join(cur_tokens[i:(i+n_win)])
                    if this_ngram in ngram_list:
                        overlap_flag = True
                        break
                    else:
                        cur_sent_ngrams.append(this_ngram)
                if not overlap_flag:
                    select_idx.append(idx)
                    select_proba.append(p_sent[batch_idx][idx])
                    select_rank.append(idx_rank)
                    ngram_list.extend(cur_sent_ngrams)
                    if len(select_idx) >= k:
                        break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # convert list to torch tensor
        batch_select_idx = torch.LongTensor(batch_select_idx)
        return batch_select_idx, batch_select_proba, batch_select_rank

    def bleu_filtering(self, sents, p_sent, k, filter_value=0.25):
        """ bleu filtering
        :param sents:   batch of lists of candidate sentence, each candidate sentence is a string. shape: [batch_size, sent_num]
        :param p_sent:  torch tensor. batch of predicted/relevance scores of each candidate sentence. shape: (batch_sizem, sent_num)
        :param k:       we are selecting the top-k sentences
        :param filter_value: the boundary value of bleu-2 + bleu-3 that defines whether we should filter a sentence

        :return:        selected index of sids
        """
        batch_size = p_sent.size(0)
        batch_select_idx = []
        batch_select_proba = []
        batch_select_rank = []
        assert len(sents) == len(p_sent)
        assert len(sents[0]) == len(p_sent[0])
        for i in range(len(sents)):
            assert len(sents[i]) == len(sents[0])
            assert len(sents[i]) == len(p_sent[i])
        for batch_idx in range(batch_size):
            _, sorted_idx = p_sent[batch_idx].sort(descending=True)
            select_idx = []
            select_proba = []
            select_rank = []
            select_sents = []
            idx_rank = 0
            for idx in sorted_idx:
                idx_rank += 1
                try:
                    cur_sent = sents[batch_idx][idx]
                except KeyError:
                    print("Error! batch: {0} \t idx: {1}".format(batch_idx, idx))
                if len(select_sents) == 0:
                    # add current sentence into the selected sentences
                    select_sents.append(cur_sent)
                    select_idx.append(idx)
                    select_proba.append(p_sent[batch_idx][idx])
                    select_rank.append(idx_rank)
                    if len(select_idx) >= k:
                        break
                else:
                    # compute bleu score
                    this_ref_sents = []
                    for this_sent in select_sents:
                        this_ref_sents.append(this_sent.split())
                    this_hypo_sent = cur_sent.split()
                    sf = bleu_score.SmoothingFunction()
                    bleu_1 = bleu_score.sentence_bleu(
                        this_ref_sents, this_hypo_sent, smoothing_function=sf.method1, weights=[1.0, 0.0, 0.0, 0.0])
                    bleu_2 = bleu_score.sentence_bleu(
                        this_ref_sents, this_hypo_sent, smoothing_function=sf.method1, weights=[0.5, 0.5, 0.0, 0.0])
                    bleu_3 = bleu_score.sentence_bleu(
                        this_ref_sents, this_hypo_sent, smoothing_function=sf.method1, weights=[1.0/3, 1.0/3, 1.0/3, 0.0])
                    if (bleu_2 + bleu_3) < filter_value:
                        # add current sentence into the selected sentences
                        select_sents.append(cur_sent)
                        select_idx.append(idx)
                        select_proba.append(p_sent[batch_idx][idx])
                        select_rank.append(idx_rank)
                        if len(select_idx) >= k:
                            break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # convert list to torch tensor
        batch_select_idx = torch.LongTensor(batch_select_idx)
        return batch_select_idx, batch_select_proba, batch_select_rank

    def origin_blocking_sent_prediction(self, s_logits, sids, s_masks, topk=3, topk_cdd=20):
        # incase some not well-trained model will predict the logits for all sentences as 0.0, we apply masks on it
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        # 1. get the top-k predicted sentences which form the hypothesis
        topk_logits, topk_pred_snids = torch.topk(masked_s_logits, topk, dim=1)
        # topk sentence index
        # pred_sids: shape: (batch_size, topk_sent)
        sids = sids.cpu()
        pred_sids = sids.gather(dim=1, index=topk_pred_snids)
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_logits, top_cdd_pred_snids = torch.topk(masked_s_logits, topk_cdd, dim=1)
        top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def trigram_blocking_sent_prediction(self, s_logits, sids, s_masks, batch_size, topk=3, topk_cdd=20):
        # use n-gram blocking
        # get all the sentence content
        batch_sents_content = []
        assert len(sids) == s_logits.size(0)      # this is the batch size
        for i in range(batch_size):
            cur_sents_content = []
            assert len(sids[i]) == len(sids[0])
            for cur_sid in sids[i]:
                cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
            batch_sents_content.append(cur_sents_content)
        assert len(batch_sents_content[0]) == len(batch_sents_content[-1])      # this is the max_sent_len (remember we are using zero-padding for batch data)
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        sids = sids.cpu()
        # 1. get the top-k predicted sentences which form the hypothesis
        ngram_block_pred_snids, ngram_block_pred_proba, ngram_block_pred_rank = self.ngram_blocking(
            batch_sents_content, masked_s_logits, n_win=3, k=topk
        )
        pred_sids = sids.gather(dim=1, index=ngram_block_pred_snids)
        topk_logits = ngram_block_pred_proba
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_pred_snids, top_cdd_logits, _ = self.ngram_blocking(
            batch_sents_content, masked_s_logits, n_win=3, k=topk_cdd
        )
        top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids

    def bleu_filtering_sent_prediction(self, s_logits, sids, s_masks, batch_size, topk=3, topk_cdd=20, bleu_bound=0.25):
        # use bleu-based filtering
        # get all the sentence content
        batch_sents_content = []
        assert len(sids) == s_logits.size(0)      # this is the batch size
        for i in range(batch_size):
            cur_sents_content = []
            assert len(sids[i]) == len(sids[0])
            for cur_sid in sids[i]:
                cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
            batch_sents_content.append(cur_sents_content)
        assert len(batch_sents_content[0]) == len(batch_sents_content[-1])      # this is the max_sent_len (remember we are using zero-padding for batch data)
        sids = sids.cpu()
        masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        # 1. get the top-k predicted sentences which form the hypothesis
        bleu_filter_pred_snids, bleu_filter_pred_proba, bleu_filter_pred_rank = self.bleu_filtering(
            batch_sents_content, masked_s_logits, k=topk, filter_value=bleu_bound)
        pred_sids = sids.gather(dim=1, index=bleu_filter_pred_snids)
        topk_logits = bleu_filter_pred_proba
        # 2. get the top-20 predicted sentences' content and proba
        top_cdd_pred_snids, top_cdd_logits, _ = self.bleu_filtering(
            batch_sents_content, masked_s_logits, k=topk_cdd, filter_value=bleu_bound)
        top_cdd_pred_sids = sids.gather(dim=1, index=top_cdd_pred_snids)
        # 3. get the bottom-20 predicted sentences' content and proba
        reverse_s_logits = (1-masked_s_logits)*s_masks.cpu()
        bottom_cdd_logits, bottom_cdd_pred_snids = torch.topk(reverse_s_logits, topk_cdd, dim=1)
        bottom_cdd_pred_sids = sids.gather(dim=1, index=bottom_cdd_pred_snids)

        return topk_logits, pred_sids, top_cdd_logits, top_cdd_pred_sids, bottom_cdd_logits, bottom_cdd_pred_sids
