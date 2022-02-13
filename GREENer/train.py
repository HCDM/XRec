import os
import json
import time
import torch
import argparse
import numpy as np
import datetime
import torch.nn as nn
from tensorboardX import SummaryWriter
from loss import XE_LOSS, BPR_LOSS, SIG_LOSS
from metric import get_example_recall_precision, compute_bleu, get_bleu, get_sentence_bleu
from model import GraphX
import random
import torch.nn.functional as F
from rouge import Rouge
from torch_geometric.utils import to_dense_batch


class TRAINER(object):

    def __init__(self, vocab_obj, args, device):
        super().__init__()

        self.m_device = device

        self.m_save_mode = True

        self.m_mean_train_loss = 0
        self.m_mean_train_precision = 0
        self.m_mean_train_recall = 0

        self.m_mean_val_loss = 0
        self.m_mean_eval_precision = 0
        self.m_mean_eval_recall = 0

        self.m_mean_eval_bleu = 0

        self.m_epochs = args.epoch_num
        self.m_batch_size = args.batch_size

        # self.m_rec_loss = XE_LOSS(vocab_obj.item_num, self.m_device)
        # self.m_rec_loss = BPR_LOSS(self.m_device)
        self.m_rec_loss = SIG_LOSS(self.m_device)
        self.m_rec_soft_loss = BPR_LOSS(self.m_device)
        # self.m_criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.m_train_step = 0
        self.m_valid_step = 0
        self.m_model_path = args.model_path
        self.m_model_file = args.model_file
        self.m_data_dir = args.data_dir
        self.m_dataset = args.data_set
        self.m_dataset_name = args.data_name

        self.m_grad_clip = args.grad_clip
        self.m_weight_decay = args.weight_decay
        # self.m_l2_reg = args.l2_reg
        self.m_feature_loss_lambda = args.feature_lambda        # the weight for the feature loss
        self.m_soft_train = args.soft_label                     # use soft label for sentence prediction
        self.m_multi_task = args.multi_task                     # use multi-task loss (sent + feat)
        self.m_valid_trigram = args.valid_trigram               # use trigram blocking for valid
        self.m_valid_trigram_feat = args.valid_trigram_feat     # use trigram + feature unigram for valid
        self.m_select_topk_s = args.select_topk_s               # select topk sentence for valid
        self.m_select_topk_f = args.select_topk_f               # select topk feature for valid

        self.m_train_iteration = 0
        self.m_valid_iteration = 0
        self.m_eval_iteration = 0
        self.m_print_interval = args.print_interval

        self.m_sid2swords = vocab_obj.m_sid2swords
        self.m_item2iid = vocab_obj.m_item2iid
        self.m_user2uid = vocab_obj.m_user2uid
        self.m_iid2item = {self.m_item2iid[k]: k for k in self.m_item2iid}
        self.m_uid2user = {self.m_user2uid[k]: k for k in self.m_user2uid}
        feature2id_file = os.path.join(self.m_data_dir, 'train/feature/feature2id.json')
        testset_combined_file = os.path.join(self.m_data_dir, 'test_combined.json')
        with open(feature2id_file, 'r') as f:
            self.d_feature2id = json.load(f)
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

        print("--"*10+"train params"+"--"*10)
        print("print_interval", self.m_print_interval)
        print("number of topk selected sentences: {}".format(self.m_select_topk_s))
        if self.m_valid_trigram:
            print("use trigram blocking for validation")
        elif self.m_valid_trigram_feat:
            print("use trigram + feature unigram for validation")
        else:
            print("use the original topk scores for validation")
        self.m_overfit_epoch_threshold = 3

    def f_save_model(self, checkpoint):
        # checkpoint = {'model':network.state_dict(),
        #     'epoch': epoch,
        #     'en_optimizer': en_optimizer,
        #     'de_optimizer': de_optimizer
        # }
        torch.save(checkpoint, self.m_model_file)

    def f_train(self, train_data, valid_data, network, optimizer, logger_obj):
        last_train_loss = 0
        last_eval_loss = 0
        self.m_mean_eval_loss = 0

        overfit_indicator = 0

        # best_eval_precision = 0
        best_eval_recall = 0
        best_eval_bleu = 0
        # self.f_init_word_embed(pretrain_word_embed, network)
        try:
            for epoch in range(self.m_epochs):
                print("++"*10, epoch, "++"*10)

                s_time = datetime.datetime.now()
                self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()
                print("validation epoch duration", e_time-s_time)

                if last_eval_loss == 0:
                    last_eval_loss = self.m_mean_eval_loss

                elif last_eval_loss < self.m_mean_eval_loss:
                    print(
                        "!"*10, "error val loss increase", "!"*10,
                        "last val loss %.4f" % last_eval_loss,
                        "cur val loss %.4f" % self.m_mean_eval_loss
                    )
                    overfit_indicator += 1

                    # if overfit_indicator > self.m_overfit_epoch_threshold:
                    # 	break
                else:
                    print(
                        "last val loss %.4f" % last_eval_loss,
                        "cur val loss %.4f" % self.m_mean_eval_loss
                    )
                    last_eval_loss = self.m_mean_eval_loss

                if best_eval_bleu < self.m_mean_eval_bleu:
                    print("... saving model ...")
                    checkpoint = {'model': network.state_dict()}
                    self.f_save_model(checkpoint)
                    best_eval_bleu = self.m_mean_eval_bleu

                print("--"*10, epoch, "--"*10)

                s_time = datetime.datetime.now()
                # train_data.sampler.set_epoch(epoch)
                self.f_train_epoch(train_data, network, optimizer, logger_obj)
                # self.f_eval_train_epoch(train_data, network, optimizer, logger_obj)
                e_time = datetime.datetime.now()

                print("epoch duration", e_time-s_time)

                if last_train_loss == 0:
                    last_train_loss = self.m_mean_train_loss

                elif last_train_loss < self.m_mean_train_loss:
                    print(
                        "!"*10, "error training loss increase", "!"*10,
                        "last train loss %.4f" % last_train_loss,
                        "cur train loss %.4f" % self.m_mean_train_loss
                    )
                    # break
                else:
                    print(
                        "last train loss %.4f" % last_train_loss,
                        "cur train loss %.4f" % self.m_mean_train_loss
                    )
                    last_train_loss = self.m_mean_train_loss

                # if best_eval_bleu < self.m_mean_eval_bleu:
                #     print("... saving model ...")
                #     checkpoint = {'model': network.state_dict()}
                #     self.f_save_model(checkpoint)
                #     best_eval_bleu = self.m_mean_eval_bleu

            s_time = datetime.datetime.now()
            self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)
            if best_eval_bleu < self.m_mean_eval_bleu:
                print("... saving model ...")
                checkpoint = {'model': network.state_dict()}
                self.f_save_model(checkpoint)
                best_eval_bleu = self.m_mean_eval_bleu

        except KeyboardInterrupt:
            print("--"*20)
            print("... exiting from training early")
            if best_eval_bleu < self.m_mean_eval_bleu:
                print("... final save ...")
                checkpoint = {'model': network.state_dict()}
                self.f_save_model(checkpoint)
                best_eval_bleu = self.m_mean_eval_bleu

            s_time = datetime.datetime.now()
            self.f_eval_epoch(valid_data, network, optimizer, logger_obj)
            e_time = datetime.datetime.now()
            print("test epoch duration", e_time-s_time)

            print(" done !!!")

    def f_train_epoch(self, train_data, network, optimizer, logger_obj):
        loss_s_list, loss_f_list, loss_list = [], [], []
        tmp_loss_s_list, tmp_loss_f_list, tmp_loss_list = [], [], []

        iteration = 0
        logger_obj.f_add_output2IO(" "*10+"training the user and item encoder"+" "*10)
        start_time = time.time()
        # Start one epoch train of the network
        network.train()
        feat_loss_weight = self.m_feature_loss_lambda

        for g_batch in train_data:
            # print("graph_batch", g_batch)
            # if i % self.m_print_interval == 0:
            #     print("... eval ... ", i)
            graph_batch = g_batch.to(self.m_device)
            logits_s, logits_f = network(graph_batch)

            labels_s = graph_batch.s_label
            loss = None
            loss_s = None
            if not self.m_soft_train:
                # If not using soft label, only the gt sentences are labeled as 1
                labels_s = (labels_s == 3)
                loss_s = self.m_rec_loss(logits_s, labels_s.float())
            else:
                loss_s = self.m_rec_soft_loss(graph_batch, logits_s, labels_s)

            # 1. Loss from feature prediction
            labels_f = graph_batch.f_label
            loss_f = self.m_rec_loss(logits_f, labels_f.float())
            # 2. multi-task loss, sum of sentence loss and feature loss
            loss = loss_s + feat_loss_weight*loss_f

            # add current sentence prediction loss
            loss_s_list.append(loss_s.item())
            tmp_loss_s_list.append(loss_s.item())
            # add current feature prediction loss
            loss_f_list.append(loss_f.item())
            tmp_loss_f_list.append(loss_f.item())
            # add current loss
            loss_list.append(loss.item())
            tmp_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            # perform gradient clip
            # if self.m_grad_clip:
            #     max_norm = 5.0
            #     torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm)

            optimizer.step()

            self.m_train_iteration += 1
            iteration += 1
            if iteration % self.m_print_interval == 0:
                logger_obj.f_add_output2IO(
                    "%d, loss:%.4f, sent loss:%.4f, weighted feat loss:%.4f, feat loss:%.4f" % (
                        iteration, np.mean(tmp_loss_list), np.mean(tmp_loss_s_list),
                        feat_loss_weight*np.mean(tmp_loss_f_list), np.mean(tmp_loss_f_list)
                    )
                )

                tmp_loss_s_list, tmp_loss_f_list, tmp_loss_list = [], [], []

        logger_obj.f_add_output2IO(
            "%d, loss:%.4f, sent loss:%.4f, weighted feat loss:%.4f, feat loss:%.4f" % (
                self.m_train_iteration, np.mean(loss_list), np.mean(loss_s_list),
                feat_loss_weight*np.mean(loss_f_list), np.mean(loss_f_list)
            )
        )
        logger_obj.f_add_scalar2tensorboard("train/loss", np.mean(loss_list), self.m_train_iteration)
        logger_obj.f_add_scalar2tensorboard("train/sent_loss", np.mean(loss_s_list), self.m_train_iteration)
        logger_obj.f_add_scalar2tensorboard("train/feat_loss", np.mean(loss_f_list), self.m_train_iteration)

        end_time = time.time()
        print("+++ duration +++", end_time-start_time)
        self.m_mean_train_loss = np.mean(loss_list)

    def f_eval_train_epoch(self, eval_data, network, optimizer, logger_obj):
        loss_list = []
        recall_list, precision_list, F1_list = [], [], []
        rouge_1_f_list, rouge_1_p_list, rouge_1_r_list = [], [], []
        rouge_2_f_list, rouge_2_p_list, rouge_2_r_list = [], [], []
        rouge_l_f_list, rouge_l_p_list, rouge_l_r_list = [], [], []
        bleu_list, bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], [], []

        self.m_eval_iteration = self.m_train_iteration

        logger_obj.f_add_output2IO(" "*10+" eval for train data"+" "*10)

        rouge = Rouge()

        network.eval()
        topk = 3

        start_time = time.time()

        with torch.no_grad():
            for i, (G, index) in enumerate(eval_data):
                eval_flag = random.randint(1, 100)
                if eval_flag != 2:
                    continue

                G = G.to(self.m_device)

                logits = network(G)
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)

                G.nodes[snode_id].data["p"] = logits
                glist = dgl.unbatch(G)

                loss = self.m_rec_loss(glist)

                for j in range(len(glist)):
                    hyps_j = []
                    refs_j = []

                    idx = index[j]
                    example_j = eval_data.dataset.get_example(idx)

                    label_sid_list_j = example_j["label_sid"]
                    gt_sent_num = len(label_sid_list_j)
                    # print("gt_sent_num", gt_sent_num)

                    g_j = glist[j]
                    snode_id_j = g_j.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
                    N = len(snode_id_j)
                    p_sent_j = g_j.ndata["p"][snode_id_j]
                    p_sent_j = p_sent_j.view(-1)
                    # p_sent_j = p_sent_j.view(-1, 2)

                    # topk_j, pred_idx_j = torch.topk(p_sent_j[:, 1], min(topk, N))
                    # topk_j, topk_pred_idx_j = torch.topk(p_sent_j, min(topk, N))
                    topk_j, topk_pred_idx_j = torch.topk(p_sent_j, gt_sent_num)
                    topk_pred_snode_id_j = snode_id_j[topk_pred_idx_j]

                    topk_pred_sid_list_j = g_j.nodes[topk_pred_snode_id_j].data["raw_id"]
                    topk_pred_logits_list_j = g_j.nodes[topk_pred_snode_id_j].data["p"]

                    # recall_j, precision_j = get_example_recall_precision(pred_sid_list_j.cpu(), label_sid_list_j, min(topk, N))

                    print("topk_j", topk_j)
                    print("label_sid_list_j", label_sid_list_j)
                    print("topk_pred_idx_j", topk_pred_sid_list_j)

                    recall_j, precision_j = get_example_recall_precision(topk_pred_sid_list_j.cpu(), label_sid_list_j, gt_sent_num)

                    recall_list.append(recall_j)
                    precision_list.append(precision_j)

                    for sid_k in label_sid_list_j:
                        refs_j.append(self.m_sid2swords[sid_k])

                    for sid_k in topk_pred_sid_list_j:
                        hyps_j.append(self.m_sid2swords[sid_k.item()])

                    hyps_j = " ".join(hyps_j)
                    refs_j = " ".join(refs_j)

                    scores_j = rouge.get_scores(hyps_j, refs_j, avg=True)

                    rouge_1_f_list.append(scores_j["rouge-1"]["f"])
                    rouge_1_r_list.append(scores_j["rouge-1"]["r"])
                    rouge_1_p_list.append(scores_j["rouge-1"]["p"])

                    rouge_2_f_list.append(scores_j["rouge-2"]["f"])
                    rouge_2_r_list.append(scores_j["rouge-2"]["r"])
                    rouge_2_p_list.append(scores_j["rouge-2"]["p"])

                    rouge_l_f_list.append(scores_j["rouge-l"]["f"])
                    rouge_l_r_list.append(scores_j["rouge-l"]["r"])
                    rouge_l_p_list.append(scores_j["rouge-l"]["p"])

                    # bleu_scores_j = compute_bleu([hyps_j], [refs_j])
                    bleu_scores_j = compute_bleu([[refs_j.split()]], [hyps_j.split()])
                    bleu_list.append(bleu_scores_j)
                    # bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_bleu([refs_j], [hyps_j])
                    bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_sentence_bleu([refs_j.split()], hyps_j.split())
                    bleu_1_list.append(bleu_1_scores_j)
                    bleu_2_list.append(bleu_2_scores_j)
                    bleu_3_list.append(bleu_3_scores_j)
                    bleu_4_list.append(bleu_4_scores_j)

                loss_list.append(loss.item())

            end_time = time.time()
            duration = end_time - start_time
            print("... one epoch", duration)

            logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)
            # logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)

        self.m_mean_eval_loss = np.mean(loss_list)
        self.m_mean_eval_recall = np.mean(recall_list)
        self.m_mean_eval_precision = np.mean(precision_list)

        self.m_mean_eval_rouge_1_f = np.mean(rouge_1_f_list)
        self.m_mean_eval_rouge_1_r = np.mean(rouge_1_r_list)
        self.m_mean_eval_rouge_1_p = np.mean(rouge_1_p_list)

        self.m_mean_eval_rouge_2_f = np.mean(rouge_2_f_list)
        self.m_mean_eval_rouge_2_r = np.mean(rouge_2_r_list)
        self.m_mean_eval_rouge_2_p = np.mean(rouge_2_p_list)

        self.m_mean_eval_rouge_l_f = np.mean(rouge_l_f_list)
        self.m_mean_eval_rouge_l_r = np.mean(rouge_l_r_list)
        self.m_mean_eval_rouge_l_p = np.mean(rouge_l_p_list)

        self.m_mean_eval_bleu = np.mean(bleu_list)
        self.m_mean_eval_bleu_1 = np.mean(bleu_1_list)
        self.m_mean_eval_bleu_2 = np.mean(bleu_2_list)
        self.m_mean_eval_bleu_3 = np.mean(bleu_3_list)
        self.m_mean_eval_bleu_4 = np.mean(bleu_4_list)

        logger_obj.f_add_output2IO("%d, NLL_loss:%.4f" % (self.m_eval_iteration, self.m_mean_eval_loss))
        logger_obj.f_add_output2IO("recall@%d:%.4f" % (topk, self.m_mean_eval_recall))
        logger_obj.f_add_output2IO("precision@%d:%.4f" % (topk, self.m_mean_eval_precision))

        logger_obj.f_add_output2IO(
            "rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f" % (
                self.m_mean_eval_rouge_1_f, self.m_mean_eval_rouge_1_p, self.m_mean_eval_rouge_1_r,
                self.m_mean_eval_rouge_2_f, self.m_mean_eval_rouge_2_p, self.m_mean_eval_rouge_2_r,
                self.m_mean_eval_rouge_l_f, self.m_mean_eval_rouge_l_p, self.m_mean_eval_rouge_l_r))
        logger_obj.f_add_output2IO("bleu:%.4f" % (self.m_mean_eval_bleu))
        logger_obj.f_add_output2IO("bleu-1:%.4f" % (self.m_mean_eval_bleu_1))
        logger_obj.f_add_output2IO("bleu-2:%.4f" % (self.m_mean_eval_bleu_2))
        logger_obj.f_add_output2IO("bleu-3:%.4f" % (self.m_mean_eval_bleu_3))
        logger_obj.f_add_output2IO("bleu-4:%.4f" % (self.m_mean_eval_bleu_4))

        network.train()

    def f_eval_epoch(self, eval_data, network, optimizer, logger_obj):
        # loss_list = []
        # recall_list, precision_list, F1_list = [], [], []
        rouge_1_f_list, rouge_1_p_list, rouge_1_r_list = [], [], []
        rouge_2_f_list, rouge_2_p_list, rouge_2_r_list = [], [], []
        rouge_l_f_list, rouge_l_p_list, rouge_l_r_list = [], [], []
        bleu_list, bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], [], []

        self.m_eval_iteration = self.m_train_iteration

        logger_obj.f_add_output2IO(" "*10+" eval the user and item encoder"+" "*10)

        rouge = Rouge()
        # topk = 3

        # start one epoch validation
        network.eval()
        start_time = time.time()
        i = 0   # count batch

        with torch.no_grad():
            for graph_batch in eval_data:
                # eval_flag = random.randint(1,5)
                # if eval_flag != 2:
                # 	continue
                # start_time = time.time()
                # print("... eval ", i)

                if i % 100 == 0:
                    print("... eval ... ", i)
                i += 1

                graph_batch = graph_batch.to(self.m_device)

                # #### logits: batch_size*max_sen_num ####
                s_logits, sids, s_masks, target_sids, _, _, _, _, _ = network.eval_forward(graph_batch)
                batch_size = s_logits.size(0)
                # get batch userid and itemid
                uid_batch = graph_batch.u_rawid
                iid_batch = graph_batch.i_rawid
                # map uid to userid and iid to itemid
                userid_batch = [self.m_uid2user[uid_batch[j].item()] for j in range(batch_size)]
                itemid_batch = [self.m_iid2item[iid_batch[j].item()] for j in range(batch_size)]

                # loss = self.m_rec_loss(glist)
                # loss_list.append(loss.item())

                # #### topk sentence ####
                # logits: batch_size*topk_sent
                # #### topk sentence index ####
                # pred_sids: batch_size*topk_sent

                if self.m_valid_trigram:
                    # apply trigram blocking for validation
                    s_topk_logits, s_pred_sids = self.trigram_blocking_sent_prediction(
                        s_logits, sids, s_masks, batch_size, topk=self.m_select_topk_s, pool_size=None
                    )
                elif self.m_valid_trigram_feat:
                    # apply trigram + feature unigram blocking for validation
                    s_topk_logits, s_pred_sids = self.trigram_unigram_blocking_sent_prediction(
                        s_logits, sids, s_masks, n_win=3, topk=self.m_select_topk_s, pool_size=None
                    )
                else:
                    # apply original topk selection for validation
                    s_topk_logits, s_pred_sids = self.origin_blocking_sent_prediction(
                        s_logits, sids, s_masks, topk=self.m_select_topk_s
                    )
                    # topk_logits, topk_pred_snids = torch.topk(s_logits, topk, dim=1)
                    # pred_sids = sids.gather(dim=1, index=topk_pred_snids)

                for j in range(batch_size):
                    refs_j = []
                    hyps_j = []
                    true_userid_j = userid_batch[j]
                    true_itemid_j = itemid_batch[j]

                    # for sid_k in target_sids[j]:
                    #     refs_j.append(self.m_sid2swords[sid_k.item()])
                    # refs_j = " ".join(refs_j)
                    for sid_k in s_pred_sids[j]:
                        hyps_j.append(self.m_sid2swords[sid_k.item()])

                    hyps_j = " ".join(hyps_j)
                    true_combined_ref = self.d_testset_combined[true_userid_j][true_itemid_j]

                    # scores_j = rouge.get_scores(hyps_j, refs_j, avg=True)
                    scores_j = rouge.get_scores(hyps_j, true_combined_ref, avg=True)

                    rouge_1_f_list.append(scores_j["rouge-1"]["f"])
                    rouge_1_r_list.append(scores_j["rouge-1"]["r"])
                    rouge_1_p_list.append(scores_j["rouge-1"]["p"])

                    rouge_2_f_list.append(scores_j["rouge-2"]["f"])
                    rouge_2_r_list.append(scores_j["rouge-2"]["r"])
                    rouge_2_p_list.append(scores_j["rouge-2"]["p"])

                    rouge_l_f_list.append(scores_j["rouge-l"]["f"])
                    rouge_l_r_list.append(scores_j["rouge-l"]["r"])
                    rouge_l_p_list.append(scores_j["rouge-l"]["p"])

                    # bleu_scores_j = compute_bleu([[refs_j.split()]], [hyps_j.split()])
                    bleu_scores_j = compute_bleu([[true_combined_ref.split()]], [hyps_j.split()])
                    bleu_list.append(bleu_scores_j)
                    # bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_sentence_bleu(
                    #     [refs_j.split()], hyps_j.split())
                    bleu_1_scores_j, bleu_2_scores_j, bleu_3_scores_j, bleu_4_scores_j = get_sentence_bleu(
                        [true_combined_ref.split()], hyps_j.split())
                    bleu_1_list.append(bleu_1_scores_j)
                    bleu_2_list.append(bleu_2_scores_j)
                    bleu_3_list.append(bleu_3_scores_j)
                    bleu_4_list.append(bleu_4_scores_j)

            end_time = time.time()
            duration = end_time - start_time
            print("... one epoch", duration)

            # logger_obj.f_add_scalar2tensorboard("eval/loss", np.mean(loss_list), self.m_eval_iteration)
            # logger_obj.f_add_scalar2tensorboard("eval/recall", np.mean(recall_list), self.m_eval_iteration)

        # self.m_mean_eval_loss = np.mean(loss_list)
        # self.m_mean_eval_recall = np.mean(recall_list)
        # self.m_mean_eval_precision = np.mean(precision_list)

        self.m_mean_eval_rouge_1_f = np.mean(rouge_1_f_list)
        self.m_mean_eval_rouge_1_r = np.mean(rouge_1_r_list)
        self.m_mean_eval_rouge_1_p = np.mean(rouge_1_p_list)

        self.m_mean_eval_rouge_2_f = np.mean(rouge_2_f_list)
        self.m_mean_eval_rouge_2_r = np.mean(rouge_2_r_list)
        self.m_mean_eval_rouge_2_p = np.mean(rouge_2_p_list)

        self.m_mean_eval_rouge_l_f = np.mean(rouge_l_f_list)
        self.m_mean_eval_rouge_l_r = np.mean(rouge_l_r_list)
        self.m_mean_eval_rouge_l_p = np.mean(rouge_l_p_list)

        # self.m_mean_eval_bleu = 0.0
        self.m_mean_eval_bleu = np.mean(bleu_list)
        self.m_mean_eval_bleu_1 = np.mean(bleu_1_list)
        self.m_mean_eval_bleu_2 = np.mean(bleu_2_list)
        self.m_mean_eval_bleu_3 = np.mean(bleu_3_list)
        self.m_mean_eval_bleu_4 = np.mean(bleu_4_list)

        # logger_obj.f_add_output2IO("%d, NLL_loss:%.4f"%(self.m_eval_iteration, self.m_mean_eval_loss))
        logger_obj.f_add_output2IO(
            "rouge-1:|f:%.4f |p:%.4f |r:%.4f, rouge-2:|f:%.4f |p:%.4f |r:%.4f, rouge-l:|f:%.4f |p:%.4f |r:%.4f" % (
                self.m_mean_eval_rouge_1_f, self.m_mean_eval_rouge_1_p, self.m_mean_eval_rouge_1_r,
                self.m_mean_eval_rouge_2_f, self.m_mean_eval_rouge_2_p, self.m_mean_eval_rouge_2_r,
                self.m_mean_eval_rouge_l_f, self.m_mean_eval_rouge_l_p, self.m_mean_eval_rouge_l_r))
        logger_obj.f_add_output2IO("bleu:%.4f" % (self.m_mean_eval_bleu))
        logger_obj.f_add_output2IO("bleu-1:%.4f" % (self.m_mean_eval_bleu_1))
        logger_obj.f_add_output2IO("bleu-2:%.4f" % (self.m_mean_eval_bleu_2))
        logger_obj.f_add_output2IO("bleu-3:%.4f" % (self.m_mean_eval_bleu_3))
        logger_obj.f_add_output2IO("bleu-4:%.4f" % (self.m_mean_eval_bleu_4))

        network.train()

    def ngram_blocking(self, sids, sents, p_sent, n_win, k, use_topk=True, pool_size=None):
        """ ngram blocking
        :param sids:        batch of lists of candidate sentence's sids (already converted to int). shape: [batch_size, sent_num]
        :param sents:       batch of lists of candidate sentence, each candidate sentence is a string. shape: [batch_size, sent_num]
        :param p_sent:      torch tensor. batch of predicted/relevance scores of each candidate sentence. shape: [batch_sizem, sent_num]
        :param n_win:       ngram window size, i.e. which n-gram we are using. n_win can be 2,3,4,...
        :param k:           we are selecting the top-k sentences
        :param use_topk:    whether we select the top-k sentences
        :param pool_size:   the number of the top-N sentences can be selected

        :return:        selected index of sids
        """
        batch_size = p_sent.size(0)
        batch_select_idx, batch_select_proba, batch_select_rank = [], [], []
        assert len(sents) == len(p_sent)
        assert len(sents) == batch_size
        assert len(sents[0]) == len(p_sent[0])
        for i in range(len(sents)):
            assert len(sents[i]) == len(sents[0])
            assert len(sents[i]) == len(p_sent[i])
        for batch_idx in range(batch_size):
            ngram_list = []
            # sort sentences based on the relevance score
            _, sorted_idx = p_sent[batch_idx].sort(descending=True)
            select_idx, select_proba, select_rank = [], [], []
            idx_rank = 0
            for idx in sorted_idx:
                idx_rank += 1
                if pool_size is not None and idx_rank > pool_size:
                    # this suggests that we have already searched all the cdd sents from pool
                    break
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
                    if p_sent[batch_idx][idx] <= 0.0:
                        # this suggest that this idx is already the pad idx
                        break
                    select_idx.append(idx)
                    select_proba.append(p_sent[batch_idx][idx])
                    select_rank.append(idx_rank)
                    ngram_list.extend(cur_sent_ngrams)
                    if use_topk and len(select_idx) >= k:
                        break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # # convert list to torch tensor
        # batch_select_idx = torch.LongTensor(batch_select_idx)
        return batch_select_idx, batch_select_proba, batch_select_rank

    def trigram_feat_unigram_blocking(self, sids, sents, p_sent, n_win=3, topk=5, use_feat_freq_in_sent=False, pool_size=None):
        """ a combination of trigram blocking and soft feature-unigram blocking
        :param sids:    batch of lists of candidate sentence's sids (already converted to int). shape: [batch_size, sent_num]
        :param sents:   batch of list of candidate sentence, each candidate sentence is a string.
                        shape: (batch_size, sent_num)
        :param p_sent:  torch tensor. batch of predicted scores of each candidate sentence.
                        shape: (batch_size, sent_num)
        :param topk:    we are selecting the top-k sentences.
        :param use_feat_freq_in_sent:  when compute the unigram feature word blocking,
                        using the frequency of the feature word in the sentence or only set the frequency
                        to be 1 when a feature appears in the sentence (regardless of real freq in that sent).

        :return:        selected index of sids
        """

        batch_size = p_sent.size(0)
        batch_select_idx, batch_select_proba, batch_select_rank = [], [], []
        feat_overlap_threshold = 1
        # 1. Perform trigram blocking, get the top-100 predicted sentences
        batch_select_idx_trigram, batch_select_proba_trigram, batch_select_rank_trigram = self.ngram_blocking(
            sids=sids, sents=sents, p_sent=p_sent, n_win=n_win, k=100, use_topk=True, pool_size=pool_size
        )
        # 2. Perform feature-unigram blocking
        for batch_idx in range(batch_size):
            feat_word_freq = dict()
            select_idx, select_proba, select_rank = [], [], []
            for idx, sent_idx in enumerate(batch_select_idx_trigram[batch_idx]):
                cur_sent = sents[batch_idx][sent_idx]
                cur_words = cur_sent.split()
                block_flag = False
                cur_feature_words = dict()
                for word in cur_words:
                    # check if this word is feature word
                    if word in self.d_feature2id.keys():
                        if word in cur_feature_words:
                            cur_feature_words[word] += 1
                        else:
                            cur_feature_words[word] = 1
                if use_feat_freq_in_sent:
                    for word, freq in cur_feature_words.items():
                        if word in feat_word_freq:
                            if freq + feat_word_freq[word] > feat_overlap_threshold:
                                block_flag = True
                                break
                        else:
                            if freq > 2:
                                block_flag = True
                                break
                    if not block_flag:
                        select_idx.append(sent_idx)
                        select_proba.append(batch_select_proba_trigram[batch_idx][idx])
                        select_rank.append(batch_select_rank_trigram[batch_idx][idx])
                        for word, freq in cur_feature_words.items():
                            if word in feat_word_freq:
                                feat_word_freq[word] += freq
                            else:
                                feat_word_freq[word] = freq
                else:
                    for word in cur_feature_words.keys():
                        if word in feat_word_freq:
                            if feat_word_freq[word] == feat_overlap_threshold:
                                block_flag = True
                                break
                    if not block_flag:
                        select_idx.append(sent_idx)
                        select_proba.append(batch_select_proba_trigram[batch_idx][idx])
                        select_rank.append(batch_select_rank_trigram[batch_idx][idx])
                        for word in cur_feature_words.keys():
                            if word in feat_word_freq:
                                feat_word_freq[word] += 1
                            else:
                                feat_word_freq[word] = 1
                        if len(select_idx) >= topk:
                            break
            batch_select_idx.append(select_idx)
            batch_select_proba.append(select_proba)
            batch_select_rank.append(select_rank)
        # # convert list to torch tensor, which is used for later gather element by index
        # batch_select_idx = torch.LongTensor(batch_select_idx)
        return batch_select_idx, batch_select_proba, batch_select_rank

    def origin_blocking_sent_prediction(self, s_logits, sids, s_masks, topk=3):
        # incase some not well-trained model will predict the logits for all sentences as 0.0, we apply masks on it
        # masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        s_logits = s_logits.cpu()
        # 1. get the top-k predicted sentences which form the hypothesis
        # topk_logits, topk_pred_snids = torch.topk(masked_s_logits, topk, dim=1)
        topk_logits, topk_pred_snids = torch.topk(s_logits, topk, dim=1)
        # topk sentence index
        # pred_sids: shape: (batch_size, topk_sent)
        sids = sids.cpu()
        pred_sids = sids.gather(dim=1, index=topk_pred_snids)

        return topk_logits, pred_sids

    def trigram_blocking_sent_prediction(self, s_logits, sids, s_masks, batch_size, topk=3, pool_size=None):
        # use n-gram blocking
        # get all the sentence content
        batch_sents_content = []
        sids_int = []
        assert len(sids) == s_logits.size(0)      # this is the batch size
        for i in range(batch_size):
            cur_sents_content = []
            cur_sids_int = []
            for cur_sid in sids[i]:
                cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
                cur_sids_int.append(int(cur_sid.item()))
            batch_sents_content.append(cur_sents_content)
            sids_int.append(cur_sids_int)
        # this is the max_sent_len (remember we are using zero-padding for batch data)
        assert len(batch_sents_content[0]) == len(batch_sents_content[-1])
        # masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        s_logits = s_logits.cpu()
        sids = sids.cpu()
        # get the top-k predicted sentences which form the hypothesis
        ngram_block_pred_snids, ngram_block_pred_proba, ngram_block_pred_rank = self.ngram_blocking(
            sids_int, batch_sents_content, s_logits, n_win=3, k=topk, use_topk=True, pool_size=pool_size
        )
        # pred_sids = sids.gather(dim=1, index=ngram_block_pred_snids)
        pred_sids = []
        for i in range(batch_size):
            pred_sids.append(sids[i].gather(dim=0, index=torch.tensor(ngram_block_pred_snids[i])))
        topk_logits = ngram_block_pred_proba

        return topk_logits, pred_sids

    def trigram_unigram_blocking_sent_prediction(self, s_logits, sids, s_masks, n_win=3, topk=5, pool_size=None):
        """use trigram blocking and soft unigram feature word blocking
        :param: s_logits:
        :param: sids:
        :param: s_masks:
        :param: topk:      select the top-k sentence. default: 5
        :param: topk_cdd:  sanity check. select the top-k candidate sentences, used to tune topk. default: 20
        """
        batch_sents_content = []
        sids_int = []
        assert sids.size(0) == s_logits.size(0)     # this is the batch_size
        batch_size = sids.size(0)
        for i in range(batch_size):
            cur_sents_content = []
            cur_sids_int = []
            for cur_sid in sids[i]:
                cur_sents_content.append(self.m_sid2swords[cur_sid.item()])
                cur_sids_int.append(int(cur_sid.item()))
            batch_sents_content.append(cur_sents_content)
            sids_int.append(cur_sids_int)
        # masked_s_logits = (s_logits.cpu()+1)*s_masks.cpu()-1
        s_logits = s_logits.cpu()
        sids = sids.cpu()
        # get the top-k predicted sentences which form the hypothesis
        topk_pred_snids, topk_pred_proba, topk_pred_rank = self.trigram_feat_unigram_blocking(
            sids=sids_int, sents=batch_sents_content, p_sent=s_logits, n_win=n_win,
            topk=topk, use_feat_freq_in_sent=False, pool_size=pool_size
        )
        pred_sids = []
        for i in range(batch_size):
            pred_sids.append(sids[i].gather(dim=0, index=torch.tensor(topk_pred_snids[i])))

        return topk_pred_proba, pred_sids
