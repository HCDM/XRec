import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

from GAT import GATNET
from DCN import DCN
import time
import json
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch


class GraphX(nn.Module):
    def __init__(self, args, vocab_obj, device):
        super().__init__()
        self.m_device = device
        # Load hyper-params from the dataset
        self.m_user_num = vocab_obj.user_num
        self.m_item_num = vocab_obj.item_num
        self.m_feature_num = vocab_obj.feature_num
        self.m_total_sent_num = vocab_obj.sent_num
        self.m_train_sent_num = vocab_obj.train_sent_num
        # Load hyper-params from the command line
        self.m_hidden_size = args.hidden_size
        self.m_feature_finetune_flag = args.feat_finetune
        self.m_sentence_finetune_flag = args.sent_finetune
        self.m_cross_num = args.cross_num
        self.m_cross_type = args.cross_type
        self.m_dnn_hidden_units = args.dnn_hidden_units
        self.m_dnn_use_bn = args.dnn_use_bn
        self.m_useritem_pretrain = args.useritem_pretrain

        if args.cond_sentence is not None:
            self.m_conditioned_sentence_flag = True
            self.m_conditioned_sentence_model = args.cond_sentence
        else:
            self.m_conditioned_sentence_flag = False
            self.m_conditioned_sentence_model = None

        if self.m_conditioned_sentence_flag:
            print("sentence predition conditioned on user&item, using {}".format(
                self.m_conditioned_sentence_model
            ))
        else:
            print("sentence predition only based on sentence node")
        print("user num", self.m_user_num)
        print("item num", self.m_item_num)
        print("feature num", self.m_feature_num)
        print("total sent num", self.m_total_sent_num)
        print("train sent num", self.m_train_sent_num)

        self.m_user_embed = nn.Embedding(self.m_user_num, args.user_embed_size)
        self.m_item_embed = nn.Embedding(self.m_item_num, args.item_embed_size)
        if args.useritem_pretrain:
            print("load user and item pretrain embedding")
            self.f_load_user_embedding(vocab_obj.m_user2uid, args.data_dir+args.user_embed_file)
            self.f_load_item_embedding(vocab_obj.m_item2iid, args.data_dir+args.item_embed_file)

        self.m_feature_embed = nn.Embedding(self.m_feature_num, args.feature_embed_size)
        # self.m_feature_embed = vocab_obj.m_fid2fembed
        self.m_feature_embed_size = args.feature_embed_size
        self.f_load_feature_embedding(vocab_obj.m_fid2fembed)

        self.m_sent_embed = nn.Embedding(self.m_train_sent_num, args.sent_embed_size)
        # self.m_sent_embed = vocab_obj.m_sid2sembed
        self.m_sent_embed_size = args.sent_embed_size
        self.f_load_sent_embedding(vocab_obj.m_sid2sembed)

        self.sent_state_proj = nn.Linear(args.sent_embed_size, args.hidden_size, bias=False)
        self.feature_state_proj = nn.Linear(args.feature_embed_size, args.hidden_size, bias=False)
        self.user_state_proj = nn.Linear(args.user_embed_size, args.hidden_size, bias=False)
        self.item_state_proj = nn.Linear(args.item_embed_size, args.hidden_size, bias=False)

        self.m_gat = GATNET(in_dim=args.hidden_size, out_dim=args.hidden_size, head_num=args.head_num, dropout_rate=args.attn_dropout_rate)
        ### node classification
        # self.output_hidden_size = args.output_hidden_size
        # self.wh = nn.Linear(self.output_hidden_size * 2, 2)
        # self.wh = nn.Linear(args.hidden_size, 2)
        if self.m_conditioned_sentence_flag:
            if self.m_conditioned_sentence_model == 'bilinear':
                self.sent_ui_cond = nn.Linear(args.hidden_size*2, args.hidden_size)
                print("Using bilinear.")
            elif self.m_conditioned_sentence_model == 'dcn':
                try:
                    assert self.m_cross_num > 0
                except AssertionError:
                    print("Number of Cross layers in DCN should be at least 1, got 0")
                try:
                    assert len(self.m_dnn_hidden_units) > 0
                except AssertionError:
                    print("Number of DNN layers in DCN should be at least 1, got 0")
                try:
                    assert self.m_cross_type == 'vector' or self.m_cross_type == 'matrix'
                except AssertionError:
                    print("Cross layer weight format should be either vector or matrix, got {}".format(
                        self.m_cross_type))
                print("Using DCN.\n{0} layers of cross network, type of weight: {1}".format(
                    self.m_cross_num, self.m_cross_type))
                print("hidden units of the dnn: {0}, using batch normalization: {1}".format(
                    self.m_dnn_hidden_units, self.m_dnn_use_bn))
                self.sent_ui_cond = DCN(
                    node_embed_size=args.hidden_size,
                    cross_num=self.m_cross_num,
                    cross_parameterization=self.m_cross_type,
                    dnn_hidden_units=self.m_dnn_hidden_units,
                    init_std=0.0001, dnn_dropout=0, dnn_activation='relu',
                    dnn_use_bn=self.m_dnn_use_bn, device=self.m_device
                )
            else:
                print("conditional sentence prediction model incorrect, can't be: {}".format(
                    self.m_conditioned_sentence_model))
                exit()
        else:
            self.sent_output = nn.Linear(args.hidden_size, 1)

        self.feat_output = nn.Linear(args.hidden_size, 1)

        self.f_initialize()

        self = self.to(self.m_device)

    def f_initialize(self):
        if not self.m_useritem_pretrain:
            nn.init.uniform_(self.m_user_embed.weight, a=-1e-3, b=1e-3)
            nn.init.uniform_(self.m_item_embed.weight, a=-1e-3, b=1e-3)

        nn.init.uniform_(self.sent_state_proj.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.feature_state_proj.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.user_state_proj.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.item_state_proj.weight, a=-1e-3, b=1e-3)

        # nn.init.uniform_(self.wh.weight, a=-1e-3, b=1e-3)
        if self.m_conditioned_sentence_flag:
            if self.m_conditioned_sentence_model == 'bilinear':
                nn.init.uniform_(self.sent_ui_cond.weight, a=-1e-3, b=1e-3)
            elif self.m_conditioned_sentence_model == 'dcn':
                # dcn's init is handled in its own class
                pass
        else:
            nn.init.uniform_(self.sent_output.weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.feat_output.weight, a=-1e-3, b=1e-3)

    def f_load_user_embedding(self, user2uid_vocab, user_embed_file):
        pre_user_embed_weight = []
        uid2uembed = {}
        with open(user_embed_file, 'r') as f_u:
            user_embed_dict = json.load(f_u)
        for userid_i in list(user2uid_vocab.keys()):
            user_embed_i = user_embed_dict[userid_i]
            uid_i = user2uid_vocab[userid_i]
            assert uid_i not in uid2uembed
            uid2uembed[uid_i] = user_embed_i
        print("pre_user_embed", len(uid2uembed))

        for u_idx in range(self.m_user_num):
            user_embed_i = uid2uembed[u_idx]
            pre_user_embed_weight.append(user_embed_i)

        self.m_user_embed.weight.data.copy_(torch.Tensor(pre_user_embed_weight))

    def f_load_item_embedding(self, item2iid_vocab, item_embed_file):
        pre_item_embed_weight = []
        iid2iembed = {}
        with open(item_embed_file, 'r') as f_i:
            item_embed_dict = json.load(f_i)
        for itemid_i in list(item2iid_vocab.keys()):
            item_embed_i = item_embed_dict[itemid_i]
            iid_i = item2iid_vocab[itemid_i]
            assert iid_i not in iid2iembed
            iid2iembed[iid_i] = item_embed_i
        print("pre_item_embed", len(iid2iembed))

        for i_idx in range(self.m_item_num):
            user_embed_i = iid2iembed[i_idx]
            pre_item_embed_weight.append(user_embed_i)

        self.m_item_embed.weight.data.copy_(torch.Tensor(pre_item_embed_weight))

    def f_load_feature_embedding(self, pre_feature_embed):

        pre_feature_embed_weight = []
        print("pre_feature_embed", len(pre_feature_embed))

        for f_idx in range(self.m_feature_num):
            feature_embed_i = pre_feature_embed[f_idx]
            pre_feature_embed_weight.append(feature_embed_i)

        self.m_feature_embed.weight.data.copy_(torch.Tensor(pre_feature_embed_weight))

        if not self.m_feature_finetune_flag:
            self.m_feature_embed.weight.requires_grad = False

    def f_load_sent_embedding(self, pre_sent_embed):

        print("pre_sent_embed", len(pre_sent_embed))
        print("sent num", self.m_train_sent_num)

        pre_sent_embed_weight = []

        for s_idx in range(self.m_train_sent_num):
            sent_embed_i = pre_sent_embed[s_idx]
            pre_sent_embed_weight.append(sent_embed_i)

        print("pre_sent_embed_weight", len(pre_sent_embed_weight))
        print("sent num", self.m_train_sent_num)

        self.m_sent_embed.weight.data.copy_(torch.Tensor(pre_sent_embed_weight))
        if not self.m_sentence_finetune_flag:
            self.m_sent_embed.weight.requires_grad = False

    def forward(self, graph_batch):
        ## init node embeddings
        # print("graph_batch", graph_batch.num_graphs)
        batch_size = graph_batch.num_graphs
        ##### feature node
        fid = graph_batch.f_rawid
        f_embed = self.m_feature_embed(fid)
        f_node_embed = self.feature_state_proj(f_embed)

        ##### sentence node
        sid = graph_batch.s_rawid
        s_embed = self.m_sent_embed(sid)
        s_node_embed = self.sent_state_proj(s_embed)

        ##### item node
        itemid = graph_batch.i_rawid
        item_embed = self.m_item_embed(itemid)
        item_node_embed = self.item_state_proj(item_embed)

        ##### user node
        userid = graph_batch.u_rawid
        user_embed = self.m_user_embed(userid)
        user_node_embed = self.user_state_proj(user_embed)

        # print("f_node_embed", f_node_embed.size())
        # print("s_node_embed", s_node_embed.size())
        # print("user_node_embed", user_node_embed.size())
        # print("item_node_embed", item_node_embed.size())

        batch_fnum = graph_batch.f_num
        batch_snum = graph_batch.s_num

        batch_cumsum_fnum = torch.cumsum(batch_fnum, dim=0)
        last_cumsum_fnum_i = 0

        batch_cumsum_snum = torch.cumsum(batch_snum, dim=0)
        last_cumsum_snum_i = 0

        # batch_nnum = graph_batch.num_nodes
        # print(batch_nnum)

        x_batch = []
        for i in range(batch_size):
            cumsum_fnum_i = batch_cumsum_fnum[i]
            x_batch.append(f_node_embed[last_cumsum_fnum_i:cumsum_fnum_i])
            last_cumsum_fnum_i = cumsum_fnum_i

            cumsum_snum_i = batch_cumsum_snum[i]
            x_batch.append(s_node_embed[last_cumsum_snum_i:cumsum_snum_i])
            last_cumsum_snum_i = cumsum_snum_i

            x_batch.append(user_node_embed[i].unsqueeze(0))
            x_batch.append(item_node_embed[i].unsqueeze(0))

            nnum_i = graph_batch[i].num_nodes
            debug_nnum_i = batch_fnum[i]+batch_snum[i]+2
            assert nnum_i == debug_nnum_i, "error node num"

        x = torch.cat(x_batch, dim=0)

        # x = torch.cat([f_node_embed, s_node_embed, user_node_embed, item_node_embed], dim=0)
        graph_batch["x"] = x

        ## go through GAT
        #### hidden: node_num*hidden_size
        hidden = self.m_gat(graph_batch.x, graph_batch.edge_index)

        #### hidden_batch: batch_size*max_node_size_per_g*hidden_size
        hidden_batch, _ = to_dense_batch(hidden, graph_batch.batch)

        ### list of sent_node_num*hidden_size
        hidden_s = []
        hidden_f = []
        hidden_ui = []

        ### speed up
        ## fetch sentence hidden vectors from graph
        for batch_idx in range(batch_size):
            g = graph_batch[batch_idx]
            hidden_g_i = hidden_batch[batch_idx]

            # get sentence node hidden state
            s_nid = g.s_nid
            # hidden_s_g_i: s_nid.size(0)*hidden_size
            hidden_s_g_i = hidden_g_i[s_nid]
            hidden_s.append(hidden_s_g_i)

            # get feature node hidden state
            f_nid = g.f_nid
            # hidden_f_g_i: f_nid.size(0)*hidden_size
            hidden_f_g_i = hidden_g_i[f_nid]
            hidden_f.append(hidden_f_g_i)

            if self.m_conditioned_sentence_flag:
                # get user node hidden state
                u_nid = g.u_nid
                hidden_u_g_i = hidden_g_i[u_nid]
                # get item node hidden state
                i_nid = g.i_nid
                hidden_i_g_i = hidden_g_i[i_nid]
                # hidden_ui_g_i: 1*(hidden_size*2)
                hidden_ui_g_i = torch.cat([hidden_u_g_i, hidden_i_g_i], dim=-1)
                # hidden_ui_g_i_s: s_nid.size(0)*(hidden_size*2)
                # ## hidden_ui_g_i_s = hidden_ui_g_i.repeat(s_nid.size(0), 1)
                hidden_ui_g_i_s = hidden_ui_g_i.expand(s_nid.size(0), -1)
                # print("hidden_ui_g_i_s shape:", hidden_ui_g_i_s.shape)
                hidden_ui.append(hidden_ui_g_i_s)

        ### make predictions
        if self.m_conditioned_sentence_flag:
            # hidden_ui: s_node_num*(hidden_size*2)
            hidden_ui = torch.cat(hidden_ui, dim=0)
            # print("hidden_ui shape:", hidden_ui.shape)
            # hidden_s: s_node_num*hidden_size
            hidden_s = torch.cat(hidden_s, dim=0)
            # print("hidden_s shape:", hidden_s.shape)
            assert hidden_s.size(0) == hidden_ui.size(0)
            if self.m_conditioned_sentence_model == 'bilinear':
                # hidden_ui: s_node_num*hidden_size
                hidden_ui = self.sent_ui_cond(hidden_ui)
                # print("after linear layer, hidden_ui shape:", hidden_ui.shape)
                # logits_s: s_node_num*1
                logits_s = torch.bmm(
                    hidden_ui.unsqueeze(1),
                    hidden_s.unsqueeze(-1)
                ).squeeze(-1)
            elif self.m_conditioned_sentence_model == 'dcn':
                # concat ui_embed with sent_embed, hidden_uis: s_node_num*(hidden_size*3)
                hidden_uis = torch.cat([hidden_ui, hidden_s], dim=-1)
                # print("hidden_uis shape: {}".format(hidden_uis.shape))
                logits_s = self.sent_ui_cond(hidden_uis)
                # print("after dcn, logits_s shape:", logits_s.shape)
            else:
                print("conditional sentence prediction model incorrect, can't be: {}".format(
                    self.m_conditioned_sentence_model))
                exit()
        else:
            ### hidden_s_batch: s_node_num*hidden_size
            hidden_s = torch.cat(hidden_s, dim=0)
            ### logits: s_node_num*1
            logits_s = self.sent_output(hidden_s)

        hidden_f = torch.cat(hidden_f, dim=0)
        logits_f = self.feat_output(hidden_f)

        return logits_s, logits_f

    def eval_forward(self, graph_batch, get_embedding=False):
        batch_size = graph_batch.num_graphs
        ##### feature node
        fid = graph_batch.f_rawid
        f_embed = self.m_feature_embed(fid)
        f_node_embed = self.feature_state_proj(f_embed)

        ##### sentence node
        sid = graph_batch.s_rawid
        s_embed = self.m_sent_embed(sid)
        s_node_embed = self.sent_state_proj(s_embed)

        ##### item node
        itemid = graph_batch.i_rawid
        item_embed = self.m_item_embed(itemid)
        item_node_embed = self.item_state_proj(item_embed)

        ##### user node
        userid = graph_batch.u_rawid
        user_embed = self.m_user_embed(userid)
        user_node_embed = self.user_state_proj(user_embed)

        batch_fnum = graph_batch.f_num
        batch_snum = graph_batch.s_num

        batch_cumsum_fnum = torch.cumsum(batch_fnum, dim=0)
        last_cumsum_fnum_i = 0

        batch_cumsum_snum = torch.cumsum(batch_snum, dim=0)
        last_cumsum_snum_i = 0

        x_batch = []
        for i in range(batch_size):
            cumsum_fnum_i = batch_cumsum_fnum[i]
            x_batch.append(f_node_embed[last_cumsum_fnum_i:cumsum_fnum_i])
            last_cumsum_fnum_i = cumsum_fnum_i

            cumsum_snum_i = batch_cumsum_snum[i]
            x_batch.append(s_node_embed[last_cumsum_snum_i:cumsum_snum_i])
            last_cumsum_snum_i = cumsum_snum_i

            x_batch.append(user_node_embed[i].unsqueeze(0))
            x_batch.append(item_node_embed[i].unsqueeze(0))

        x = torch.cat(x_batch, dim=0)

        # x = torch.cat([f_node_embed, s_node_embed, user_node_embed, item_node_embed], dim=0)
        graph_batch["x"] = x

        ## go through GAT
        #### hidden: node_num*hidden_size
        hidden = self.m_gat(graph_batch.x, graph_batch.edge_index)

        #### hidden_batch: batch_size*max_node_size_per_g*hidden_size
        hidden_batch, mask_batch = to_dense_batch(hidden, graph_batch.batch)

        ### list of 1*max_sent_node_num_per_g*hidden_size
        hidden_s_batch = []
        hidden_ui_batch = []
        sid_batch = []
        mask_s_batch = []
        target_sid_batch = []
        max_s_num_batch = 0

        hidden_f_batch = []
        fid_batch = []
        mask_f_batch = []
        target_f_label_batch = []
        max_f_num_batch = 0

        for batch_idx in range(batch_size):
            g = graph_batch[batch_idx]
            hidden_g_i = hidden_batch[batch_idx]

            s_nid = g.s_nid
            s_num = s_nid.size(0)
            max_s_num_batch = max(max_s_num_batch, s_num)

            f_nid = g.f_nid
            f_num = f_nid.size(0)

            max_f_num_batch = max(max_f_num_batch, f_num)

        ## fetch sentence hidden vectors
        for batch_idx in range(batch_size):
            g = graph_batch[batch_idx]
            hidden_g_i = hidden_batch[batch_idx]

            s_nid = g.s_nid
            s_num = s_nid.size(0)
            pad_s_num = max_s_num_batch-s_num

            #### sent_num*hidden_size
            hidden_s_g_i = hidden_g_i[s_nid]
            pad_s_g_i = torch.zeros(pad_s_num, hidden_s_g_i.size(1)).to(self.m_device)
            hidden_pad_s_g_i = torch.cat([hidden_s_g_i, pad_s_g_i], dim=0)
            hidden_s_batch.append(hidden_pad_s_g_i.unsqueeze(0))
            if self.m_conditioned_sentence_flag:
                u_nid = g.u_nid
                hidden_u_g_i = hidden_g_i[u_nid]
                i_nid = g.i_nid
                hidden_i_g_i = hidden_g_i[i_nid]
                hidden_ui_g_i = torch.cat([hidden_u_g_i, hidden_i_g_i], dim=-1)
                hidden_ui_g_i_s = hidden_ui_g_i.repeat(max_s_num_batch, 1)
                # print("hidden_ui_g_i_s shape:", hidden_ui_g_i_s.shape)
                hidden_ui_batch.append(hidden_ui_g_i_s.unsqueeze(0))

            #### pad id can be further improved
            sid = g.s_rawid
            pad_sid_i = torch.zeros(pad_s_num).to(self.m_device)
            sid_pad_i = torch.cat([sid, pad_sid_i], dim=0)
            sid_batch.append(sid_pad_i.unsqueeze(0))

            mask_s = torch.zeros(max_s_num_batch).to(self.m_device)
            mask_s[:s_num] = 1
            mask_s_batch.append(mask_s.unsqueeze(0))

            ### change gt_label: gt sent raw id
            target_sid = g.gt_label
            target_sid_batch.append(target_sid)

            f_nid = g.f_nid
            f_num = f_nid.size(0)
            pad_f_num = max_f_num_batch-f_num

            hidden_f_g_i = hidden_g_i[f_nid]
            pad_f_g_i = torch.zeros(pad_f_num, hidden_f_g_i.size(1)).to(self.m_device)
            hidden_pad_f_g_i = torch.cat([hidden_f_g_i, pad_f_g_i], dim=0)
            hidden_f_batch.append(hidden_pad_f_g_i.unsqueeze(0))

            fid = g.f_rawid
            pad_fid_i = torch.zeros(pad_f_num).to(self.m_device)
            fid_pad_i = torch.cat([fid, pad_fid_i], dim=0)
            fid_batch.append(fid_pad_i.unsqueeze(0))

            mask_f = torch.zeros(max_f_num_batch).to(self.m_device)
            mask_f[:f_num] = 1
            mask_f_batch.append(mask_f.unsqueeze(0))

            # pad_f_label_i = torch.zeros(pad_f_num).to(self.m_device)
            # target_f_label_i = torch.cat([g.f_label, pad_f_label_i], dim=0)
            # target_f_label_batch.append(target_f_label_i.unsqueeze(0))

            target_f_label_i = g.f_label
            target_f_label_batch.append(target_f_label_i)

        # hidden_s_batch: batch_size*max_s_num_batch*hidden_size
        hidden_s_batch = torch.cat(hidden_s_batch, dim=0)

        # sid_batch: batch_size*max_s_num_batch
        sid_batch = torch.cat(sid_batch, dim=0)

        # mask_s_batch: batch_size*max_s_num_batch
        mask_s_batch = torch.cat(mask_s_batch, dim=0)

        # make predictions
        if self.m_conditioned_sentence_flag:
            # hidden_ui_batch: batch_size*max_s_num_batch*(2 hidden_size)
            hidden_ui_batch = torch.cat(hidden_ui_batch, dim=0)
            # print("hidden_ui_batch:", hidden_ui_batch.shape)
            if self.m_conditioned_sentence_model == 'bilinear':
                # s_logits: batch_size*max_s_num_batch*hidden_size
                s_logits = self.sent_ui_cond(hidden_ui_batch)
                # print("s_logits shape:", s_logits.shape)
                # s_logits: batch_size*max_s_num_batch
                """ bmm
                bmm:
                    (batch_size*max_s_num_batch, 1, hidden_size) *
                    (batch_size*max_s_num_batch, hidden_size, 1)
                after bmm:
                    (batch_size*max_s_num_batch, 1, 1)
                after reshape:
                    (batch_size, max_s_num_batch)
                """
                s_logits = torch.bmm(
                    s_logits.unsqueeze(2).view(-1, 1, self.m_hidden_size),
                    hidden_s_batch.unsqueeze(-1).view(-1, self.m_hidden_size, 1)
                ).view(batch_size, -1)
                # print("after bmm, s_logits shape:", s_logits.shape)
                # s_logits: batch_size*max_s_num_batch
                s_logits = torch.sigmoid(s_logits)*mask_s_batch
                # print("after sigmoid, s_logits shape:", s_logits.shape)
            elif self.m_conditioned_sentence_model == 'dcn':
                # hidden_uis_batch: (batch_size, max_s_num_batch, (hidden_size*3))
                hidden_uis_batch = torch.cat([hidden_ui_batch, hidden_s_batch], dim=-1)
                # print("hidden_uis_batch shape:", hidden_uis_batch.shape)
                uis_hidden_size = hidden_uis_batch.size(-1)
                s_logits = self.sent_ui_cond(
                    hidden_uis_batch.view(-1, uis_hidden_size)).view(batch_size, -1)
                # print("after dcn, s_logits shape:", s_logits.shape)
                # s_logits: (batch_size, max_s_num_batch)
                s_logits = torch.sigmoid(s_logits)*mask_s_batch
                # print("after sigmoid, s_logits shape:", s_logits.shape)
            else:
                print("conditional sentence prediction model incorrect, can't be: {}".format(
                    self.m_conditioned_sentence_model))
                exit()

        else:
            # s_logits: batch_size*max_s_num_batch*1
            s_logits = self.sent_output(hidden_s_batch)
            # s_logits: batch_size*max_s_num_batch
            s_logits = s_logits.squeeze(-1)
            s_logits = torch.sigmoid(s_logits)*mask_s_batch

        hidden_f_batch = torch.cat(hidden_f_batch, dim=0)
        fid_batch = torch.cat(fid_batch, dim=0)
        mask_f_batch = torch.cat(mask_f_batch, dim=0)

        f_logits = self.feat_output(hidden_f_batch)
        f_logits = f_logits.squeeze(-1)
        f_logits = torch.sigmoid(f_logits)*mask_f_batch

        if not get_embedding:
            return (s_logits, sid_batch, mask_s_batch, target_sid_batch,
                    f_logits, fid_batch, mask_f_batch, target_f_label_batch, hidden_f_batch)
        else:
            # print(graph_batch.x.shape)
            graph_batch_x, mask_graph_batch_x = to_dense_batch(graph_batch.x, graph_batch.batch)
            # print(graph_batch_x.shape)
            # print(mask_graph_batch_x.shape)
            return (s_logits, sid_batch, mask_s_batch, target_sid_batch,
                    f_logits, fid_batch, mask_f_batch, target_f_label_batch,
                    hidden_f_batch, graph_batch_x, mask_graph_batch_x)
