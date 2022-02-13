"""
g:
f_nid
f_rawid

s_nid
s_rawid

u_nid
u_rawid

i_nid
i_rawid

label: [0, 1, ..., 1]
gt_label: [sent id]
"""


import os
import json
import torch
import numpy as np
# from torch.utils.data import Dataset
import pandas as pd
import argparse
import copy
from collections import Counter
# import dgl
# from dgl.data.utils import save_graphs, load_graphs

from torch_geometric.data import Dataset

def readJson(fname):
    """
    read json file which contains multiple lines of data
    """
    data = []
    line_num = 0
    with open(fname, encoding="utf-8") as f:
        for line in f:
            # print("line", line)
            line_num += 1
            try:
                data.append(json.loads(line))
            except:
                print("error", line_num)
    return data


class Vocab():
    def __init__(self):
        # mapping user/item to id
        self.m_user2uid = {}
        self.m_item2iid = {}
        # number of user/item
        self.m_user_num = 0
        self.m_item_num = 0
        # mapping feature to id
        self.m_feature2fid = {}
        self.m_feature_num = 0
        # mapping sentence to id
        self.m_sent2sid = {}
        self.m_sent_num = 0
        # mapping feature id to feature embedding
        self.m_fid2fembed = {}
        # mapping sentence id to sentence embedding
        self.m_sid2sembed = {}
        # count number of sentence on train/test set
        self.m_train_sent_num = 0
        self.m_test_sent_num = 0

    def f_set_user2uid_vocab(self, user2uid):
        self.m_user2uid = user2uid
        self.m_user_num = len(self.m_user2uid)

    def f_set_item2iid_vocab(self, item2iid):
        self.m_item2iid = item2iid
        self.m_item_num = len(self.m_item2iid)

    def f_set_feature2fid_vocab(self, feature2fid):
        self.m_feature2fid = feature2fid
        self.m_feature_num = len(self.m_feature2fid)

    def f_set_sent2sid_vocab(self, sent2sid):
        self.m_sent2sid = sent2sid
        self.m_sent_num = len(self.m_sent2sid)

    def f_load_sent_content_train(self, sent_content_file):

        self.m_sid2swords = {}
        # Load the id2sentence mapping which contains the sentence content
        sent_content = readJson(sent_content_file)[0]
        # keys of sent_content are sentence_ids
        sentid_list = list(sent_content.keys())
        # number of sentence is the len of the sentence_ids list
        sent_num = len(sent_content)
        for sent_idx in range(sent_num):
            # get current sentence id
            sentid_i = sentid_list[sent_idx]
            # Construct the mapping between sentence id and its relative order in the sentid_list
            # so that the trainset sentence ids can be combined with testset sentence ids
            if sentid_i not in self.m_sent2sid:
                sid_i = len(self.m_sent2sid)
                self.m_sent2sid[sentid_i] = sid_i

            sid_i = self.m_sent2sid[sentid_i]

            sentwords_i = sent_content[sentid_i]

            self.m_sid2swords[sid_i] = sentwords_i

        print("load sent num train", len(self.m_sid2swords))

    def f_load_sent_content_eval(self, sent_content_file):
        # Load id2sentence (eval) mapping which contains the sentence content
        sent_content = readJson(sent_content_file)[0]
        # Get sentence id (eval)
        sentid_list = list(sent_content.keys())
        # m_sent2sid is already being feeded by the sentence id on train set
        train_sent_num = len(self.m_sent2sid)
        self.m_train_sent_num = train_sent_num
        print("train_sent_num", train_sent_num)
        sent_num = len(sent_content)
        for sent_idx in range(sent_num):
            sentid_i = sentid_list[sent_idx]
            sentwords_i = sent_content[sentid_i]
            # add number of train sentence to sentid_i (eval)
            sentid_i = train_sent_num+int(sentid_i)
            sentid_i = str(sentid_i)

            if sentid_i not in self.m_sent2sid:
                sid_i = len(self.m_sent2sid)
                self.m_sent2sid[sentid_i] = sid_i

            sid_i = self.m_sent2sid[sentid_i]

            self.m_sid2swords[sid_i] = sentwords_i

        print("load sent num eval", sent_num)
        print("total sent num", len(self.m_sent2sid))

    def f_load_sent_embed(self, sent_embed_file):
        ### sid 2 embed
        sent2sid_dict = self.m_sent2sid

        sent_embed = readJson(sent_embed_file)
        sent_embed_num = len(sent_embed)
        print("sent num", sent_embed_num)

        #### sent_embed {sentid: embed}
        for i in range(sent_embed_num):
            data_i = sent_embed[i]

            sentid_i = list(data_i.keys())[0]
            sentembed_i = data_i[sentid_i]

            if sentid_i not in sent2sid_dict:
                print("error missing sent", sentid_i)
                continue

            sid_i = sent2sid_dict[sentid_i]
            if sid_i not in self.m_sid2sembed:
                self.m_sid2sembed[sid_i] = sentembed_i

    def f_load_feature_embed(self, feature_embed_file):
        self.m_feature2fid = {}
        self.m_fid2fembed = {}

        feature_embed = readJson(feature_embed_file)[0]
        feature_embed_num = len(feature_embed)
        print("feature_embed_num", feature_embed_num)

        featureid_list = list(feature_embed.keys())
        for featureid_i in featureid_list:
            featureembed_i = feature_embed[featureid_i]

            if featureid_i not in self.m_feature2fid:
                fid_i = len(self.m_feature2fid)
                self.m_feature2fid[featureid_i] = fid_i

            fid_i = self.m_feature2fid[featureid_i]
            if fid_i not in self.m_fid2fembed:
                self.m_fid2fembed[fid_i] = featureembed_i

    @property
    def user_num(self):
        self.m_user_num = len(self.m_user2uid)
        return self.m_user_num

    @property
    def item_num(self):
        self.m_item_num = len(self.m_item2iid)
        return self.m_item_num

    @property
    def feature_num(self):
        self.m_feature_num = len(self.m_feature2fid)
        return self.m_feature_num

    @property
    def sent_num(self):
        self.m_sent_num = len(self.m_sent2sid)
        return self.m_sent_num

    @property
    def train_sent_num(self):
        return self.m_train_sent_num


class RATEBEER(Dataset):
    def __init__(self):
        super().__init__()
        # user-feature | item-feature | sentence-feature
        # edges and the tf-idf weight
        self.m_uid2fid2tfidf_dict = {}
        self.m_iid2fid2tfidf_dict = {}
        self.m_sid2fid2tfidf_dict = {}

        self.m_uid_list = []
        self.m_iid_list = []
        self.m_cdd_sid_list_list = []
        self.m_label_sid_list_list = []

        self.m_graph_path = ""

    def load_train_graph_data(self, graph_path):
        print("... train ...", graph_path)
        self.m_graph_path = graph_path

    def load_eval_graph_data(self, graph_path):
        print("... eval ...", graph_path)
        self.m_graph_path = graph_path

    def __len__(self):
        graph_path = self.m_graph_path
        print("... graph path ...", self.m_graph_path)

        file_num = 0

        graph_summary_file = "graph_summary.txt"
        graph_summary_file = os.path.join(graph_path, graph_summary_file)

        if os.path.isfile(graph_summary_file):
            with open(graph_summary_file, "r") as f:
                line_val = f.readline()
                file_num = line_val.strip()
                file_num = int(file_num)
        else:
            for file_name in os.listdir(graph_path):
                abs_file_name = os.path.join(graph_path, file_name)
                if os.path.isfile(abs_file_name):
                    file_num += 1
                else:
                    print(file_name)

        # file_num = 1000
        print("file_num", file_num)
        return file_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = idx
        g_file_i = self.m_graph_path+str(i)+".pt"

        g_i = torch.load(g_file_i)
        # print("gt_label", g_i.gt_label)

        return g_i
