import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import argparse
import copy
from collections import Counter 
import dgl
from dgl.data.utils import save_graphs, load_graphs


class Vocab():
    def __init__(self):

        self.m_user2uid = None
        self.m_item2iid = None

        self.m_user_num = 0
        self.m_item_num = 0

        self.m_feature2fid = None
        self.m_feature_num = 0

        self.m_sent2sid = None
        self.m_sent_num = 0

    def f_set_vocab(self, user2uid, item2iid, feature2fid, sent2sid):
        self.m_user2uid = user2uid
        self.m_item2iid = item2iid
        self.m_feature2fid = feature2fid
        self.m_sent2sid = sent2sid
        
        self.m_user_num = len(self.m_user2uid)
        self.m_item_num = len(self.m_item2iid)
        self.m_feature_num = len(self.m_feature2fid)
        self.m_sent_num = len(self.m_sent2sid)

    @property
    def user_num(self):
        return self.m_user_num
    
    @property
    def item_num(self):
        return self.m_item_num

    @property
    def feature_num(self):
        return self.m_feature_num


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

class RATEBEER(Dataset):
    def __init__(self, args, useritem_sent_file, sent_embed_file, sent_feature_file, user_feature_file, item_feature_file):
        super().__init__()

        self.m_data_dir = args.data_dir
        self.m_batch_size = args.batch_size
        
        #### read pair data 
        user2uid_dict = {}
        item2iid_dict = {}
        sent2sid_dict = {}

        uid_list = []
        iid_list = []
        sid_list_list = []
        fid_list_list = []

        #### useritem_sent {userid: {itemid: [sentid]}}
        useritem_sent = readJson(useritem_sent_file)
        useritem_sent_num = len(useritem_sent)
        print("useritem_sent_num", useritem_sent_num)

        for i in range(useritem_sent_num):
            data_i = useritem_sent[i]

            userid_i = list(data_i.keys())[0]
            itemid_list_i = list(data_i[userid_i].keys())

            for itemid_ij in itemid_list_i:
                sentid_list_ij = data_i[userid_i][itemid_ij]

                if userid_i not in user2uid_dict:
                    user2uid_dict[userid_i] = len(user2uid_dict)

                if itemid_ij not in item2iid_dict:
                    item2iid_dict[itemid_ij] = len(item2iid_dict)

                sid_list_i = []
                for sentid_ijk in sentid_list_ij:
                    if len(sentid_ijk) == 0:
                        continue
                        
                    if sentid_ijk not in sent2sid_dict:
                        sent2sid_dict[sentid_ijk] = len(sent2sid_dict)
                        sid_list_i.append(sent2sid_dict[sentid_ijk])

                uid_i = user2uid_dict[userid_i]
                iid_i = item2iid_dict[itemid_ij]
                
                uid_list.append(uid_i)
                iid_list.append(iid_i)
                sid_list_list.append(sid_list_i)

        ### sid 2 embed
        sid2embed_dict = {}

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

            if sentid_i not in sid2embed_dict:
                sid_i = sent2sid_dict[sentid_i]
                sid2embed_dict[sid_i] = sentembed_i

        ### sent_feature {sentid: {featureid: feature tf-idf}}
        sid2fid2tfidf_dict = {}

        sentid2fid2tfidf = readJson(sent_feature_file)
        sent_num = len(sentid2fid2tfidf)

        feature2fid_dict = {}

        if sent_num != sent_embed_num:
            print("sent num error", sent_num, sent_embed_num)

        for i in range(sent_num):
            data_i = sentid2fid2tfidf[i]

            sentid_i = list(data_i.keys())[0]
            featureid_tfidf_dict_i = data_i[sentid_i]

            sid_i = sent2sid_dict[sentid_i]
            if sid_i not in sid2fid2tfidf_dict:
                sid2fid2tfidf_dict[sid_i] = {}

            for feautreid_ij in featureid_tfidf_dict_i:
                if feautreid_ij not in feature2fid_dict:
                    feature2fid_dict[feautreid_ij] = len(feature2fid_dict)
                
                fid_ij = feature2fid_dict[feautreid_ij]
                tfidf_ij = featureid_tfidf_dict_i[feautreid_ij]
                
                sid2fid2tfidf_dict[sid_i][fid_ij] = tfidf_ij
                

        ### user_feature {userid: {featureid: feature tf-idf}}
        uid2fid2tfidf_dict = {}

        userid2fid2tfidf = readJson(user_feature_file)
        user_num = len(userid2fid2tfidf)

        if user_num != len(user2uid_dict):
            print("user num error", user_num, len(user2uid_dict))

        for i in range(user_num):
            data_i = userid2fid2tfidf[i]

            userid_i = list(data_i.keys())[0]
            featureid_tfidf_dict_i = data_i[userid_i]

            uid_i = user2uid_dict[userid_i]
            if uid_i not in uid2fid2tfidf_dict:
                uid2fid2tfidf_dict[uid_i] = {}

            for feautreid_ij in featureid_tfidf_dict_i:
                if feautreid_ij not in feature2fid_dict:
                    print("error missing feature", userid_i, feautreid_ij)
                
                fid_ij = feature2fid_dict[feautreid_ij]
                tfidf_ij = featureid_tfidf_dict_i[feautreid_ij]
                
                uid2fid2tfidf_dict[uid_i][fid_ij] = tfidf_ij

        ### item_feature {itemid: {featureid: feature tf-idf}}
        iid2fid2tfidf_dict = {}

        itemid2fid2tfidf = readJson(item_feature_file)
        item_num = len(itemid2fid2tfidf)

        if item_num != len(item2iid_dict):
            print("item num error", item_num, len(item2iid_dict))

        for i in range(item_num):
            data_i = item2iid_dict[i]

            itemid_i = list(data_i.keys())[0]
            featureid_tfidf_dict_i = data_i[itemid_i]

            iid_i = item2iid_dict[itemid_i]
            if iid_i not in iid2fid2tfidf_dict:
                iid2fid2tfidf_dict[iid_i] = {}

            for feautreid_ij in featureid_tfidf_dict_i:
                if feautreid_ij not in feature2fid_dict:
                    print("error missing feature", itemid_i, feautreid_ij)
                
                fid_ij = feature2fid_dict[feautreid_ij]
                tfidf_ij = featureid_tfidf_dict_i[feautreid_ij]
                
                iid2fid2tfidf_dict[iid_i][fid_ij] = tfidf_ij

    
        sent2sid_dict["PAD"] = len(sent2sid_dict)
        feature2fid_dict["PAD"] = len(feature2fid_dict)

        self.m_pad_sid = sent2sid_dict["PAD"]
        self.m_pad_fid = feature2fid_dict["PAD"]

        vocab_obj = Vocab()
        vocab_obj.f_set_vocab(user2uid_dict, item2iid_dict, feature2fid_dict, sent2sid_dict)

        self.m_uid2fid2tfidf_dict = uid2fid2tfidf_dict
        self.m_iid2fid2tfidf_dict = iid2fid2tfidf_dict
        self.m_sid2fid2tfidf_dict = sid2fid2tfidf_dict

        self.m_uid_list = uid_list
        self.m_iid_list = iid_list
        self.m_sid_list_list = sid_list_list

        print("... load train data ...", len(self.m_uid2fid2tfidf_dict), len(self.m_iid2fid2tfidf_dict), len(self.m_sid2fid2tfidf_dict))

    def __len__(self):
        return len(self.m_uid_list)

    def add_feature_node(self, G, uid, iid, sid_list):
        fid2nid = {}
        nid2fid = {}

        nid = 0
        fid2tfidf_dict_user = self.m_uid2fid2tfidf_dict[uid]
        for fid in fid2tfidf_dict_user:
            if fid not in fid2nid.keys():
                fid2nid[fid] = nid
                nid2fid[nid] = fid

                nid += 1

        fid2tfidf_dict_item = self.m_iid2fid2tfidf_dict[iid]
        for fid in fid2tfidf_dict_item:
            if fid not in fid2nid.keys():
                fid2nid[fid] = nid
                nid2fid[nid] = fid

                nid += 1
        
        for sid in sid_list:
            fid2tfidf_dict_sent = self.m_sid2fid2tfidf_dict[sid]

            for fid in fid2tfidf_dict_sent:
                if fid not in fid2nid.keys():
                    fid2nid[fid] = nid
                    nid2fid[nid] = fid

                    nid += 1

        fid_node_num = len(nid2fid)

        G.add_nodes(fid_node_num)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(fid_node_num)
        G.ndata["id"] = torch.LongTensor(list(nid2fid.values()))
        G.ndata["dytpe"] = torch.zeros(fid_node_num)

        return fid2nid, nid2fid

    def create_graph(self, uid, iid, sid_list):
        G = dgl.DGLGraph()
        
        ### add feature nodes
        fid2nid, nid2fid = self.add_feature_node(G, uid, iid, sid_list)
        feature_node_num = len(fid2nid)
        
        ### add sent nodes
        sent_node_num = len(sid_list)
        G.add_nodes(sent_node_num)
        G.ndata["unit"][feature_node_num:] = torch.ones(sent_node_num)
        G.ndata["dytpe"][feature_node_num:] = torch.ones(sent_node_num)
        sid2nid = [i+feature_node_num for i in range(sent_node_num)]

        feat_sent_node_num = feature_node_num+sent_node_num

        ### add user, item nodes
        ### add user node
        user_node_num = 1
        G.add_nodes(user_node_num)
        G.ndata["unit"][feat_sent_node_num:] = torch.ones(user_node_num)
        G.ndata["dytpe"][feat_sent_node_num:] = torch.ones(user_node_num)*2
        uid2nid = [i+feat_sent_node_num for i in range(user_node_num)]

        # for i in range()
        


        G.set_e_initializer()


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i = idx

        uid_i = self.m_uid_list[i]
        iid_i = self.m_iid_list[i]
        sid_list_i = self.m_sid_list_list[i]
        
        G = self.create_graph(uid_i, iid_i, sid_list_i)

        return G

    

