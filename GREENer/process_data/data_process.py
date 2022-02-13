import os
import io
import json
# from dgl.convert import graph
import torch
import numpy as np
import random
import pandas as pd
import argparse
import pickle

from torch.utils.data import dataset
from torch.utils.data import DataLoader

from ratebeer_process import RATEBEER, RATEBEER_TRAIN, RATEBEER_VALID, RATEBEER_TEST


class DATA():
    def __init__(self):
        print("data")
        self.only_item_attr_graph = True

    def f_load_ratebeer(self, args):
        self.m_data_name = args.data_name

        """save train data"""

        sent_content_file = args.data_dir+"train/sentence/id2sentence.json"
        sent_embed_file = args.data_dir+"train/sentence/sid2sentembed.json"
        sent_feature_file = args.data_dir+"train/sentence/sentence2feature.json"

        useritem_candidate_label_sen_file = args.data_dir+"train/useritem2sentids.json"
        # useritem_candidate_soft_label_sen_file = args.data_dir+"train/useritem2sentids_withbpr.json"

        user_feature_file = args.data_dir+"train/user/user2feature.json"
        item_feature_file = args.data_dir+"train/item/item2feature.json"
        feature_embed_file = args.data_dir+"train/feature/featureid2embedding.json"

        graph_dir = args.graph_dir
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        graph_train_dir = graph_dir+"train/"
        if not os.path.exists(graph_train_dir):
            os.makedirs(graph_train_dir)

        train_data_obj = RATEBEER_TRAIN()
        vocab_obj = train_data_obj.load_raw_data(
            sent_content_file, sent_embed_file, feature_embed_file,
            useritem_candidate_label_sen_file,
            user_feature_file, item_feature_file, sent_feature_file,
            graph_train_dir, True
        )

        train_data_file = graph_dir+"train/train_data.pickle"
        train_data = {"train_data": train_data_obj}
        with open(train_data_file, "wb") as f:
            pickle.dump(train_data, f)

        """save valid data"""

        sent_content_file = args.data_dir+"valid/sentence/id2sentence.json"
        useritem_candidate_label_sen_file = args.data_dir+"valid/useritem2sentids_withproxy.json"

        graph_test_dir = graph_dir+"valid/"
        if not os.path.exists(graph_test_dir):
            os.makedirs(graph_test_dir)

        valid_data_obj = RATEBEER_VALID()
        valid_data_obj.load_raw_data(
            vocab_obj,
            train_data_obj.m_uid2fid2tfidf_dict,
            train_data_obj.m_iid2fid2tfidf_dict,
            train_data_obj.m_sid2fid2tfidf_dict,
            sent_content_file,
            useritem_candidate_label_sen_file,
            graph_test_dir, True
        )

        vocab_file = graph_dir+"vocab.pickle"
        vocab = {"vocab": vocab_obj}
        with open(vocab_file, "wb") as f:
            pickle.dump(vocab, f)

        """save test data"""

        sent_content_file = args.data_dir+"test/sentence/id2sentence.json"
        useritem_candidate_label_sen_file = args.data_dir+"test/useritem2sentids_withproxy.json"

        print("sentence file", sent_content_file)

        graph_test_dir = graph_dir+"test/"
        if not os.path.exists(graph_test_dir):
            os.makedirs(graph_test_dir)

        test_data_obj = RATEBEER_TEST()
        test_data_obj.load_raw_data(
            vocab_obj,
            train_data_obj.m_uid2fid2tfidf_dict,
            train_data_obj.m_iid2fid2tfidf_dict,
            train_data_obj.m_sid2fid2tfidf_dict,
            sent_content_file,
            useritem_candidate_label_sen_file,
            graph_test_dir, True
        )

    def f_load_soft_ratebeer(self, args):
        self.m_data_name = args.data_name

        """save train data"""
        print("---"*8 + " TRAIN DATA " + "---"*8)
        sent_content_file = args.data_dir+"train/sentence/id2sentence.json"
        sent_embed_file = args.data_dir+"train/sentence/sid2sentembed.json"
        sent_feature_file = args.data_dir+"train/sentence/sentence2feature.json"
        # useritem_candidate_label_sen_file = args.data_dir+"train/useritem2sentids.json"
        if self.only_item_attr_graph:
            useritem_candidate_soft_label_sen_file = args.data_dir+"train/useritem2sentids_item_attr_withbpr.json"
        else:
            useritem_candidate_soft_label_sen_file = args.data_dir+"train/useritem2sentids_withbpr.json"
        user_feature_file = args.data_dir+"train/user/user2feature.json"
        item_feature_file = args.data_dir+"train/item/item2feature.json"
        feature_embed_file = args.data_dir+"train/feature/featureid2embedding.json"

        graph_dir = args.graph_dir
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        graph_train_dir = graph_dir+"train_soft/"
        if not os.path.exists(graph_train_dir):
            os.makedirs(graph_train_dir)
        print("save graph to: {}".format(graph_train_dir))

        train_data_obj = RATEBEER_TRAIN()
        vocab_obj = train_data_obj.load_raw_data(
            sent_content_file, sent_embed_file, feature_embed_file,
            useritem_candidate_soft_label_sen_file,
            user_feature_file, item_feature_file, sent_feature_file,
            graph_train_dir, True
        )

        train_data_file = graph_dir+"train_soft/train_data.pickle"
        train_data = {"train_data": train_data_obj}
        with open(train_data_file, "wb") as f:
            pickle.dump(train_data, f)

        """save valid data"""
        print("---"*8 + " VALID DATA " + "---"*8)
        sent_content_file = args.data_dir+"valid/sentence/id2sentence.json"
        if self.only_item_attr_graph:
            useritem_candidate_label_sen_file = args.data_dir+"valid/useritem2sentids_item_attr_withproxy.json"
        else:
            useritem_candidate_label_sen_file = args.data_dir+"valid/useritem2sentids_withproxy.json"
        print("sentence file: {}".format(sent_content_file))
        print("proxy file: {}".format(useritem_candidate_label_sen_file))
        graph_test_dir = graph_dir+"valid/"
        if not os.path.exists(graph_test_dir):
            os.makedirs(graph_test_dir)
        print("save graph to: {}".format(graph_test_dir))

        valid_data_obj = RATEBEER_VALID()
        valid_data_obj.load_raw_data(
            vocab_obj,
            train_data_obj.m_uid2fid2tfidf_dict,
            train_data_obj.m_iid2fid2tfidf_dict,
            train_data_obj.m_sid2fid2tfidf_dict,
            sent_content_file,
            item_feature_file,
            useritem_candidate_label_sen_file,
            graph_test_dir, True
        )

        vocab_file = graph_dir+"vocab.pickle"
        vocab = {"vocab": vocab_obj}
        with open(vocab_file, "wb") as f:
            pickle.dump(vocab, f)
        print("vocab file save at: {}".format(vocab_file))

        """save test data"""
        print("---"*8 + " TEST DATA " + "---"*8)
        sent_content_file = args.data_dir+"test/sentence/id2sentence.json"
        if self.only_item_attr_graph:
            useritem_candidate_label_sen_file = args.data_dir+"test/useritem2sentids_item_attr_withproxy.json"
        else:
            useritem_candidate_label_sen_file = args.data_dir+"test/useritem2sentids_withproxy.json"
        print("sentence file: {}".format(sent_content_file))
        print("proxy file: {}".format(useritem_candidate_label_sen_file))
        graph_test_dir = graph_dir+"test/"
        if not os.path.exists(graph_test_dir):
            os.makedirs(graph_test_dir)
        print("save graph to: {}".format(graph_test_dir))

        test_data_obj = RATEBEER_TEST()
        test_data_obj.load_raw_data(
            vocab_obj,
            train_data_obj.m_uid2fid2tfidf_dict,
            train_data_obj.m_iid2fid2tfidf_dict,
            train_data_obj.m_sid2fid2tfidf_dict,
            sent_content_file,
            item_feature_file,
            useritem_candidate_label_sen_file,
            graph_test_dir, True
        )

    def f_load_only_item_ratebeer(self, args):
        """ Only load and save valid / test data.
        For valid and test graph, we only have item-side attributes and item-side sentences.
        """
        pass

    def f_load_only_item_feat_ratebeer(self, args):
        """ Only load and save valid / test data.
        For valid and test graph, we only have item-side attributes and item-side sentences or
        user-side sentences which only contain item-side attributes.
        """
        pass

    def f_load_ratebeer_resume(self, args):
        self.m_data_name = args.data_name

        """
        save train data 
        """

        graph_dir = args.graph_dir

        train_data_file = graph_dir+"train/train_data.pickle"
        # train_data = {"train_data": train_data_obj}
        train_data = None
        with open(train_data_file, "rb") as f:
            train_data = pickle.load(f)
        train_data_obj = train_data["train_data"]

        """
        save valid data
        """

        vocab_file = graph_dir+"vocab.pickle"
        vocab = None
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
        vocab_obj = vocab["vocab"]

        # sent_content_file = args.data_dir+"valid/sentence/id2sentence.json"
        # useritem_candidate_label_sen_file = args.data_dir+"valid/useritem2sentids_withproxy.json"
        
        # graph_test_dir = graph_dir+"valid/"
        # if not os.path.exists(graph_test_dir):
        #     os.makedirs(graph_test_dir)

        # valid_data_obj = RATEBEER_VALID()
        # valid_data_obj.load_raw_data(vocab_obj, train_data_obj.m_uid2fid2tfidf_dict, train_data_obj.m_iid2fid2tfidf_dict, train_data_obj.m_sid2fid2tfidf_dict, sent_content_file, useritem_candidate_label_sen_file, graph_test_dir, True)

        # vocab_file = graph_dir+"vocab.pickle"
        # vocab = {"vocab": vocab_obj}
        # with open(vocab_file, "wb") as f:
        #     pickle.dump(vocab, f)
        
        """
        save test data
        """

        sent_content_file = args.data_dir+"test/sentence/id2sentence.json"
        useritem_candidate_label_sen_file = args.data_dir+"test/useritem2sentids_withproxy.json"
        
        print("sentence file", sent_content_file)

        graph_test_dir = graph_dir+"test/"
        if not os.path.exists(graph_test_dir):
            os.makedirs(graph_test_dir)

        test_data_obj = RATEBEER_TEST()
        test_data_obj.load_raw_data(vocab_obj, train_data_obj.m_uid2fid2tfidf_dict, train_data_obj.m_iid2fid2tfidf_dict, train_data_obj.m_sid2fid2tfidf_dict, sent_content_file, useritem_candidate_label_sen_file, graph_test_dir, True)
