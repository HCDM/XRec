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


# from torch.utils.data import dataset
# from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader

# from movie import MOVIE, MOVIE_TEST
# from movie_sample import MOVIE, MOVIE_TEST
# from movie_iter import MOVIE_LOADER, MOVIE_TEST

from ratebeer import RATEBEER
from ratebeer_process import Vocab


class DATA():
    def __init__(self):
        print("data")

    def f_load_graph_ratebeer(self, args):
        self.m_data_name = args.data_name

        graph_dir = args.graph_dir
        graph_train_dir = graph_dir+"train_soft/"

        train_data = RATEBEER()
        train_data.load_train_graph_data(graph_train_dir)

        if args.train:
            graph_test_dir = graph_dir+"valid/"
            valid_data = RATEBEER()
            valid_data.load_eval_graph_data(graph_test_dir)
        else:
            graph_test_dir = graph_dir+"test/"
            valid_data = RATEBEER()
            valid_data.load_eval_graph_data(graph_test_dir)

        vocab_file = graph_dir+"vocab.pickle"
        vocab = None
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)

        vocab_obj = vocab["vocab"]
        print("user num", len(vocab_obj.m_user2uid))
        print("item num", len(vocab_obj.m_item2iid))

        batch_size = args.batch_size

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)

        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4)

        # print("train_loader", train_loader)

        return train_loader, valid_loader, vocab_obj
