from datetime import datetime
import numpy as np
import torch
import random
import torch.nn as nn

import pickle
import argparse
from data_process import DATA
import json
import os

import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    seed = 1234
    set_seed(seed)

    data_obj = DATA()

    s_time = datetime.now()
    if "beer" in args.data_name:
        # data_obj.f_load_ratebeer(args)
        data_obj.f_load_soft_ratebeer(args)
        # data_obj.f_load_ratebeer_resume(args)
    elif "trip" in args.data_name:
        data_obj.f_load_soft_ratebeer(args)

    e_time = datetime.now()
    print("... save data duration ... ", e_time-s_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### data
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='ratebeer')
    parser.add_argument('--data_file', type=str, default='data.pickle')
    parser.add_argument('--graph_dir', type=str, default='../output_graph/')

    args = parser.parse_args()

    main(args)
