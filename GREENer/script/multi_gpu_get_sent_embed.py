from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import csv
import pickle
import time
import json
# import ray
import argparse

from transformers import (AutoConfig, AutoTokenizer, AutoModel)
import torch
import torch.nn.functional as F
import datetime
from ratebeer import BEER

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

print("cuda", torch.cuda.is_available())


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_sentence_embed(tokenizer, model, sent, device):
    encoded_input = tokenizer(sent, padding=True, truncation=True, max_length=30, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # corpus_embeddings = corpus_embeddings.cpu()

    return corpus_embeddings

def get_corpus_embed(tokenizer, model, device, data_loader, id2sent_dict):
    print("data_loader", len(data_loader))

    new_data = []
    iter_idx = 0

    for data in data_loader:
        iter_idx += 1
        if iter_idx % 50 == 0:
            print("iter_idx", iter_idx)
        
        for data_i in data:
            new_data_i = {}




def main(args):

    local_rank = args.local_rank

    torch.distributed.init_process_group(backend="nccl")

    output_dir = "../checkpoint/test-mlm-wwm"

    config_file = os.path.join(output_dir, "tokenizer_config.json")
    model_path = os.path.join(output_dir)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained(output_dir)

    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    ### load sentences
    dataset_path = "/p/reviewde/data/ratebeer/sentences"
    id2sent_file = "id2sentence.json"

    max_corpus_size = 100
    embedding_cache_path = 'ratebeer-embeddings-size-{}.pkl'.format(max_corpus_size)

    id2sent_abs_file = os.path.join(dataset_path, id2sent_file)
    id2sent_abs_f = open(id2sent_abs_file, "rb")

    # id2sent_dict = pickle.load(id2sent_abs_f)
    id2sent_dict = readJson(id2sent_abs_f)
    sent_num = len(id2sent_dict)
    print("sentences num", sent_num)

    data_obj = BEER(args)

    batch_size = args.batch_size

    train_sampler = DistributedSampler(dataset=data_obj)
    train_loader = DataLoader(dataset=data_obj, batch_size=batch_size, sampler=train_sampler, num_workers=2, collate_fn=data_obj.collate)

    print("xxx"*3, " Start ", "xxx"*3)
    start_time = time.time()
    print("start_time", datetime.datetime.now())

    output_data = get_corpus_embed(tokenizer, ddp_model, device, train_loader, id2sent_dict)

    print("output_data", len(output_data))

    end_time = time.time()

    duration = end_time-start_time
    print("duration", duration)
    print("end time", datetime.datetime.now())

    output_pair_file = "new_test_example_"+str(local_rank)+".json"
    if args.output_dir != "":
        output_data_path = os.path.join(dataset_path, args.output_dir)
    else:
        output_data_path = dataset_path
    output_pair_abs_file = os.path.join(output_data_path, output_pair_file)
    print("output pair file", output_pair_abs_file)

    with open(output_pair_abs_file, "w") as f:
        # for data_idx in range(data_num):
        # output_data = data_obj.m_input_batch_list
        print("output data num", len(output_data))
        for data_i in output_data:
            # data_i = data[data_idx]
            user_i = data_i["user"]
            item_i = data_i["item"]
            candidate_i = [i.item() for i in data_i["target"]]
            review_i = [i for i in list(data_i["review"])]
            
            # print("candidate_i", candidate_i)
            # print(candidate_i)
            # # print(review_i)
            # exit()
            line = {"user": user_i, "item": item_i, "candidate": candidate_i, "review": review_i}

            json.dump(line, f)
            f.write("\n")
        print("finished writing")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=1)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=80)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=0.01)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=1234)
    parser.add_argument("--data_dir", type=str, help="Directory for data.", default="/p/reviewde/data/ratebeer/sentences")
    parser.add_argument("--output_dir", type=str, help="Directory for data.", default="")
    parser.add_argument("--data_input_file", type=str, help="data filename.", default="test_example_100.json")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    args = parser.parse_args()

    main(args)