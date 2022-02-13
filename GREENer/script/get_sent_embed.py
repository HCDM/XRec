from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import csv
import pickle
import time
import json
import argparse

from transformers import (AutoConfig, AutoTokenizer, AutoModel)
import torch
import torch.nn.functional as F

print("cuda", torch.cuda.is_available())

def main(args):
    def readJson(fname):
        data = []
        print("file name", fname)
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

    def get_sentence_embed(sent):
        encoded_input = tokenizer(sent, padding=True, truncation=True, max_length=100, return_tensors='pt')
        encoded_input = encoded_input.to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)

        corpus_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        corpus_embeddings = corpus_embeddings.cpu()

        return corpus_embeddings

    ### load the pretrained model
    output_dir = "../../../checkpoint/test-mlm-wwm"

    config_file = os.path.join(output_dir, "tokenizer_config.json")
    model_path = os.path.join(output_dir)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    ### load the data

    ### load sentences
    # dataset_path = "/p/reviewde/data/ratebeer/graph/medium_30/train/sentence"
    dataset_path = args.data_dir
    id2sent_file = "id2sentence.json"

    id2sent_abs_file = os.path.join(dataset_path, id2sent_file)
    # id2sent_abs_f = open(id2sent_abs_file, "rb")

    # id2sent_dict = pickle.load(id2sent_abs_f)
    id2sent_dict = readJson(id2sent_abs_file)[0]
    sent_num = len(id2sent_dict)
    print("sentences num", sent_num)

    print("xxx"*3, " Start ", "xxx"*3)
    start_time = time.time()


    id2sent_embed_dict = {}

    sentid_list = id2sent_dict.keys()
    sentid_list = list(sentid_list)

    # sent_num = 200

    batch_size = 128
    batch_num = sent_num//batch_size

    for batch_idx in range(batch_num):

        batch_sentid = []
        batch_sent = []
        for idx in range(batch_size):
            sent_idx = batch_size*batch_idx+idx
            sentid_i = sentid_list[sent_idx]
            batch_sentid.append(sentid_i)

            sent_i = id2sent_dict[sentid_i]
            batch_sent.append(sent_i)

        candidate_embed_batch = get_sentence_embed(batch_sent)
        for idx in range(batch_size):
            sentid_i = batch_sentid[idx]
            sentembed_i = candidate_embed_batch[idx].cpu().numpy()
            # print(sentembed_i.shape)
            id2sent_embed_dict[sentid_i] = sentembed_i

    # exit()
    batch_sent_num = batch_num*batch_size
    left_sent_num = sent_num-batch_sent_num
    batch_sentid = []
    batch_sent = []
    for sent_idx in range(batch_sent_num, sent_num):
        sentid_i = sentid_list[sent_idx]
        batch_sentid.append(sentid_i)

        sent_i = id2sent_dict[sentid_i]
        batch_sent.append(sent_i)

    candidate_embed_batch = get_sentence_embed(batch_sent)
    for sent_idx in range(left_sent_num):
        sentid_i = batch_sentid[sent_idx]
        sentembed_i = candidate_embed_batch[sent_idx].cpu().numpy()

        id2sent_embed_dict[sentid_i] = sentembed_i

    end_time = time.time()

    duration = end_time-start_time
    print("duration", duration)

    outputput_pair_file = "sid2sentembed.json"
    output_pair_abs_file = os.path.join(dataset_path, outputput_pair_file)
    print("output pair file", output_pair_abs_file)

    print("len id2sent_embed_dict", len(id2sent_embed_dict))

    with open(output_pair_abs_file, "w") as f:
        for idx in range(sent_num):
            sentid_i = sentid_list[idx]
            sentembed_i = id2sent_embed_dict[sentid_i]
            
            sentembed_i = sentembed_i.tolist()
            
            # print(sentembed_i)

            line = {sentid_i:sentembed_i}

            json.dump(line, f)
            f.write("\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### data
    parser.add_argument('--data_dir', type=str, default='data')

    args = parser.parse_args()

    main(args)