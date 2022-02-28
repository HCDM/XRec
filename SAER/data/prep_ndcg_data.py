'''
Prepare related data for testing NDCG scores
Randomly choose a set of predefined related items for each user
'''

import os
import random
from collections import defaultdict
import json

import config

DIR_PATH = os.path.dirname(__file__)

DATA_FILE = config.DATA_FILE
TEST_FILE = config.TEST_FILE
NDCG_FILE = config.NDCG_FILE

REL_SIZE = 15

def main():

  user_rel_items = defaultdict(dict)
  with open(TEST_FILE) as data_file:
    lines = data_file.read().split('\n')

    for line in lines:
      tokens = json.loads(line)
      iid = int(tokens[0])
      uid = int(tokens[1])
      score = float(tokens[2])
      user_rel_items[uid][iid] = score

  ndcg_data = {}
  for uid, rel_items in user_rel_items.items():
    if len(rel_items) < REL_SIZE:
      continue

    rel_list = list(rel_items.keys())
    rel_list = random.sample(rel_list, REL_SIZE)

    ndcg_data[uid] = rel_list

  print('Num of users:', len(ndcg_data))
  with open(NDCG_FILE, 'w') as ndcg_file:
    json.dump(ndcg_data, ndcg_file)


if __name__ == '__main__':
  main()
