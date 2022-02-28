import os

DIR_PATH = os.path.dirname(__file__)

DATASET = 'yelp'
# DATASET = 'ratebeer'

OUTPUT_PATH = os.path.join(DIR_PATH, f'{DATASET}_output')

src_file_name = dict(
  yelp='review.json',
  ratebeer='Ratebeer.txt'
)[DATASET]

UI_FILTER_CONFIG = dict(
  yelp=dict(
    min_item_count=15,
    min_user_count=30,
    random_drop_item=0
  ),
  ratebeer=dict(
    min_item_count=30,
    min_user_count=40,
    random_drop_item=0.5 # ratebeer is too dense, drop some items to reduce the corpus size
  )
)[DATASET]

SRC_FILE = os.path.join(DIR_PATH, DATASET, src_file_name)

DATA_FILE = os.path.join(OUTPUT_PATH, 'data.txt')
FEATURE_FILE = os.path.join(OUTPUT_PATH, 'features.json')

ITEM_FILE = os.path.join(OUTPUT_PATH, 'items.txt')
USER_FILE = os.path.join(OUTPUT_PATH, 'users.txt')

ITEM_FEATURE_FILE = os.path.join(OUTPUT_PATH, 'item_features.json')

VOC_FILE = os.path.join(OUTPUT_PATH, 'vocabulary.json')
VOC_WE_FILE = os.path.join(OUTPUT_PATH, 'voc_we.txt')

SPLIT_PATH = os.path.join(OUTPUT_PATH, 'split')
TRAIN_FILE = os.path.join(SPLIT_PATH, 'train.txt')
DEV_FILE = os.path.join(SPLIT_PATH, 'dev.txt')
TEST_FILE = os.path.join(SPLIT_PATH, 'test.txt')

NDCG_FILE = os.path.join(OUTPUT_PATH, 'ndcg.json')
