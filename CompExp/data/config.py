import os

DIR_PATH = os.path.dirname(__file__)

DATASET = 'rb'

OUTPUT_PATH = os.path.join(DIR_PATH, f'{DATASET}_output')

src_file_name = dict(
  rb='ratebeer/Ratebeer.txt',
  ta='tripadvisor/'
)[DATASET]

UI_FILTER_CONFIG = dict(
  rb=dict(
    min_item_count=20,
    min_user_count=20
  ),
  ta=dict(
    min_item_count=15,
    min_user_count=20
  )
)[DATASET]

SRC_FILE = os.path.join(DIR_PATH, src_file_name)

DATA_FILE = os.path.join(OUTPUT_PATH, 'data.txt')
ATTR_FILE = os.path.join(OUTPUT_PATH, 'attributes.json')

ITEM_FILE = os.path.join(OUTPUT_PATH, 'items.txt')
USER_FILE = os.path.join(OUTPUT_PATH, 'users.txt')

ITEM_CATS_FILE = os.path.join(OUTPUT_PATH, 'item_cats.json')

VOC_FILE = os.path.join(OUTPUT_PATH, 'vocabulary.json')
VOC_WE_FILE = os.path.join(OUTPUT_PATH, 'voc_we.txt')

SPLIT_PATH = os.path.join(OUTPUT_PATH, 'split')
TRAIN_FILE = os.path.join(SPLIT_PATH, 'train.txt')
DEV_FILE = os.path.join(SPLIT_PATH, 'dev.txt')
TEST_FILE = os.path.join(SPLIT_PATH, 'test.txt')

NDCG_FILE = os.path.join(OUTPUT_PATH, 'ndcg.json')

ASPECT_FILE = os.path.join(OUTPUT_PATH, 'aspects.json')
ITEM_ATTR_FILE = os.path.join(OUTPUT_PATH, 'item_attr.json')

NGRAM_IDF_FILE = os.path.join(OUTPUT_PATH, 'ngram_idf.json')
