# Dataset
DS = 'rb'

OUTPUT_PATH = f'data/{DS}_output'

# Corpus
TRAIN_CORPUS = f'{OUTPUT_PATH}/split/train.txt'
DEV_CORPUS = f'{OUTPUT_PATH}/split/dev.txt'
TEST_CORPUS = f'{OUTPUT_PATH}/split/test.txt'

GEN_PAIR_CORPUS = f'{OUTPUT_PATH}/gen_pairs.txt'

# User & Item files
USER_FILE = f'{OUTPUT_PATH}/users.txt'
ITEM_FILE = f'{OUTPUT_PATH}/items.txt'

# Vocabulary & Word Embedding
VOC_FILE = f'{OUTPUT_PATH}/vocabulary.json'
WORD_EMBEDDING_FILE = f'{OUTPUT_PATH}/voc_we.txt'

# Attribute list
ATTR_FILE = f'{OUTPUT_PATH}/attributes.json'
ITEM_ATTR_FILE = f'{OUTPUT_PATH}/item_attr.json'
ITEM_CATS_FILE = f'{OUTPUT_PATH}/item_cats.json'

# Sentimental list
SEN_FILE = f'{OUTPUT_PATH}/rating_words.json'

NGRAM_IDF_FILE = f'{OUTPUT_PATH}/ngram_idf.json'

# Checkpoints
SAVE_DIR = f'{DS}_checkpoints'

# Trainer behaviors
PATIENCE = 5
PRINT_EVERY = 1000
SAVE_EVERY = 5

MAX_LENGTH = 25  # Maximum sentence length to consider
