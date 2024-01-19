from .base import *

MODEL_TYPE = 'CompExp'

# LSTM or GRU as RNN
RNN_TYPE = 'GRU'

# size of the word & exp embedding
D_WORD_EBD = 300
D_EXP_EBD = 300
MODEL_LAYERS = 1
DROPOUT = 0.1

REF_ATTN = False

# [Training]

TRAINING_TASK = 'e2e'
BATCH_SIZE = 128

LR = 1e-5
L2_PENALTY = 1e-4

N_ITEM_EXPS = 5
N_REF_EXPS = 10

NORM_VCT = False
KEPPA = 3

REWRITE = True

MAX_ITERS = 2000

LOSS_LAMBDA = dict(
    pg=5,
    gen=1
)

BLEU_WEIGHT = ((2., .1), (.8, .2))

USE_IDF = True
