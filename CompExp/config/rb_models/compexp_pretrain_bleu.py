from .base import *

MODEL_TYPE = 'CompExp'

# LSTM or GRU as RNN
RNN_TYPE = 'GRU'

# size of the word & exp embedding
D_WORD_EBD = 300
D_EXP_EBD = 300
MODEL_LAYERS = 1
DROPOUT = 0.1

# [Training]

TRAINING_TASK = 'pretrain'
BATCH_SIZE = 256

LR = 1e-3
L2_PENALTY = 1e-4

N_ITEM_EXPS = 10
N_REF_EXPS = 10
N_POS_EXPS = 0
N_USER_EXPS = 0

NORM_VCT = False
KEPPA = 3

EXT_LOSS = 'BLEU'

LOSS_LAMBDA = dict(
    ext=50,
    gen=1,
    ext_entropy=0
)

USE_IDF = True

MAX_ITERS = 4000
