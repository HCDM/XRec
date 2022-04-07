'''
Comparative Explanation
'''

import os
from functools import lru_cache
import torch

import config
from src.voc import voc
from src.utils import CheckpointManager
from src.models import CompExp
from src.dataset import ReviewDataset, RewriteDataset, TAReviewDataset

DIR_PATH = os.path.dirname(__file__)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


@lru_cache(maxsize=None)
def load_dataset(datatype='train'):
    path = dict(
        train=config.TRAIN_CORPUS,
        dev=config.TEST_CORPUS,
        test=config.TEST_CORPUS
    )[datatype]

    train_set = load_dataset('train') if datatype != 'train' else None

    file = os.path.join(DIR_PATH, path)

    print(f'Reading {datatype} data from {file}...')

    DS = TAReviewDataset if config.DS == 'ta' else ReviewDataset
    dataset = DS.load(file, max_length=config.MAX_LENGTH, train_set=train_set)

    print(f'Read {len(dataset)} reviews')

    return dataset


def load_rewrite_dataset():
    path = config.GEN_PAIR_CORPUS

    file = os.path.join(DIR_PATH, path)

    print(f'Reading data from {file}...')

    dataset = RewriteDataset.load(file, max_length=config.MAX_LENGTH)

    print(f'Read {len(dataset)} reviews')

    return dataset


def init_word_embedding(embedding_path):
    print('Init word embedding from: ', embedding_path)

    embedding_path = os.path.join(DIR_PATH, embedding_path)
    with open(embedding_path, encoding='utf-8') as file:
        lines = file.read().strip().split('\n')

    tokens_of_lines = [l.split(' ') for l in lines]
    words = [l[0] for l in tokens_of_lines]
    weight = [[float(str_emb) for str_emb in l[1:]] for l in tokens_of_lines]

    for i, word in enumerate(words):
        assert voc[i] == word

    # also init the embedding for special tokens
    while len(weight) < voc.size():
        embedding_len = len(weight[0])
        weight.append([0] * embedding_len)

    weight = torch.FloatTensor(weight)

    return weight


@lru_cache(maxsize=None)
def get_n_ui():
    ''' get the number of users & items '''
    with open(config.USER_FILE) as usf, open(config.ITEM_FILE) as imf:
        u = len([i for i in usf.read().split('\n') if i])
        i = len([i for i in imf.read().split('\n') if i])

        if config.DS == 'ta':
            i *= 5  # time n_category

        return u, i


def build_model(model_config, checkpoint):
    N_USERS, N_ITEMS = get_n_ui()

    if not checkpoint:
        # Initialize word embeddings
        pre_we_weight = init_word_embedding(config.WORD_EMBEDDING_FILE)

    if model_config.MODEL_TYPE == 'CompExp':
        model = CompExp(
            len(voc),
            model_config.D_WORD_EBD,
            model_config.D_EXP_EBD,

            model_config.RNN_TYPE,
            model_config.DROPOUT,
            norm_vct=model_config.NORM_VCT,
            keppa=model_config.KEPPA
        )

    else:
        raise Exception('Invalid model type')

    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif model_config.MODEL_TYPE in ['CompExpGen']:  # 'CompExp',
        print('Use pretrained word embedding')
        model.load_pretrained_word_ebd(pre_we_weight)

    # print(model.ctx_out.weight, model.ctx_out.bias)
    # exit()

    # Use appropriate device
    model = model.to(device)

    return model


def init(mdl_name=None, ckpt_name=None):
    if not mdl_name:
        mdl_name = config.DEFAULT_MODEL_NAME

    SAVE_PATH = os.path.join(DIR_PATH, config.SAVE_DIR, mdl_name)
    print('Saving path:', SAVE_PATH)

    ckpt_mng = CheckpointManager(SAVE_PATH)

    checkpoint, continue_training = None, False
    if ckpt_name:
        print('Load checkpoint:', ckpt_name)
        ckpt_tokons = ckpt_name.split('/')
        if len(ckpt_tokons) == 1:
            checkpoint = ckpt_mng.load(ckpt_tokons[0], device)
            continue_training = True

        elif len(ckpt_tokons) == 2:
            load_path = os.path.join(DIR_PATH, config.SAVE_DIR, ckpt_tokons[0])
            load_ckpt_mng = CheckpointManager(load_path)
            checkpoint = load_ckpt_mng.load(ckpt_tokons[1], device)
            continue_training = False

        else:
            raise Exception('Invalid checkpoint path:', ckpt_name)

    model_config = config.load(mdl_name)
    model = build_model(model_config, checkpoint)

    return model, {
        'checkpoint': checkpoint if continue_training else None,
        'ckpt_mng': ckpt_mng,
        'model_config': model_config
    }
