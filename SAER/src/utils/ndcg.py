''' utils for ndcg calculation '''

import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def np_dcg(rel_list):
    if type(rel_list) is not np.ndarray:
        rel_list = np.array(rel_list)

    return np.sum(
        (2 ** rel_list - 1) / np.log2(np.arange(2, rel_list.size + 2))
    )


def np_idcg(rel_list):
    rel_list = np.sort(rel_list)[::-1]
    return np_dcg(rel_list)


# torch ndcg util functions
def dcg(rel_tensor, k=None):
    if not torch.is_tensor(rel_tensor):
        rel_tensor = torch.tensor(rel_tensor).to(DEVICE)

    rank_tensor = torch.arange(
        1, rel_tensor.size(-1) + 1, dtype=torch.float).to(DEVICE)

    if k is None:
        k = rel_tensor.size(-1)

    return (
        (2 ** rel_tensor - 1) / (rank_tensor + 1).log2()
    )[..., :k].sum(-1)


def idcg(rel_tensor, k=None):
    rel_tensor, _ = rel_tensor.sort(-1, descending=True)
    return dcg(rel_tensor)


def ndcg(rel_list, k=None, method='torch'):
    '''
    Input:
      rel_list: list of relevance, (*, N) tensor
      method: use numpy or torch to calculate
    Output:
      ndcg: (*) tensor
    '''

    if method == 'numpy':
        return np_dcg(rel_list) / np_idcg(rel_list)

    return dcg(rel_list, k) / idcg(rel_list, k)
