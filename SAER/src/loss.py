import torch
# import torch.nn as nn
from torch.nn.functional import mse_loss, cross_entropy, nll_loss, binary_cross_entropy_with_logits

import config

from .utils import idcg
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def mask_nll_loss(inp, target, mask):
  inp = inp.view(-1, inp.size(-1))
  target = target.view(-1)

  mask = mask.view(-1)

  loss = nll_loss(inp, target, reduction='none').masked_select(mask).mean()

  return loss


def mask_ce_loss(inp, target, mask):
  '''
  Calculate the cross entropy loss and a binary mask tensor describing the padding of the target tensor.

  Inputs:
    inp: tensor of raw values; size=(seq, batch, classes)
    target: tensor of classes; size=(seq, batch)
    mask: binary mask; size=(seq, batch)

  Outputs:
    loss: scalar
  '''

  # format the sizes
  inp = inp.view(-1, inp.size(-1))
  target = target.view(-1)

  mask = mask.view(-1)

  loss = cross_entropy(inp, target, reduction='none').masked_select(mask).mean()

  return loss


def rmse_loss(preds, truth):
  return mse_loss(preds, truth).sqrt()


def bpr_loss(preds, truth, weight=None):
  assert preds.size(-1) == truth.size(-1)

  loss_all = None
  n_samples = preds.size(-1)

  for k in range(1, n_samples):
    diff = preds[..., 0] - preds[..., k]
    target = truth[..., 0] - truth[..., k]

    target[target > 0] = 1
    target[target == 0] = .5
    target[target < 0] = 0

    # only select diff pairs
    diff = diff[target != 0.5]
    target = target[target != 0.5]

    loss = binary_cross_entropy_with_logits(diff, target, weight=weight, reduction='none')

    loss_all = loss if loss_all is None else torch.cat([loss_all, loss])

  return loss_all.mean()


def rank_hinge_loss(preds, truth, boundary=config.HINGE_THRESHOLD):
  assert preds.size(-1) == truth.size(-1)

  # (..., 1) - (..., k)
  diff = preds[..., 0:1] - preds[..., 1:]
  target = truth[..., 0:1] - truth[..., 1:]

  # only select diff pairs to rank
  diff = diff[target != 0]
  target = target[target != 0]

  # hinge loss
  diff[target < 0] *= -1
  loss = boundary - diff
  loss = loss[loss > 0].mean()

  return loss


def lambda_rank_loss(preds, scores):
  '''
  (batch, item_size)
  '''

  batch_size = preds.size(0)
  item_size = preds.size(-1)

  _, indices = preds.sort(descending=True)

  # (batch)
  idcg_var = idcg(scores)

  ranks = torch.zeros(scores.size(), device=DEVICE)
  for i in range(batch_size):
    ranks[i][indices[i]] = torch.arange(1, item_size + 1, dtype=torch.float, device=DEVICE)

  pairs = []
  score_pairs = []
  weight = []

  # only use first item to form pairs
  # for i in range(item_size):
  for i in range(1):
    for j in range(i + 1, item_size):
      delta_ndcg = (2 ** scores[:, i] - 2 ** scores[:, j]) * \
        (
          1 / (ranks[:, i] + 1).log() -
          1 / (ranks[:, j] + 1).log()
        ) / idcg_var

      delta_ndcg = delta_ndcg.abs()
      weight.append(delta_ndcg)

      pairs.append(preds[:, (i, j)])
      score_pairs.append(scores[:, (i, j)])

  pairs = torch.cat(pairs)
  score_pairs = torch.cat(score_pairs)
  weight = torch.cat(weight)

  return bpr_loss(pairs, score_pairs, weight)
