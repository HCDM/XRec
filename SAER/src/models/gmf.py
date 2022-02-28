from torch.nn import nn


class GMF(nn.Module):
  '''
  Generalized Matrix Factorization
  '''

  def __init__(
    self,
    n_users,
    n_items,
    d_ebd,
    weighted=True
  ):
    super(GMF, self).__init__()

    self.user_ebd = nn.Embedding(n_users, d_ebd)
    self.item_ebd = nn.Embedding(n_items, d_ebd)

    # downscale the embedding
    self.user_ebd.weight = nn.Parameter(self.user_ebd.weight * 1e-2)
    self.item_ebd.weight = nn.Parameter(self.item_ebd.weight * 1e-2)

    self.weighted = weighted
    if weighted:
      self.weight = nn.Linear(d_ebd, 1)

  def rate(self, data):
    user_vct = self.user_ebd(data.users)
    item_vct = self.item_ebd(data.items)

    output_var = user_vct * item_vct

    if self.weighted:
      output_var = self.weight(output_var).squeeze(-1)
    else:
      output_var = output_var.sum(1)

    output_var = output_var
    return output_var

  def forward(self, data):
    self.rate(data)
