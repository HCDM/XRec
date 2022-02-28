''' RNN Type Adaptor '''

# import torch
from torch import nn

class RNN(nn.Module):
  def __init__(self, rnn_type, *args, **kargs):
    super().__init__()

    if rnn_type == 'LSTM':
        self.model = nn.LSTM(*args, **kargs)
    elif rnn_type == 'GRU':
        self.model = nn.GRU(*args, **kargs)
    else:
      raise Exception('Invalid RNN type: ' + str(rnn_type))

  def forward(self, *args, **kargs):
    return self.model(*args, **kargs)


class RNNStateAdaptor(nn.Module):
  def __init__(self, rnn_type, input_size, hidden_size, n_layers=1):
    super().__init__()

    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.to_hidden = nn.Sequential(
      nn.Linear(input_size, hidden_size * n_layers),
      nn.Tanh()
    )

    self.rnn_type = rnn_type
    if rnn_type == 'LSTM':
      self.to_cell = nn.Sequential(
        nn.Linear(input_size, hidden_size * n_layers),
        nn.Tanh()
      )

  def forward(self, input_var):
    batch_size = input_var.size(0)

    # (batch, state_size) -> (batch, layer * hidden) -> (layer, batch, hiddem)
    init_hidden = self.to_hidden(input_var).view(batch_size, self.n_layers, self.hidden_size).transpose(0, 1).contiguous()

    if self.rnn_type == 'LSTM':
      init_cells = self.to_cell(input_var).view(batch_size, self.n_layers, self.hidden_size).transpose(0, 1).contiguous()

      # LSTM needs hidden & cell state
      init_hidden = (init_hidden, init_cells)

    return init_hidden
