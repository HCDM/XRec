import torch
from torch import nn

from .rnn import RNN
from .attn import InnerAttn


class RNNEncoder(nn.Module):
  '''
  Inputs:
    input_var: (word_seq, batch, hidden)
    word_mask: (word_seq, seq)
    rvw_lens: (batch) number of words

  Outputs:
    rvw_vcts: (batch, 2 * hidden)
    word_attn: (sen_batch, word_seq)
  '''

  def __init__(self, hidden_size, word_ebd_size, rnn_type='GRU', n_layers=1):
    super().__init__()

    self.word_rnn = RNN(rnn_type, word_ebd_size, hidden_size, num_layers=n_layers, bidirectional=True)

    self.word_inner_attn = InnerAttn(2 * hidden_size)
    self.hidden_size = hidden_size
    # self.out = nn.Linear(2 * hidden_size, hidden_size),

  def forward(self, input_var, word_mask=None, rvw_lens=None):
    # convert rvw_lens to mask if given & no mask
    if rvw_lens is not None and word_mask is None:
      word_mask = torch.zeros((input_var.size(0), input_var.size(1)), dtype=torch.bool, device=input_var.device)

      for i, l in enumerate(rvw_lens):
        word_mask[:l, i] = 1

    # convert mask to rvw_lens if given & no rvw_lens
    elif rvw_lens is None and word_mask is not None:
      rvw_lens = word_mask.long().sum(0)

    word_input = nn.utils.rnn.pack_padded_sequence(input_var, rvw_lens, enforce_sorted=False)

    word_output, _ = self.word_rnn(word_input)

    word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output)

    # (seq, batch, hidden * 2) -> (batch, seq, hidden * 2)
    word_output = word_output.transpose(0, 1)
    word_mask = word_mask.transpose(0, 1)

    # (batch, seq, hidden * 2) -> (batch, hidden * 2)
    rvw_vcts, word_attn = self.word_inner_attn(word_output, word_mask)

    return rvw_vcts[:, :self.hidden_size] + rvw_vcts[:, self.hidden_size:], word_attn
