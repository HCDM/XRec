import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from .rnn import RNN
from .attn import InnerAttn


class TextRNNEncoder(nn.Module):
    '''
    Inputs:
      input_var: (batch, seq, hidden)
      word_mask: (batch, seq)
      rvw_lens: (batch) number of words

    Outputs:
      rvw_vcts: (batch, 2 * hidden)
      word_attn: (sen_batch, word_seq)
    '''

    def __init__(self, hidden_size, n_words, d_word_ebd, rnn_type='GRU', inner_attn=False, n_layers=1, dropout=0.):
        super().__init__()

        self.word_ebd = nn.Embedding(n_words, d_word_ebd)
        self.dropout = nn.Dropout(dropout)

        self.word_rnn = RNN(rnn_type, d_word_ebd, hidden_size,
                            num_layers=n_layers, bidirectional=True, batch_first=True)

        self.word_inner_attn = None
        if inner_attn:
            self.word_inner_attn = InnerAttn(2 * hidden_size)

        self.hidden_size = hidden_size
        # self.out = nn.Linear(2 * hidden_size, hidden_size),

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def forward(self, word_var, word_mask=None, lens=None):
        # embed
        if len(word_var.size()) == 2:
            embedded = self.word_ebd(word_var)
        else:
            # (batch, seq, n_word) @ (1, n_word, d_ebd) = (batch, seq, d_ebd)
            embedded = torch.matmul(word_var, self.word_ebd.weight.unsqueeze(0))

        embedded = self.dropout(embedded)

        # convert mask to lens if given & no lens
        if lens is None and word_mask is not None:
            lens = word_mask.long().sum(1)

        # encode with RNN
        word_input = nn.utils.rnn.pack_padded_sequence(
            embedded, lens, enforce_sorted=False, batch_first=True)

        word_output, hidden = self.word_rnn(word_input)

        word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)

        if not self.word_inner_attn:
            # (batch, seq, 2 * hidden) -> (batch, seq, hidden)
            word_output = word_output[..., :self.hidden_size] + word_output[..., self.hidden_size:]
            # (dir * n_layers, batch, hidden) -> (batch, hidden)
            hidden = hidden.sum(0)
            return word_output, hidden

        # convert rvw_lens to mask if given & no mask
        if lens is not None and word_mask is None:
            word_mask = torch.zeros((word_output.size(0), word_output.size(
                1)), dtype=torch.bool, device=word_output.device)

            for i, l in enumerate(lens):
                word_mask[i, :l] = 1

        # (batch, seq, hidden * 2) -> (batch, hidden * 2)
        rvw_vcts, word_attn = self.word_inner_attn(word_output, word_mask)

        return rvw_vcts[:, :self.hidden_size] + rvw_vcts[:, self.hidden_size:], word_attn


class PositionalEncoding(nn.Module):
    r'''Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    '''

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r'''Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        '''

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ReviewTransEncoder(nn.Module):
    '''
    Inputs:
      input_var: (word_seq, batch, hidden)
      word_mask: (word_seq, seq)
      rvw_lens: (batch) number of words

    Outputs:
      rvw_vcts: (batch, 2 * hidden)
      word_attn: (sen_batch, word_seq)
    '''

    def __init__(self, hidden_size, word_ebd_size, n_head=12, n_layers=6, max_len=50):
        super().__init__()

        self.pos_encoder = PositionalEncoding(word_ebd_size, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(
            word_ebd_size, n_head, dim_feedforward=hidden_size)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.word_ebd_size = word_ebd_size

    def forward(self, input_var, word_mask=None, rvw_lens=None):
        if rvw_lens is not None and word_mask is None:
            word_mask = torch.zeros((input_var.size(0), input_var.size(
                1)), dtype=torch.bool, device=input_var.device)

            for i, l in enumerate(rvw_lens):
                word_mask[:l, i] = 1

        src = input_var * math.sqrt(self.word_ebd_size)
        src = self.pos_encoder(src)

        # transformer mask use 1 for padding
        word_mask = ~word_mask.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=word_mask)

        return output[0], None
