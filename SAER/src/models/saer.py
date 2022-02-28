import torch
from torch import nn

from .rater import Rater
from .text_rnn import SAERTextRNN
from .mlp import MLP

# from .attn import Attn
from .review_encoder import RNNEncoder
from ..utils import AttrDict


class SAER(nn.Module):
    '''
    Inputs:
        data: AttrDict
            users
            items
            scores
            words
            mask
        word_var: (seq, batch)

    Outputs:
    '''

    def __init__(
        self,
        n_users,
        n_items,
        d_ebd,
        ui_mlp_sizes,

        r_mlp_sizes,

        t_hidden_size,
        t_n_words,
        t_d_word_ebd,
        t_n_layers,
        t_dropout,
        t_rnn_type,

        match_tensor_type
    ):
        super(SAER, self).__init__()

        self.match_tensor_type = match_tensor_type

        if self.match_tensor_type == 'concat':
            ui_input_size = d_ebd * 2
        elif self.match_tensor_type == 'multiply':
            ui_input_size = d_ebd
        elif self.match_tensor_type == 'multiply_minus':
            ui_input_size = d_ebd * 2
        elif self.match_tensor_type == 'multiply_concat':
            ui_input_size = d_ebd * 3
        elif self.match_tensor_type == 'outter':
            ui_input_size = d_ebd ** 2
        elif self.match_tensor_type == 'outter_concat':
            ui_input_size = d_ebd ** 2 + d_ebd * 2
        else:
            raise Exception('Invalid match tensor type')

        self.ui_mlp = MLP(ui_input_size, ui_mlp_sizes)

        self.user_ebd = nn.Embedding(n_users, d_ebd)
        self.item_ebd = nn.Embedding(n_items, d_ebd)
        self.user_t_ebd = nn.Embedding(n_users, 100)
        self.item_t_ebd = nn.Embedding(n_items, 100)

        t_input_size = 200

        nn.init.uniform_(self.user_ebd.weight, a=-1e-6, b=1e-6)
        nn.init.uniform_(self.item_ebd.weight, a=-1e-6, b=1e-6)

        nn.init.uniform_(self.user_t_ebd.weight, a=-1e-6, b=1e-6)
        nn.init.uniform_(self.item_t_ebd.weight, a=-1e-6, b=1e-6)

        self.rater = Rater(ui_mlp_sizes[-1], r_mlp_sizes, act='leakyrelu')

        self.word_ebd = nn.Embedding(t_n_words, t_d_word_ebd)

        self.textgen = SAERTextRNN(t_hidden_size, t_d_word_ebd, t_n_words,
                                   t_n_layers, t_dropout, t_rnn_type, d_state=t_input_size)
        self.textgen.word_ebd = self.word_ebd

        self.dropout = nn.Dropout(p=0.4)

    def _input_tensor(self, user_vct, item_vct):
        if self.match_tensor_type == 'concat':
            input_var = torch.cat([user_vct, item_vct], dim=1)
        elif self.match_tensor_type == 'multiply_minus':
            input_var = torch.cat(
                [user_vct - item_vct, user_vct * item_vct], dim=1)
        elif self.match_tensor_type == 'multiply':
            input_var = user_vct * item_vct
        elif self.match_tensor_type == 'multiply_concat':
            input_var = user_vct * item_vct
            input_var = torch.cat([user_vct, item_vct, input_var], dim=1)
        elif self.match_tensor_type == 'outter':
            input_var = torch.einsum(
                'bi,bj->bij', user_vct, item_vct).flatten(start_dim=1)
        elif self.match_tensor_type == 'outter_concat':
            input_var = torch.einsum(
                'bi,bj->bij', user_vct, item_vct).flatten(start_dim=1)
            input_var = torch.cat([user_vct, item_vct, input_var], dim=1)

        return input_var

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def get_input(self, data):
        user_var, item_var = data.users, data.items

        user_vct = self.user_ebd(user_var)
        item_vct = self.item_ebd(item_var)

        user_vct = self.dropout(user_vct)
        item_vct = self.dropout(item_vct)

        ui_input_var = self._input_tensor(user_vct, item_vct)

        ui_var = self.ui_mlp(ui_input_var)

        user_t_vct = self.user_t_ebd(user_var)
        item_t_vct = self.item_t_ebd(item_var)
        ui_t_var = torch.cat([user_t_vct, item_t_vct], dim=1)
        ui_t_var = self.dropout(ui_t_var)

        return AttrDict(
            user_vct=user_vct,
            item_vct=item_vct,
            ui_var=ui_var,
            ui_t_var=ui_t_var,
        )

    def forward(self, data, word_var=None, tf_rate=1, mode='both'):
        input_dict = self.get_input(data)

        if mode != 'review':
            ratings = self.rater(input_dict.ui_var)

        if mode != 'rate':
            review_output = self.textgen(
                input_dict, word_var, data=data, tf_rate=1)

        if mode == 'rate':
            return ratings
        elif mode == 'review':
            return review_output
        else:
            return ratings, review_output

    def rate(self, data):
        return self(data, mode='rate')

    def review(self, data, word_var, tf_rate=1):
        return self(data, word_var, tf_rate, mode='review')


class SentimentRegressor(nn.Module):
    '''
    Inputs:
        sen_var: padded sentence variable (word_seq, sen_batch)
        word_mask: (word_seq, sen_batch)
        rvw_lens: array of reviews' sentence lengths, [rvw_batch]
    Outputs:
        rvw_output: (rvw_batch)
        word_attn: (sen_batch, word_seq)
    '''

    def __init__(self, hidden_size, n_words, d_word_ebd, dropout=0., encoder_type='GRU'):
        super().__init__()

        self.word_ebd = nn.Embedding(n_words, d_word_ebd)
        self.dropout = nn.Dropout(dropout)

        self.encoder = RNNEncoder(hidden_size, d_word_ebd)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def forward(self, word_var, word_mask=None, rvw_lens=None):
        if len(word_var.size()) == 2:
            embedded = self.word_ebd(word_var)
        else:
            # (seq, batch, n_word) @ (1, n_word, d_ebd) = (seq, batch. d_ebd)
            embedded = torch.matmul(
                word_var, self.word_ebd.weight.unsqueeze(0))

        embedded = self.dropout(embedded)

        rvw_var, word_attn = self.encoder(embedded, word_mask, rvw_lens)

        rvw_var = self.dropout(rvw_var)
        rvw_output = self.out(rvw_var).squeeze(-1)
        return rvw_output, word_attn


class TextClassifier(nn.Module):
    '''
    Inputs:
        word_dist: word weight distribution
        word_mask: (word_seq, sen_batch)
        rvw_lens: array of reviews' sentence lengths, [rvw_batch]
    Outputs:
        rvw_output: (rvw_batch)
    '''

    def __init__(self, hidden_size, n_words, d_word_ebd, dropout=0., encoder_type='GRU'):
        super().__init__()

        self.word_ebd = nn.Embedding(n_words, d_word_ebd)
        self.word_ebd_dropout = nn.Dropout(dropout)

        self.encoder = RNNEncoder(hidden_size, d_word_ebd)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def forward(self, word_dist, word_mask=None, rvw_lens=None):
        if len(word_dist.size()) == 2:
            embedded = self.word_ebd(word_dist)
        else:
            # (seq, batch, n_word) @ (1, n_word, d_ebd) = (seq, batch. d_ebd)
            embedded = torch.matmul(
                word_dist, self.word_ebd.weight.unsqueeze(0))

        embedded = self.word_ebd_dropout(embedded)

        rvw_var, word_attn = self.encoder(embedded, word_mask, rvw_lens)

        rvw_output = self.out(rvw_var).squeeze(-1)
        return rvw_output
