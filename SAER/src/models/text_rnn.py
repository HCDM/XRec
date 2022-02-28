''' Text Generator '''

import random
import torch
from torch import nn

from .rnn import RNN, RNNStateAdaptor
from .attn import Attn
from ..utils import AttrDict

class ReviewHiRNN(nn.Module):
    '''
    Model review state as Hierarchical RNN

    Inputs:
        state: user-item vector used as the initial state for review; array of shape=(batch, vector_size)
        rvw_len: list of lengths of the review sentence
        target_var: review target words; shape=(seq, batch)
    Outputs:
        output_var: probabilities of words in each position; shape=(seq, batch, n_words)
    '''

    def __init__(self, d_state, d_hidden, n_words, word_ebd_size, n_layers, dropout, rnn_type):
        super().__init__()

        self.d_state = d_state
        self.d_hidden = d_hidden
        self.output_size = n_words
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        self.text = TextRNN(
            word_ebd_size,
            d_hidden,
            word_ebd_size,
            n_words,
            n_layers,
            dropout,
            rnn_type
        )

        self.init_input = nn.Parameter(torch.zeros(d_hidden))

        self.rnn = RNN(rnn_type, d_hidden, d_hidden, num_layers=n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.to_hidden = RNNStateAdaptor(rnn_type, d_state, d_hidden, n_layers)

        self.searcher = None

    def load_pretrained_word_ebd(self, weight):
        self.text.load_pretrained_word_ebd(weight)

    def review_state(self, state, rvw_lens):
        batch_size = len(rvw_lens)
        hidden = self.to_hidden(state)

        outputs = []

        input_var = self.init_input.view(1, 1, -1).expand(-1, batch_size, -1)

        for _ in range(max(rvw_lens)):
            input_var, hidden = self.rnn(input_var, hidden)
            outputs.append(input_var)

        # shape=(seq, batch, hidden)
        output_var = torch.cat(outputs, 0)

        # shape=(batch, hidden)
        sen_state_seq = torch.cat([
            output_var[:l, i] for i, l in enumerate(rvw_lens)
        ], 0)

        return sen_state_seq

    def search(self, state, rvw_lens):
        sen_state_seq = self.review_state(state, rvw_lens)

        results = []
        for i in range(sen_state_seq.size(0)):
            init_state = sen_state_seq[i].unsqueeze(0)
            res = self.searcher(init_state)
            results.append(res)

        return results

    def set_searcher(self, searcher):
        self.searcher = searcher

    def forward(self, state, rvw_lens, target_var):
        if self.searcher:
            return self.search(state, rvw_lens)

        sen_state_seq = self.review_state(state, rvw_lens)

        sen_var, _ = self.text(sen_state_seq, target_var, tf_rate=1)
        return sen_var


class TextRNN(nn.Module):
    '''
    Generate word sequence for a given state

    Inputs:
        state: (batch, d_state) init state, be converted to RNN state
        input_seq: (seq_length, batch_size) input sequence batch
        tf_rate: 0-1, 0: no teacher forcing, 1: always teacher forcing, default 1

    Outputs:
        output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence; shape=(seq_length, batch_size, voc.num_words)
    '''

    def __init__(self, d_hidden, d_input, n_words, n_layers, dropout, rnn_type, d_state=None):
        super(TextRNN, self).__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.output_size = n_words
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        self.dropout = nn.Dropout(dropout)

        # Define layers
        self.word_ebd = nn.Embedding(n_words, d_input)
        self.word_ebd_dropout = nn.Dropout(dropout)

        self.adapt_init_state = d_state is not None
        if self.adapt_init_state:
            self.to_hidden = RNNStateAdaptor(rnn_type, d_state, d_hidden, n_layers)

        self.decoder = RNN(rnn_type, d_input, d_hidden, num_layers=n_layers, dropout=(0 if n_layers == 1 else dropout))

        self.out = nn.Linear(d_hidden, self.output_size)

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def get_init_hidden(self, state):
        return self.to_hidden(state) if self.adapt_init_state else state

    def decode(self, input_seq, hidden, data=None, input_dict=None):
        # Get embedding of current input word
        embedded = self.word_ebd(input_seq)
        embedded = self.word_ebd_dropout(embedded)

        output, hidden = self.decoder(embedded, hidden)

        output = self.out(output).log_softmax(-1)

        return AttrDict(
            output=output,
            hidden=hidden
        )

    def forward(self, input_dict, input_seq, data=None, tf_rate=1):
        init_hidden = self.get_init_hidden(input_dict.ui_t_var)

        # teacher forcing applies to entire sequence
        teacher_forcing = random.random() < tf_rate

        if teacher_forcing:
            # Teacher forcing: feed the entire sequence of batch ground truth
            return self.decode(input_seq, init_hidden, data, input_dict)
        else:
            # No teacher forcing: Forward batch of sequences through decoder one time step at a time
            hidden = init_hidden
            outputs = []
            max_length = input_seq.size(0)

            decoder_var = input_seq[0].view(1, -1)

            for _ in range(max_length):
                output_step, hidden = self.decode(decoder_var, hidden, data)

                outputs.append(output_step)

                # next input is decoder's own current output
                _, decoder_var = output_step.max(2)

            output_var = torch.cat(outputs, 0)

            # TODO handle no tf response with AttrDict
            return output_var


class SAERTextRNN(TextRNN):
    '''
    Generate word sequence for a given state

    Inputs:
        state: (batch, d_state) init state, be converted to RNN state
        input_seq: (seq_length, batch_size) input sequence batch
        tf_rate: 0-1, 0: no teacher forcing, 1: always teacher forcing, default 1

    Outputs:
        output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence; shape=(seq_length, batch_size, voc.num_words)
    '''

    def __init__(self, d_hidden, d_input, n_words, n_layers, dropout, rnn_type, d_state=None, use_attn=False, use_rating_gate=True):
        super().__init__(d_hidden, d_input, n_words, n_layers, dropout, rnn_type, d_state)

        self.use_attn = use_attn
        if use_attn:
            self.mem_attn = Attn(d_hidden, method='general')
            self.out = nn.Linear(2 * d_hidden, self.output_size)

        self.use_rating_gate = use_rating_gate
        if self.use_rating_gate:
            self.rating_gate = nn.Sequential(
                nn.Linear(d_hidden, 1),
                nn.Sigmoid()
            )
            self.score_out = nn.Linear(32, d_hidden)

            self.o_out = nn.Linear(d_hidden, d_hidden)

        self.feature_gate = nn.Sequential(
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )
        self.feature_state = nn.Linear(d_hidden, 16)
        self.feature_attn = Attn(d_input, method='general', input_dim=80)

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def get_init_hidden(self, state):
        return self.to_hidden(state) if self.adapt_init_state else state

    def decode(self, input_seq, hidden, data=None, input_dict=None):
        # Get embedding of current input word
        embedded = self.word_ebd(input_seq)
        embedded = self.word_ebd_dropout(embedded)

        # if self.adapt_input:
        #     input_ctx = input_ctx.unsqueeze(0).expand(embedded.size(0), -1, -1)
        #     embedded = torch.cat((embedded, input_ctx), 2)
        #     embedded = self.to_input(embedded)

        # Forward through RNN; rnn_output shape = (seq_len, batch_size, d_hidden)
        output, hidden = self.decoder(embedded, hidden)
        output = self.dropout(output)

        # user gate
        ui = torch.cat([input_dict.user_vct, input_dict.item_vct], 1).unsqueeze(0).expand(embedded.size(0), -1, -1)

        fg = self.feature_gate(output)
        features = self.word_ebd(data.i_features)
        f_attn_input = torch.cat([
            self.feature_state(output.transpose(0, 1)), ui.transpose(0, 1)
        ], dim=2)
        # f_attn_input = ui.transpose(0, 1)
        _, f_attn = self.feature_attn(f_attn_input, features, mask=data.if_mask)
        f_attn = f_attn.transpose(0, 1)

        if self.use_rating_gate:
            rg = self.rating_gate(output)

            # scores = input_dict.ratings.unsqueeze(1)
            scores = input_dict.ui_var

            score_out = self.score_out(scores).unsqueeze(0).expand(embedded.size(0), -1, -1)

            output = (self.o_out(output) + rg * score_out).tanh()
        else:
            rg = 0.

        output = self.out(output).softmax(-1) # / 100 reduce logits may increase diversity
        # print(output)

        f_word_dist = torch.zeros_like(output, device=output.device)
        f_word_dist.scatter_(2, data.i_features.unsqueeze(0).expand(output.size(0), -1, -1), f_attn)

        output = (1 - fg) * output + fg * f_word_dist
        output = output.log()

        return AttrDict(
            output=output,
            hidden=hidden,
            feature_gates=fg.squeeze(-1),
            rate_gates=rg.squeeze(-1),
        )

    def forward(self, input_dict, input_seq, data=None, tf_rate=1):
        init_hidden = self.get_init_hidden(input_dict.ui_t_var)

        # teacher forcing applies to entire sequence
        teacher_forcing = random.random() < tf_rate

        if teacher_forcing:
            # Teacher forcing: feed the entire sequence of batch ground truth
            return self.decode(input_seq, init_hidden, data, input_dict)
        else:
            # No teacher forcing: Forward batch of sequences through decoder one time step at a time
            hidden = init_hidden
            outputs = []
            max_length = input_seq.size(0)

            decoder_var = input_seq[0].view(1, -1)

            for _ in range(max_length):
                output_step, hidden = self.decode(decoder_var, hidden, data)

                outputs.append(output_step)

                # next input is decoder's own current output
                _, decoder_var = output_step.max(2)

            output_var = torch.cat(outputs, 0)

            # TODO handle no tf response with AttrDict
            return output_var
