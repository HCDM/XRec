import random
import torch
from torch import nn

from .rnn import RNN
from ..utils import AttrDict


class TextRNNGenerator(nn.Module):
    '''
    Generate word sequence for a given state
    Inputs:
        state: (batch, d_state) init state, be converted to RNN state
        input_seq: (seq_length, batch_size) input sequence batch
        tf_rate: 0-1, 0: no teacher forcing, 1: always teacher forcing, default 1
    Outputs:
        output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence; shape=(seq_length, batch_size, voc.num_words)
    '''

    def __init__(self, d_hidden, d_input, n_words, n_layers=1, dropout=0., rnn_type='GRU'):
        super(TextRNNGenerator, self).__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.output_size = n_words
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        self.dropout = nn.Dropout(dropout)

        # Define layers
        self.word_ebd = nn.Embedding(n_words, d_input)
        self.word_ebd_dropout = nn.Dropout(dropout)

        self.word_rnn = RNN(rnn_type, d_input, d_hidden, num_layers=n_layers, batch_first=True)

        self.out = nn.Linear(d_hidden, self.output_size)

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def decode(self, input_seq, hidden):
        # Get embedding of current input word
        embedded = self.word_ebd(input_seq)
        embedded = self.word_ebd_dropout(embedded)

        output, hidden = self.word_rnn(embedded, hidden)

        output = self.out(output).log_softmax(-1)

        return AttrDict(
            output=output,
            hidden=hidden
        )

    def forward(self, input_seq, init_hidden, tf_rate=1):
        # teacher forcing applies to entire sequence
        teacher_forcing = random.random() < tf_rate

        if teacher_forcing:
            # Teacher forcing: feed the entire sequence of batch ground truth
            return self.decode(input_seq, init_hidden)
        else:
            # No teacher forcing: Forward batch of sequences through decoder one time step at a time
            hidden = init_hidden
            outputs = []
            max_length = input_seq.size(0)

            decoder_var = input_seq[0].view(1, -1)

            for _ in range(max_length):
                output_step, hidden = self.decode(decoder_var, hidden)

                outputs.append(output_step)

                # next input is decoder's own current output
                _, decoder_var = output_step.max(2)

            output_var = torch.cat(outputs, 0)

            return output_var
