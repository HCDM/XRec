# import random
from itertools import accumulate, chain
import torch
from torch import nn

from .voc import voc
from .utils import AttrDict, gumbel_softmax


def extract(model, batch_data):
    probs = model(batch_data).probs

    exted_probs, exted_indices = torch.max(probs,  1)

    item_words = batch_data.item_words.tolist()
    item_words_lens = batch_data.item_words_lens.tolist()
    item_exp_counts = batch_data.item_exp_mask.sum(1).tolist()

    probs = probs.tolist()

    exps = []
    i_offset = 0

    for i, (ep, j) in enumerate(zip(exted_probs.tolist(), exted_indices.tolist())):
        ext_idx = i_offset + j

        ext_exp, ext_len = item_words[ext_idx], item_words_lens[ext_idx]

        exp = [voc[w] for w in ext_exp[:ext_len]]

        exps.append(exp)
        i_offset += item_exp_counts[i]

    return exps


def decode(searcher, batch_data):
    search_result = searcher(batch_data)
    exps = search_result.words
    words_lens = search_result.words_lens

    exps = exps.tolist()
    words_lens = words_lens.tolist()

    exps = [
        [
            voc[w_idx] for w_idx in exp[:l]
            if w_idx != voc.eos_idx
        ]
        for exp, l in zip(exps, words_lens)  # review
    ]

    return exps


class ExtractDecoder:
    def __init__(self, model, greedy=False):
        super().__init__()
        self.model = model
        self.greedy = greedy

    def _indices_to_words(self, item_exted_indices, batch_data):
        item_exted_indices = item_exted_indices.tolist()
        item_exp_counts = batch_data.item_exp_mask.sum(1).tolist()
        item_idx_offsets = chain([0], accumulate(item_exp_counts))

        exted_indices = [idx + offset for idx, offset in zip(item_exted_indices, item_idx_offsets)]

        words = batch_data.item_words[exted_indices]
        words_lens = batch_data.item_words_lens[exted_indices]

        return words, words_lens

    def __call__(self, batch_data):
        result = self.model(batch_data)
        probs = result.probs

        if self.greedy:
            _, item_exted_indices = torch.max(probs, 1)
        else:
            item_exted_indices = torch.multinomial(probs, 1).view(-1)

        words, words_lens = self._indices_to_words(item_exted_indices, batch_data)
        ref_weights = result.energies.transpose(1, 2)[range(item_exted_indices.size(0)), item_exted_indices]

        return AttrDict(
            words=words,
            words_lens=words_lens,
            ref_weights=ref_weights
        ).update(result)


class GumbelSoftmaxExtractDecoder(ExtractDecoder):
    def __init__(self, model, temperature=0.5):
        super().__init__(model)
        self.temperature = temperature

    def __call__(self, batch_data):
        result = self.model(batch_data)
        log_probs = (result.probs + 1e-20).log()

        # print(result.probs)
        # print(batch_data.item_exp_mask)

        ext_dist = gumbel_softmax(log_probs, self.temperature, ST=True, mask=batch_data.item_exp_mask)
        _, item_exted_indices = ext_dist.max(-1)

        words, words_lens = self._indices_to_words(item_exted_indices, batch_data)

        # apply ext distribution to words to rewrite
        # words_dist = torch.zeros((*words.size(), len(voc)), device=words.device)
        # words_dist.scatter_(2, words.unsqueeze(-1), 1)
        # # (batch, seq, voc) * (batch, 1, 1)
        # words_dist = words_dist *

        # sample_weights = ext_dist[ext_dist == 1].view(-1, 1, 1)

        return AttrDict(
            words=words,
            words_lens=words_lens,
            ext_dist=ext_dist
        ).update(result)


class AbstractSearchDecoder(nn.Module):
    def forward(self, batch_data):
        users = batch_data.users
        start_var = torch.full((users.size(0), 1), voc.sos_idx, dtype=torch.long, device=users.device)

        return self.decode(start_var, batch_data)


class SearchDecoder(AbstractSearchDecoder):
    """
    Inputs:
        user_var
        item_var
    Outputs:
        words
        probs
        hiddens
        words_lens
    """

    def __init__(self, model, max_length=15, greedy=False, sample_length=float('inf'), topk=0):
        super().__init__()
        self.model = model
        self.max_length = max_length
        self.greedy = greedy
        self.sample_length = sample_length
        self.topk = topk

    def set_greedy(self, greedy):
        self.greedy = greedy

    def decode(self, start_var, data=None, max_length=None):
        if not max_length:
            max_length = self.max_length

        greedy = self.greedy

        words = []
        probs = []
        # hiddens = []

        batch_size = start_var.size(0)
        words_lens = torch.zeros(batch_size, dtype=torch.long, device=start_var.device)

        decoder_var = start_var
        hidden = None
        for i in range(max_length):
            if not i:
                output_dict = self.model(decoder_var, data)
                init_result = output_dict
            else:
                output_dict = self.model.decode(decoder_var, hidden)

            output, hidden = output_dict.output, output_dict.hidden

            # (batch, 1, n_words) -> (batch, n_words)
            word_probs = output.squeeze(1).exp()

            # temporarily disable sample eos at begining
            # if not i:
            #     word_probs[..., voc.eos_idx] = 0

            # (batch)
            if not greedy and i < self.sample_length:
                if self.topk:
                    # not free sampling, limit it to the popular terms
                    k = self.topk
                    word_probs, word_idxes = word_probs.topk(k)

                    word_var = torch.multinomial(word_probs, 1)
                    prob_var = word_probs.gather(-1, word_var).view(-1)
                    word_var = word_idxes.gather(-1, word_var).view(-1)
                else:
                    word_var = torch.multinomial(word_probs, 1)
                    prob_var = word_probs.gather(-1, word_var).view(-1)
                    word_var = word_var.view(-1)
            else:
                prob_var, word_var = word_probs.max(-1)

            # [(batch),...]
            words.append(word_var)
            probs.append(prob_var)
            # hiddens.append(hidden)

            # verify eos
            is_eos = word_var == voc.eos_idx
            not_end = words_lens == 0

            if i != max_length - 1:
                words_lens[not_end * is_eos] = i + 1
                # words_lens[not_end * is_eos] = i  # exclude eos

                # break if whole batch end
                if (words_lens != 0).all():
                    break

                # next input is decoder's own current output
                # add seq dim
                decoder_var = word_var.unsqueeze(1)
            else:
                # reach max len
                words_lens[not_end] = max_length

        # (batch, seq)
        words = torch.stack(words, dim=1)
        probs = torch.stack(probs, dim=1)

        return AttrDict(
            words=words,
            probs=probs,
            # hiddens=hiddens,
            words_lens=words_lens
        ).update(init_result)
