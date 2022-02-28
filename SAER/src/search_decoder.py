import random
import torch
from torch import nn
from torch.nn.functional import mse_loss
# from collections import Counter
from .voc import voc

from .utils import AttrDict, gumbel_softmax


class AbstractSearchDecoder(nn.Module):
    '''
    Inputs:
        batch_data: AttrDict{
            users: (batch)
            items: (batch)
        }

    Outputs:
        AttrDict{
            words: (seq, batch)
            probs: (seq, batch)
            hiddens: list
            rvw_lens: list
        }
    '''
    def forward(self, batch_data):
        user_var, _ = batch_data.users, batch_data.items
        input_dict = self.model.get_input(batch_data)

        # convert ui to TextRNN init hidden
        hidden = self.model.textgen.get_init_hidden(input_dict.ui_t_var)

        start_var = torch.full(user_var.view(-1).size(), voc.sos_idx, dtype=torch.long, device=user_var.device)

        return self.decode(hidden, start_var, data=batch_data, input_dict=input_dict)


class SearchDecoder(AbstractSearchDecoder):
    def __init__(self, model, max_length=15, greedy=False, sample_length=float('inf'), topk=0):
        super().__init__()
        self.model = model
        self.max_length = max_length
        self.greedy = greedy
        self.sample_length = sample_length
        self.topk = topk

    def set_greedy(self, greedy):
        self.greedy = greedy

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        if not max_length:
            max_length = self.max_length

        greedy = self.greedy

        words = []
        probs = []
        hiddens = []

        batch_size = start_var.size(-1)
        rvw_lens = torch.zeros(batch_size, dtype=torch.long).to(start_var.device)

        decoder_var = start_var.view(1, -1)

        for i in range(max_length):
            output_dict = self.model.textgen.decode(decoder_var, hidden, data=data, input_dict=input_dict)

            output, hidden = output_dict.output, output_dict.hidden

            # (1, batch, n_words) -> (batch, n_words)
            word_probs = output.exp().squeeze(0)

            # (batch)
            if not greedy and i < self.sample_length:
                if self.topk:
                    word_probs, word_idxes = word_probs.topk(self.topk)

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
            hiddens.append(hidden)

            # verify eos
            is_eos = word_var == voc.eos_idx
            not_end = rvw_lens == 0

            if i != max_length - 1:
                rvw_lens[not_end * is_eos] = i + 1

                # break if whole batch end
                if (rvw_lens != 0).all():
                    break

                # next input is decoder's own current output
                # add seq dim
                decoder_var = word_var.unsqueeze(0)
            else:
                # reach max len
                rvw_lens[not_end] = max_length

        # (seq, batch)
        words = torch.stack(words, dim=0)
        probs = torch.stack(probs, dim=0)

        return AttrDict(
            words=words,
            probs=probs,
            hiddens=hiddens,
            rvw_lens=rvw_lens
        )

class BeamSearchDecoder(AbstractSearchDecoder):
    def __init__(self, model, max_length=15, beam_width=5, mode='best'):
        super(BeamSearchDecoder, self).__init__()
        self.model = model
        self.beam_width = beam_width
        self.mode = mode
        self.max_length = max_length

    class BeamSearchNode:
        def __init__(self, hidden, idx, value=None, previousNode=None, logp=0, depth=0):
            self.prevnode = previousNode
            self.hidden = hidden
            self.value = value
            self.idx = idx
            self.logp = logp
            self.depth = depth

        def eval(self):
            # for now, simply choose the one with maximum average
            return self.logp / float(self.depth)

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        sos, eos = voc.sos_idx, voc.eos_idx

        # Number of sentence to generate
        endnodes = []

        # Start with the start of the sentence token
        root_idx = start_var
        root = self.BeamSearchNode(hidden, root_idx)
        leaf = [root]

        for dep in range(self.max_length):
            candidates = []

            for prevnode in leaf:
                decoder_input = prevnode.idx.view(1, 1)

                # Forward pass through decoder
                # decode for one step using decoder

                output_dict = self.model.textgen.decode(decoder_input, prevnode.hidden, data=data, input_dict=input_dict)

                output, decoder_hidden = output_dict.output, output_dict.hidden

                output = nn.functional.log_softmax(output, dim=2)

                values, indexes = output.topk(self.beam_width)

                for i in range(self.beam_width):
                    idx = indexes[0][0][i]
                    value = values[0][0][i]

                    node = self.BeamSearchNode(decoder_hidden, idx, value, prevnode, value + prevnode.logp, dep + 1)

                    candidates.append(node)

            candidates.sort(key=lambda n: n.logp, reverse=True)

            leaf = []
            for candiate in candidates[:self.beam_width]:
                if candiate.idx == eos:
                    endnodes.append(candiate)
                else:
                    leaf.append(candiate)

            # sentecnes don't need to be beam_width exactly, here just for simplicity
            if len(endnodes) >= self.beam_width:
                endnodes = endnodes[:self.beam_width]
                break

        # arrive max length before having enough results
        if len(endnodes) < self.beam_width:
            endnodes = endnodes + leaf[:self.beam_width - len(endnodes)]

        # choose the max/random from the results
        if self.mode == 'all':
            return [self.trace_tokens(n, sos) for n in endnodes]

        if self.mode == 'random':
            endnode = random.choice(endnodes)
        else:
            endnode = max(endnodes, key=lambda n: n.eval())

        tokens, probs = self.trace_tokens(endnode, sos)
        lengths = [tokens.size(0)]

        return AttrDict(
            words=tokens.unsqueeze(1),
            probs=probs.unsqueeze(1),
            rvw_lens=lengths
        )

    def trace_tokens(self, endnode, sos):
        tokens = []
        scores = []
        while endnode.idx != sos:
            tokens.append(endnode.idx)
            scores.append(endnode.value)
            endnode = endnode.prevnode

        tokens.reverse()
        scores.reverse()

        tokens = torch.stack(tokens)
        scores = torch.stack(scores)

        return tokens, scores


class DebiasSearchDecoder(nn.Module):
    def __init__(self, model, global_model, max_length=15, n_debias=5):
        super().__init__()
        self.model = model
        self.global_model = global_model
        self.max_length = max_length

        self.global_lambda = 0.4
        self.n_debias = n_debias

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        if not max_length:
            max_length = self.max_length

        words = []
        probs = []
        hiddens = []

        batch_size = start_var.size(-1)
        rvw_lens = torch.zeros(batch_size, dtype=torch.long).to(start_var.device)

        decoder_var = start_var.view(1, -1)

        global_hidden = None

        for i in range(max_length):
            output_dict = self.model.textgen.decode(decoder_var, hidden, data=data, input_dict=input_dict)

            output, hidden = output_dict.output, output_dict.hidden

            global_output, global_hidden = self.global_model.textgen.decode(decoder_var, global_hidden)

            # (1, batch, n_words) -> (batch, n_words)
            word_probs = output.squeeze(0).log_softmax(-1)

            # (batch)
            if i < self.n_debias:
                global_word_probs = global_output.squeeze(0).log_softmax(-1)

                word_probs = word_probs - self.global_lambda * global_word_probs

            prob_var, word_var = word_probs.max(-1)

            # [(batch),...]
            words.append(word_var)
            probs.append(prob_var)
            hiddens.append(hidden)

            # verify eos
            is_eos = word_var == voc.eos_idx
            not_end = rvw_lens == 0

            if i != max_length - 1:
                rvw_lens[not_end * is_eos] = i + 1

                # break if whole batch end
                if (rvw_lens != 0).all():
                    break

                # next input is decoder's own current output
                # add seq dim
                decoder_var = word_var.unsqueeze(0)
            else:
                # reach max len
                rvw_lens[not_end] = max_length

        # (seq, batch)
        words = torch.stack(words, dim=0)
        probs = torch.stack(probs, dim=0)

        return AttrDict(
            words=words,
            probs=probs,
            hiddens=hiddens,
            rvw_lens=rvw_lens
        )


class GumbelDecoder(AbstractSearchDecoder):
    def __init__(self, model, max_length=15, temperature=0.5):
        super().__init__()
        self.model = model
        self.max_length = max_length

        self.temperature = temperature

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        if not max_length:
            max_length = self.max_length

        word_dists = []

        batch_size = start_var.size(-1)
        rvw_lens = torch.zeros(batch_size, dtype=torch.long).to(start_var.device)

        decoder_var = start_var.view(1, -1)

        for i in range(max_length):
            output_dict = self.model.textgen.decode(decoder_var, hidden, data=data, input_dict=input_dict)

            output, hidden = output_dict.output, output_dict.hidden

            # (1, batch, n_words) -> (batch, n_words)
            log_probs = output.log_softmax(-1).squeeze(0)

            # (batch, n_words)
            word_dist = gumbel_softmax(log_probs, self.temperature, ST=True)
            _, word_var = word_dist.max(-1)

            # [(batch),...]
            word_dists.append(word_dist)

            # verify eos
            is_eos = word_var == voc.eos_idx
            not_end = rvw_lens == 0

            if i != max_length - 1:
                rvw_lens[not_end * is_eos] = i + 1

                # break if whole batch end
                if (rvw_lens != 0).all():
                    break

                # next input is decoder's own current output
                # add seq dim
                decoder_var = word_var.unsqueeze(0)
            else:
                # reach max len
                rvw_lens[not_end] = max_length

        # (seq, batch, n_words)
        word_dists = torch.stack(word_dists, dim=0)

        return AttrDict(
            words=word_dists, # words distribution
            rvw_lens=rvw_lens
        )


class MCSearchDecoder(AbstractSearchDecoder):
    '''
    Monte Carlo Search Decoder
    '''

    def __init__(self, model, max_length=15, ranker=None, topk=5, rollout=5):
        super().__init__()
        self.model = model
        self.max_length = max_length
        self.ranker = ranker
        self.topk = topk
        self.rollout = rollout
        self.n_actions = 5

        self.sen_gate_th = 0.35
        self.fea_gate_th = 0.15

        self.sampler = SearchDecoder(model, greedy=False)

    def mc_search(self, hidden, word_idxes, prefix_words, scores, data, input_dict, search_length, rollout_idx):
        '''
        hidden: (batch, d_hidden)
        word_idxes: (batch, k)
        prefix_words: (seq, batch)
        scores: (batch)
        '''

        # Select
        word_idxes = word_idxes[rollout_idx]
        hidden = hidden[:, rollout_idx]
        prefix_words = prefix_words[:, rollout_idx]
        scores = scores[rollout_idx]

        data = AttrDict({
            k: t[rollout_idx]
            for k, t in data
        })

        input_dict = AttrDict({
            k: t[rollout_idx]
            for k, t in input_dict
        })

        # Expand: flatten(rollout, topk, batch)
        word_idxes = word_idxes.transpose(0, 1).expand(self.rollout, -1, -1).flatten(0, 2)

        d_expand = self.rollout * self.n_actions
        hidden = hidden.expand(d_expand, -1, -1, -1).transpose(0, 1).flatten(1, 2).contiguous()

        prefix_words = prefix_words.expand(d_expand, -1, -1).transpose(0, 1).flatten(1, 2)

        scores = scores.expand(d_expand, -1).flatten(0, 1)

        data = AttrDict({
            k: t.expand(d_expand, *t.size()).flatten(0, 1)
            for k, t in data
        })

        input_dict = AttrDict({
            k: t.expand(d_expand, *t.size()).flatten(0, 1)
            for k, t in input_dict
        })

        # Monte Carlo Search with Rollout policy
        sampled_result = self.sampler.decode(hidden, word_idxes, max_length=search_length, data=data, input_dict=input_dict)

        # combine with given state & action
        samples = torch.cat([
            prefix_words, sampled_result.words
        ], dim=0)
        sample_lens = sampled_result.rvw_lens + len(prefix_words)

        sampled_scores, _ = self.ranker(samples, rvw_lens=sample_lens)

        action_values = mse_loss(sampled_scores, scores, reduction='none').view(self.rollout, self.n_actions, -1).mean(0).transpose(0, 1)

        return action_values

    def decode(self, hidden, start_var, data=None, input_dict=None, max_length=None):
        if not max_length:
            max_length = self.max_length

        scores = self.model.rate(data)
        words = []
        probs = []
        hiddens = []

        batch_size = start_var.size(-1)
        rvw_lens = torch.zeros(batch_size, dtype=torch.long, device=start_var.device)

        decoder_var = start_var.view(1, -1)

        feature_counts = torch.zeros(batch_size, len(voc), dtype=torch.long, device=start_var.device)

        for i in range(max_length):
            output_dict = self.model.textgen.decode(decoder_var, hidden, data=data, input_dict=input_dict)

            output, hidden = output_dict.output, output_dict.hidden

            # (1, batch, n_words) -> (batch, n_words)
            word_probs = output.squeeze(0).exp()

            # (batch, k)
            word_k_probs, word_k_idx = word_probs.topk(self.topk)

            # feature constraint reweight
            fea_idx = output_dict.feature_gates.squeeze(0) > self.fea_gate_th
            if fea_idx.sum() > 0:
                word_k_probs[fea_idx] /= 1 + 2 * feature_counts[fea_idx].gather(-1, word_k_idx[fea_idx])

            # random draw one from topk: (batch, 1)
            word_var = word_k_idx.gather(-1, torch.multinomial(word_k_probs, 1))

            # sentiment gate constraint
            rollout_idx = output_dict.rate_gates.squeeze(0) > self.sen_gate_th
            if i != 0 and rollout_idx.sum() > 0:
                # print(f'{rollout_idx.sum()} need rollout at {i}')

                action_idxes = torch.multinomial(word_k_probs, self.n_actions, replacement=True)
                actions = word_k_idx.gather(-1, action_idxes)

                seach_length = max_length - i - 1
                prefix_words = torch.stack(words, dim=0) if words else torch.tensor([], dtype=torch.long, device=actions.device)

                action_values = self.mc_search(hidden, actions, prefix_words, scores, data, input_dict, seach_length, rollout_idx)

                word_var[rollout_idx] = actions[rollout_idx].gather(-1, action_values.argmin(1, keepdim=True))

            # feature gate constraint
            if fea_idx.sum() > 0:
                feature_counts_delta = torch.zeros_like(feature_counts[fea_idx]).scatter_(1, word_var[fea_idx], 1)
                feature_counts[fea_idx] += feature_counts_delta

            prob_var = word_probs.gather(-1, word_var).squeeze(-1)
            word_var = word_var.squeeze(-1)

            # [(batch),...]
            words.append(word_var)
            probs.append(prob_var)
            hiddens.append(hidden)

            # verify eos
            is_eos = word_var == voc.eos_idx
            not_end = rvw_lens == 0

            if i != max_length - 1:
                rvw_lens[not_end * is_eos] = i + 1

                # break if whole batch end
                if (rvw_lens != 0).all():
                    break

                # next input is decoder's own current output
                # add seq dim
                decoder_var = word_var.unsqueeze(0)
            else:
                # reach max len
                rvw_lens[not_end] = max_length

        # (seq, batch)
        words = torch.stack(words, dim=0)
        probs = torch.stack(probs, dim=0)

        return AttrDict(
            words=words,
            probs=probs,
            hiddens=hiddens,
            rvw_lens=rvw_lens
        )
