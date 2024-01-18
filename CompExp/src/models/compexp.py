# from itertools import accumulate, chain
import functools
import torch
from torch import nn

from .text_encoder import TextRNNEncoder
from .text_generator import TextRNNGenerator
from ..utils import AttrDict


class CompExp(nn.Module):
    '''
    Inputs:
      data: AttrDict
        users
        items
        scores
        words
        mask

    Outputs:
    '''

    def __init__(
        self,

        n_words,
        d_word_ebd,
        d_exp_ebd,

        rnn_type,
        dropout,
        norm_vct=False,
        keppa=1,  # extraction temperature

        ext_policy='max'
    ):
        super().__init__()

        self.t_encoder = TextRNNEncoder(d_exp_ebd, n_words, d_word_ebd, inner_attn=False, dropout=dropout)

        self.r_2_hidden = nn.Embedding(41, 16)
        self.ctx_out = nn.Linear(d_exp_ebd + 16, d_exp_ebd)

        self.dropout = nn.Dropout(p=0.4)

        self.t_decoder = TextRNNGenerator(d_exp_ebd, d_word_ebd, n_words, rnn_type=rnn_type, dropout=dropout)

        # self.energy_2_step = nn.Sequential(
        #     nn.Linear(1, 1),
        #     nn.Sigmoid(),
        #     nn.Linear(1, 1)
        # )

        self.energy_2_step = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU()
        )

        self.norm_vct = norm_vct

        self.ext_policy = ext_policy
        self.keppa = keppa

    @property
    @functools.lru_cache()
    def extractor(self):
        # return 1
        return CompExpExt(self)

    def load_pretrained_word_ebd(self, weight):
        self.t_encoder.load_pretrained_word_ebd(weight)
        self.t_decoder.load_pretrained_word_ebd(weight)

    def extract(self, data):
        _, flatten_exp_vcts = self.t_encoder(data.item_words, lens=data.item_words_lens)
        _, flatten_ref_vcts = self.t_encoder(data.ref_words, lens=data.ref_words_lens)

        # fuse ref with delta ratings
        flatten_ctx_vcts = self.ctx_out(
            torch.cat([
                flatten_ref_vcts, self.r_2_hidden(data.delta_ratings.long() + 20)  # deltta rating from -20 to 20
            ], dim=1)
        )
        # flatten_exp_vcts = self.dropout(flatten_exp_vcts)
        # flatten_ctx_vcts = self.dropout(flatten_ctx_vcts)

        # norm
        if self.norm_vct:
            flatten_exp_vcts = flatten_exp_vcts / flatten_exp_vcts.norm(2, dim=1, keepdim=True)
            flatten_ctx_vcts = flatten_ctx_vcts / flatten_ctx_vcts.norm(2, dim=1, keepdim=True)

        # recover batch dimention
        item_exp_mask = data.item_exp_mask
        exp_vcts = torch.zeros((*item_exp_mask.size(), flatten_exp_vcts.size(-1)), device=flatten_exp_vcts.device)
        exp_vcts[item_exp_mask] = flatten_exp_vcts

        ref_exp_mask = data.ref_exp_mask
        ctx_vcts = torch.zeros((*ref_exp_mask.size(), flatten_ctx_vcts.size(-1)), device=flatten_ctx_vcts.device)
        ctx_vcts[ref_exp_mask] = flatten_ctx_vcts

        # (batch, n_ref_exps, n_item_exps)
        energies = ctx_vcts.bmm(exp_vcts.transpose(1, 2))
        if not self.norm_vct:
            energy_norm = ctx_vcts.norm(2, dim=2, keepdim=True) * exp_vcts.norm(2, dim=2).unsqueeze(1)
            energy_norm[energy_norm == 0] += 0.01
            energies /= energy_norm

        # (batch, n_item_exps)
        # exp_energies = (energies * ref_weights.unsqueeze(-1)).sum(1)
        # exp_energies[~data.item_exp_mask] = float('-inf')
        ref_weights = ref_exp_mask.float() / ref_exp_mask.sum(1, keepdim=True)
        exp_energies = ((energies * self.keppa).exp() * ref_weights.unsqueeze(-1)).sum(1)
        exp_energies[~data.item_exp_mask] = 0
        ext_probs = exp_energies / exp_energies.sum(1, keepdim=True)

        return ext_probs, energies, exp_vcts, ctx_vcts

    def decode(self, *args, **kargs):
        return self.t_decoder.decode(*args, **kargs)

    def forward(self, decode_input, data):
        ext_probs, energies, exp_vcts, ctx_vcts = self.extract(data)

        if self.ext_policy == 'max':
            _, ext_indices = torch.max(ext_probs, 1)
        elif self.ext_policy == 'sample':
            ext_indices = torch.multinomial(ext_probs, 1).squeeze(-1)

        ext_exp_vcts = exp_vcts[range(ext_indices.size(0)), ext_indices]

        # (batch, n_ref_exps, n_item_exps) -> (batch, n_ref_exps)
        ext_ref_energies = energies.transpose(1, 2)[range(ext_indices.size(0)), ext_indices]
        # (batch, n_ref_exps, 1)
        step_sizes = self.energy_2_step(
            (
                ext_ref_energies.exp()
            ).unsqueeze(-1)
        )

        ref_exp_mask = data.ref_exp_mask

        # (batch, dim)
        dir_vcts = (ctx_vcts * step_sizes).sum(1) / ref_exp_mask.sum(1, keepdim=True)
        # dir_vcts = dir_vcts / dir_vcts.norm(2, dim=1, keepdim=True)

        src_vcts = ext_exp_vcts + dir_vcts

        if self.norm_vct:
            src_vcts = src_vcts / src_vcts.norm(2, dim=1, keepdim=True)

        init_hidden = src_vcts.unsqueeze(0)
        gen_result = self.t_decoder(decode_input, init_hidden)

        return AttrDict(
            ext_probs=ext_probs,
            ext_indices=ext_indices,
            output=gen_result.output,
            hidden=gen_result.hidden
        )


class CompExpExt(nn.Module):
    def __init__(self, CompExp_model):
        super().__init__()
        self.CompExp_model = CompExp_model

    def forward(self, data):
        ext_probs, energies, _, _ = self.CompExp_model.extract(data)

        return AttrDict(
            probs=ext_probs,
            energies=energies
        )
