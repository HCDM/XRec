import json
import re
from .utils import EmbeddingMap

import config


class Vocabulary(EmbeddingMap):
    '''
    embedding maps for words
    '''

    pad = '<pad>'  # Padding token
    sos = '<sos>'  # Start of Sentence token
    eos = '<eos>'  # End of Sentence token
    num = '<num>'  # End of Sentence token
    unk = '<unk>'  # pretrained word embedding usually has this

    def __init__(self, tokens=[]):
        super().__init__(tokens)

        for special_token in [self.pad, self.sos, self.eos, self.num, self.unk]:
            if special_token not in self:
                self.append(special_token)

        self.pad_idx = self[self.pad]
        self.sos_idx = self[self.sos]
        self.eos_idx = self[self.eos]
        self.num_idx = self[self.num]
        self.unk_idx = self[self.unk]

    def word_2_idx(self, word):
        if word in self:
            return self[word]
        if re.fullmatch(r'([0-9]+|[0-9]*\.[0-9]+)', word):
            return self.num_idx

        return self.unk_idx

    def words_2_idx(self, words):
        idxes = [
            self.word_2_idx(word) for word in words if word
        ]
        # idxes.append(self.eos_idx)
        return idxes


with open(config.VOC_FILE) as vf:
    voc = Vocabulary(json.load(vf))
