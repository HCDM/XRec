class EmbeddingMap:
    '''
    convert index to token and vice versa
    '''

    def __init__(self, tokens=[]):
        self.token_2_index = {}
        self.tokens = [t for t in tokens] # clone

        for index, token in enumerate(tokens):
            self.token_2_index[token] = index

    def append(self, token):
        self.tokens.append(token)
        self.token_2_index[token] = len(self.tokens) - 1

    def size(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.token_2_index

    def __len__(self):
        return len(self.token_2_index)

    def __getitem__(self, key):
        if type(key) is int:
            return self.tokens[key]
        else:
            return self.token_2_index[key]
