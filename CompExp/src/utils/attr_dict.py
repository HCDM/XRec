import torch


class AttrDict(object):
    def __init__(self, d={}, **kargs):
        self.__dict__.update(d, **kargs)

    def update(self, d={}, **kargs):
        self.__dict__.update(d, **kargs)
        return self

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        raise AttributeError(f'\'AttrDict\' object has no attribute \'{key}\'')

    def to(self, device):
        ''' move dict of data to device if tensor '''
        self.update({
            k: i.to(device)
            for k, i in self.__dict__.items()
            if torch.is_tensor(i)
        })

        return self

    def __iter__(self):
        return iter(self.__dict__.items())

    def __contains__(self, key):
        return key in self.__dict__
