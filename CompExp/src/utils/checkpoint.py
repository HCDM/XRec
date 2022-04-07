import os
import pickle
import torch

META_FILE = 'meta.pt'


class CheckpointManager:
    def __init__(self, path):
        self.path = path

    def __fullpath(self, name):
        return os.path.join(self.path, name)

    def __ensure_folder(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, name, data, best=False):
        filepath = self.__fullpath(f'{name}.tar')
        self.__ensure_folder()
        torch.save(data, filepath)

        if best:
            self.set_best(name)

    def save_meta(self, **kargs):
        self.__ensure_folder()
        with open(self.__fullpath(META_FILE), 'wb') as fout:
            pickle.dump(kargs, fout)

    def delete(self, name):
        ckpt_path = self.__fullpath(f'{name}.tar')
        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)

    def load(self, name, device):
        filepath = self.__fullpath(f'{name}.tar')
        checkpoint = torch.load(filepath, map_location=device)

        meta = {}
        if os.path.exists(self.__fullpath(META_FILE)):
            with open(self.__fullpath(META_FILE), 'rb') as fin:
                meta = pickle.load(fin)

        return {**checkpoint, **meta}

    def set_best(self, name):
        best_path = self.__fullpath('best.tar')
        if os.path.exists(best_path) or os.path.islink(best_path):
            os.remove(best_path)

        os.symlink(self.__fullpath(f'{name}.tar'), best_path)

    def loadMeta(self):
        pass
