import torch
from .config import *
import importlib

# Torch device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

del torch

def load(model_name):
  return importlib.import_module(f'.{DS}_models.' + model_name, package=__name__)
