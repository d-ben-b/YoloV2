import torch
from model import load_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None
with open('class_names', 'r') as f:
    CLASS_NAMES = f.read().split('\n')