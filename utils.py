from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import re
def max_length(tensor):
    return max(len(t) for t in tensor)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

def get_torch_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
