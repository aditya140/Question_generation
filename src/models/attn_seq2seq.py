from torch import nn
import random
import torch
from torch.nn import Parameter
import numpy as np
import heapq
import torch.nn.functional as F
import sys

sys.path.append("./src/")
from inference.inference_helpers import Beam


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_size, rnn_units, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, rnn_units, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        pass
