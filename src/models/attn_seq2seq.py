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
    def __init__(self, input_dim, emb_dim, hidden_size, rnn_units, dropout,bidir):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, rnn_units, dropout=dropout,bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        src = src.transpose(0, 1)
        print(src.shape)
        embedded = self.dropout(self.embedding(src))
        print(embedded.shape)
        output, (hidden, cell) = self.rnn(embedded)
        print(output.shape,hidden.shape,cell.shape)
        return hidden, cell


class Attention(nn.Module):
    def __init__(self,):
        super(Attention,self).__init__()


class Decoder(nn.Module):
    def __init__(
        self, output_dim, emb_dim, hidden_size, rnn_units, dropout,bidir
    ):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size * (2 if bidir else 1), rnn_units, dropout=dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidir else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        print("Decoder")
        print( input.shape, hidden.shape, cell.shape)
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell





class Seq2seq(nn.Module):
    def __init__(
        self,
        input_vocab,
        output_vocab,
        enc_emb_dim,
        dec_emb_dim,
        hidden_size,
        rnn_units,
        enc_dropout,
        dec_dropout,
        bidir,
        **kwargs
    ):
        """[summary]

        Arguments:
            nn {[type]} -- [description]
            input_vocab {[type]} -- [description]
            output_vocab {[type]} -- [description]
            enc_emb_dim {[type]} -- [description]
            dec_emb_dim {[type]} -- [description]
            hidden_size {[type]} -- [description]
            rnn_units {[type]} -- [description]
            enc_dropout {[type]} -- [description]
            dec_dropout {[type]} -- [description]
        """
        super(Seq2seq, self).__init__()
        self.bidir=bidir
        self.encoder = Encoder(
            input_vocab, enc_emb_dim, hidden_size, rnn_units, enc_dropout,bidir=bidir,
        )
        self.decoder = Decoder(
            output_vocab, dec_emb_dim, hidden_size, rnn_units, dec_dropout,bidir=bidir,
        )
        self.template_zeros = Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, src, trg):
        """[summary]

        Arguments:
            src {[type]} -- [description]
            trg {[type]} -- [description]

        Keyword Arguments:
            teacher_forcing_ratio {float} -- [description] (default: {0.5})

        Returns:
            [type] -- [description]
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = self.template_zeros.repeat(trg_len, batch_size, trg_vocab_size)
        hidden, cell = self.encoder(src)
        if self.bidir:
            hidden=torch.cat((hidden[0:2,:,:],hidden[2:,:,:]),dim=2)
            cell=torch.cat((cell[0:2,:,:],cell[2:,:,:]),dim=2)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < torch.tensor(0.8)
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

    

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
