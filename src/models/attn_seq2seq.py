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
        self.bidir=bidir
        self.hidden_size = hidden_size
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, rnn_units, dropout=dropout,bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        """[summary]

        Arguments:
            src {[tensor]} --  Shape: batch X srcLen

        Returns:
            [output] -- Shape: srcLen X batch X hidden * (2 if bidir else 1) 
            [hidden] -- Shape: (4 if bidir else 2) X batch X hidden
            [cell] -- Shape: (4 if bidir else 2) X batch X hidden

        """
        src = src.transpose(0, 1)
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        if self.bidir:
            hidden = hidden.view(self.rnn_units,2,-1,self.hidden_size)
            cell = cell.view(self.rnn_units,2,-1,self.hidden_size)
            hidden = torch.tanh(torch.cat((hidden[:,0,:,:], hidden[:,1,:,:]), dim = -1))
            cell = torch.tanh(torch.cat((cell[:,0,:,:], cell[:,1,:,:]), dim = -1))
        return output,hidden, cell


class Attention(nn.Module):
    def __init__(self,):
        super(Attention,self).__init__()

    def forward(self,encoder_hidden, hidden):
        score = self.score(encoder_hidden,hidden)
        return F.softmax(score)

    def score(self,encoder_hidden,hidden):
        hs = encoder_hidden.permute(1,0,2)
        ht = hidden.unsqueeze(-1)
        score = hs.bmm(ht).squeeze(-1)
        return score


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
        self.attention = Attention()
        self.fc = nn.Linear(hidden_size * (2 if bidir else 1) * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_hidden):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        hidden_last=hidden[0,:,:]
        attention_weights = self.attention(encoder_hidden,hidden_last)
        attention_context = attention_weights.unsqueeze(1).bmm(encoder_hidden.permute(1,0,2)).squeeze(1)
        decoder_output = torch.cat((attention_context,output.squeeze(0)),dim=1)
        prediction = self.fc(decoder_output)
        return prediction, hidden, cell





class AttnSeq2seq(nn.Module):
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
        super(AttnSeq2seq, self).__init__()
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
        encoder_outputs , hidden , cell = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
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
