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
        src = src.transpose(0, 1)
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, output_dim, emb_dim, hidden_size, rnn_units, dropout, embedding=None
    ):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, rnn_units, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
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
        self.encoder = Encoder(
            input_vocab, enc_emb_dim, hidden_size, rnn_units, enc_dropout,
        )
        self.decoder = Decoder(
            output_vocab, dec_emb_dim, hidden_size, rnn_units, dec_dropout,
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
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < torch.tensor(0.8)
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

    def greedy(self, src, start_token, stop_token, max_len=10):
        """[summary]

        Arguments:
            src {[type]} -- [description]
            start_token {[type]} -- [description]
            stop_token {[type]} -- [description]

        Keyword Arguments:
            max_len {int} -- [description] (default: {10})

        Returns:
            [type] -- [description]
        """
        batch_size = src.shape[1]
        hidden, cell = self.encoder(src)
        input = torch.tensor(start_token).unsqueeze(0).to(self.template_zeros.device)
        stop = False
        outputs = []
        while not stop:
            output, hidden, cell = self.decoder(input, hidden, cell)
            top1 = output.argmax(1)
            input = top1
            if top1.item() == stop_token or len(outputs) > max_len:
                stop = True
            outputs.append(top1.item())
        return outputs

    def greedy_batch(self, src, start_token, stop_token, max_len=10):
        """[summary]

        Arguments:
            src {[type]} -- [description]
            start_token {[type]} -- [description]
            stop_token {[type]} -- [description]

        Keyword Arguments:
            max_len {int} -- [description] (default: {10})

        Returns:
            [type] -- [description]
        """
        batch_size = src.shape[0]
        hidden, cell = self.encoder(src)
        input = torch.tensor(
            [start_token] * batch_size, device=self.template_zeros.device
        )
        stop = False
        outputs = []
        while not stop:
            output, hidden, cell = self.decoder(input, hidden, cell)
            top1 = output.argmax(1)
            topk = torch.topk(output, 5)
            input = top1
            if len(outputs) > max_len:
                stop = True
            outputs.append(top1.cpu().tolist())
        outputs = np.array(outputs).transpose().tolist()
        return outputs

    def beam(self, src, start_token, stop_token, beam_width=3, max_len=10):
        """[summary]

        Arguments:
            inp {[type]} -- [description]
            start_token {[type]} -- [description]
            stop_token {[type]} -- [description]

        Keyword Arguments:
            beam_width {int} -- [description] (default: {3})
            max_len {int} -- [description] (default: {10})
        """
        beam = Beam(beam_width)
        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        hidden, cell = self.encoder(src)
        start_token = (
            torch.tensor(start_token).unsqueeze(0).to(self.template_zeros.device)
        )
        stop_token = (
            torch.tensor(stop_token).unsqueeze(0).to(self.template_zeros.device)
        )
        beam.add(score=1.0, sequence=start_token, hidden=hidden, cell=cell)
        for _ in range(max_len):
            new_beam = Beam(beam_width)
            for score, seq, hid, cel in beam:
                if not torch.eq(seq[-1:], stop_token):
                    out, new_hid, new_cel = self.decoder(seq[-1:], hid, cel)
                    out = F.softmax(out,dim=1)
                    out = out.topk(beam_width)
                    for i in range(beam_width):
                        new_score = score * out.values[0][i]
                        new_beam.add(
                            score=new_score,
                            sequence=torch.cat([seq, out.indices[0][i].unsqueeze(0)]),
                            hidden=new_hid,
                            cell=new_cel,
                        )
                else:
                    new_beam.add(score=score, sequence=seq, hidden=hid, cell=cel)
            beam = new_beam
        opt = [(i.tolist(), seq.tolist()) for i, seq, _, _ in beam]
        return opt

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
