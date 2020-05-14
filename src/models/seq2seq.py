from torch import nn
import random
import torch
from torch.nn import Parameter
import numpy as np


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
    def __init__(self, output_dim, emb_dim, hidden_size, rnn_units, dropout, embedding=None):
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
            input_vocab,
            enc_emb_dim,
            hidden_size,
            rnn_units,
            enc_dropout,
        )
        self.decoder = Decoder(
            output_vocab,
            dec_emb_dim,
            hidden_size,
            rnn_units,
            dec_dropout,
        )
        self.template_zeros=Parameter(torch.zeros(1),requires_grad=True)

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
    def beam_decode(self,):
        pass


    def decode(self, input, inpLang, optLang, max_len=10):
        """[summary]

        Arguments:
            input {[type]} -- [description]
            inpLang {[type]} -- [description]
            optLang {[type]} -- [description]

        Keyword Arguments:
            max_len {int} -- [description] (default: {10})

        Returns:
            [type] -- [description]
        """        
        stop_token = optLang.word2idx[optLang.special["eos_token"]]
        start_token = optLang.word2idx[optLang.special["init_token"]]
        src = (
            torch.tensor(inpLang.encode(input), device=self.device)
            .unsqueeze(1)
            .transpose(0, 1)
        )
        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        hidden, cell = self.encoder(src)
        input = torch.tensor(start_token, device=self.device).unsqueeze(0)
        stop = False
        outputs = []
        while not stop:
            output, hidden, cell = self.decoder(input, hidden, cell)
            top1 = output.argmax(1)
            topk = torch.topk(output, 5)
            input = top1
            if top1.item() == stop_token or len(outputs) > max_len:
                stop = True
            outputs.append(top1.item())
        return " ".join(optLang.decode(outputs))

    def batch_decode(self, input, inpLang, optLang, max_len=10):
        """[summary]

        Arguments:
            input {[type]} -- [description]
            inpLang {[type]} -- [description]
            optLang {[type]} -- [description]

        Keyword Arguments:
            max_len {int} -- [description] (default: {10})

        Returns:
            [type] -- [description]
        """        
        stop_token = optLang.word2idx[optLang.special["eos_token"]]
        start_token = optLang.word2idx[optLang.special["init_token"]]
        src = torch.tensor(inpLang.encode_batch(input), device=self.device)
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.output_dim
        hidden, cell = self.encoder(src)
        input = torch.tensor([start_token] * batch_size, device=self.device)
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
        return [" ".join(i) for i in optLang.decode_batch(outputs)]

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

