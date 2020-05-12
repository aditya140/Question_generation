from torch import nn
import random
import torch
from torch.nn import Parameter
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, embedding=None):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        if embedding != None:
            self.embedding.load_state_dict({"weight": embedding})
            self.embedding.weight.requires_grad = False
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = src.transpose(0, 1)
        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, embedding=None):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        if embedding != None:
            self.embedding.load_state_dict({"weight": embedding})
            self.embedding.weight.requires_grad = False
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell


class Seq2seq(torch.jit.ScriptModule):
    def __init__(
        self,
        INPUT_VOCAB,
        OUTPUT_VOCAB,
        ENC_EMB_DIM,
        DEC_EMB_DIM,
        HID_DIM,
        N_LAYERS,
        ENC_DROPOUT,
        DEC_DROPOUT,
        glove_inp=None,
        glove_opt=None,
    ):
        """[summary]

        Arguments:
            nn {[type]} -- [description]
            INPUT_VOCAB {[type]} -- [description]
            OUTPUT_VOCAB {[type]} -- [description]
            ENC_EMB_DIM {[type]} -- [description]
            DEC_EMB_DIM {[type]} -- [description]
            HID_DIM {[type]} -- [description]
            N_LAYERS {[type]} -- [description]
            ENC_DROPOUT {[type]} -- [description]
            DEC_DROPOUT {[type]} -- [description]

        Keyword Arguments:
            glove_inp {[type]} -- [description] (default: {None})
            glove_opt {[type]} -- [description] (default: {None})
        """        
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(
            INPUT_VOCAB,
            ENC_EMB_DIM,
            HID_DIM,
            N_LAYERS,
            ENC_DROPOUT,
            embedding=glove_inp,
        )
        self.decoder = Decoder(
            OUTPUT_VOCAB,
            DEC_EMB_DIM,
            HID_DIM,
            N_LAYERS,
            DEC_DROPOUT,
            embedding=glove_opt,
        )
        self.template_zeros=Parameter(torch.zeros(1),requires_grad=True)

    @torch.jit.script_method
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
