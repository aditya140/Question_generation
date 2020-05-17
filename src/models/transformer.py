import sys
sys.path.append("./src/")
from torch import nn
import random
import torch
from torch.nn import Parameter
import numpy as np
import heapq
import torch.nn.functional as F
from inference.inference_helpers import Beam


class EncoderLayer(nn.Module):
    def __init__(self,hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 ):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self,src,src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])),requires_grad=False)
        
    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim = -1)                
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)  
        return x, attention



class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):        
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x



class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 max_length = 100):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  )
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = Parameter(torch.sqrt(torch.FloatTensor([hid_dim])),requires_grad=False)
        self.template=Parameter(torch.zeros(1))
        
    def forward(self, trg, enc_src, trg_mask, src_mask):  
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.template.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 ):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):  
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 max_length = 100):
        super().__init__()
        self.tok_embedding=nn.Embedding(input_dim,hid_dim)
        self.pos_embedding=nn.Embedding(max_length,hid_dim)

        self.layers=nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  ) for _ in range(n_layers)])
        self.dropout=nn.Dropout(dropout)
        self.scale=Parameter(torch.sqrt(torch.FloatTensor([hid_dim])),requires_grad=False)
        self.template=Parameter(torch.zeros(1))

    def forward(self,src,src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.template.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class transformer(nn.Module):
    def __init__(self, 
                 input_vocab,
                 output_vocab,
                 hidden_dim,
                 enc_layers,
                 enc_heads,
                 enc_pf_dim,
                 enc_dropout,
                 dec_layers,
                 dec_heads,
                 dec_pf_dim,
                 dec_dropout,
                 src_pad_idx, 
                 trg_pad_idx,
                 max_len,
                 **kwargs):
        super().__init__()
        self.encoder = Encoder(input_vocab,hidden_dim,enc_layers,enc_heads,enc_pf_dim,enc_dropout,max_length=max_len)
        self.decoder = Decoder(output_vocab,hidden_dim,dec_layers,dec_heads,dec_pf_dim,dec_dropout,max_length=max_len)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.template = Parameter(torch.zeros(1))
        
    def make_src_mask(self, src):        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.template.device)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2).to(self.template.device)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(self.template.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

    def initialize_weights(self):
        if hasattr(self, 'weight') and self.weight.dim() > 1:
            nn.init.xavier_uniform_(self.weight.data)
    
    def greedy(self, src, start_token, stop_token, max_len=100):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        trg_tensor = torch.tensor(start_token).unsqueeze(0).unsqueeze(0).to(self.template.device)
        stop = False
        while not stop:
            trg_mask = self.make_trg_mask(trg_tensor)
            output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            top1 = output.argmax(2)[:,-1]
            trg_tensor = torch.cat((trg_tensor,top1.unsqueeze(0)),dim=1)
            if top1.item() == stop_token or trg_tensor.shape[1] > max_len:
                stop = True
        return trg_tensor[0].tolist()







def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


