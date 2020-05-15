import torch.nn as nn
import torch
from torch.nn import Parameter

class Inference(nn.Module):
    def __init__(self,model,inpLang,optLang):
        super().__init__()
        self.model=model
        self.inpLang=inpLang
        self.optLang=optLang
        self.decode_start=self.optLang.word2idx[self.optLang.special["init_token"]]
        self.decode_stop=self.optLang.word2idx[self.optLang.special["eos_token"]]
        self.template_tensor=Parameter(torch.tensor(0),requires_grad=False)

    def encode(self,inp):
        assert isinstance(inp,str)
        return self.inpLang.encode(inp)

    def encode_batch(self,inp):
        assert bool(inp) and isinstance(inp, list) and all(isinstance(elem, str) for elem in inp)
        return self.inpLang.encode_batch(inp)

    def decode_batch(self,seq):
        return self.optLang.decode_batch(seq)

    def decode(self,seq):
        return self.optLang.decode(seq)

    

class GreedyDecoder(Inference):
    def __init__(self,model,inpLang,optLang):
        super(GreedyDecoder,self).__init__(model,inpLang,optLang)

    def greedy_str(self,inp,max_len=10):
        src = (
            torch.tensor(self.encode(inp))
            .unsqueeze(1)
            .transpose(0, 1)
        ).to(self.template_tensor.device)
        opt=self.model.greedy(src,self.decode_start,self.decode_stop,max_len=max_len)
        opt=self.decode(opt)
        return opt

    def greedy_batch(self,inp,max_len=10):
        src = (
            torch.tensor(self.encode_batch(inp))
        ).to(self.template_tensor.device)
        opt=self.model.greedy_batch(src,self.decode_start,self.decode_stop,max_len=max_len)
        opt=self.decode_batch(opt)
        return opt

    def greedy(self,inp,max_len=10):
        if isinstance(inp,str):
            return self.greedy_str(inp)
        if bool(inp) and isinstance(inp, list) and all(isinstance(elem, str) for elem in inp):
            return self.greedy_batch(inp)
        
