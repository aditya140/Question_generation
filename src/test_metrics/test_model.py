import sys

sys.path.append("./src/")
import pandas as pd
import os
import torch
from nltk.translate.bleu_score import corpus_bleu
from utils import load_model,load_test_df
from models.seq2seq import Seq2seq
from models.transformer import transformer
from inference.inference_helpers import GreedyDecoder, BeamDecoder
import argparse

def bleu_metric(inp,opt):
    score = corpus_bleu([inp], [opt])
    return score




class Model_tester:
    def __init__(self,model,inpLang,optLang,max_len):
        super().__init__()
        self.model=model
        self.inpLang=inpLang
        self.optLang=optLang
        self.mode=None
        self.max_len=max_len

    def set_inference_mode(self,mode,beam_size=None):
        self.mode=mode
        if self.mode=="greedy":
            self.decoder=GreedyDecoder(model=self.model, inpLang=self.inpLang, optLang=self.optLang)
            self.generate=lambda x:self.decoder.greedy(x,max_len=self.max_len)
        if self.mode=="beam" and beam_size!=None:
            self.decoder=BeamDecoder(model=model,inpLang=inpLang,optLang=optLang)
            self.generate=lambda x:self.decoder.beam(x,max_len=self.max_len,beam_width=self.beam_size)
    def predict(self,inp):
        return self.generate(inp)

    def generate_metrics(df):
        pass


def load_model_from_version(name,version):
    state_dict, inpLang, optLang, hp = load_model(name=name, version=version)
    if name=="seq2seq":
        model = Seq2seq(**hp)
    elif name=="transformer":
        src_pad_idx=inpLang.word2idx[inpLang.special["pad_token"]]
        trg_pad_idx=optLang.word2idx[optLang.special["pad_token"]]
        model = transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,**hp)
    model.load_state_dict(state_dict)
    return model,inpLang,optLang,hp



if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Testing framework for question generation")
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--version",type=str)
    args=parser.parse_args()
    
    model,inpLang,optLang,hp = load_model_from_version(args.model_name,args.version)
    test_df=load_test_df(args.model_name,args.version)
    