import sys

sys.path.append("./src/")
import pandas as pd
import os
import torch
from utils import load_model,load_test_df
from models.seq2seq import Seq2seq
from models.transformer import transformer
from inference.inference_helpers import GreedyDecoder, BeamDecoder
import argparse
import unicodedata
import re
from tqdm import tqdm
import bleu
tqdm.pandas()

def bleu_metric(ref,candidate,wt=1):
    weight=[1/wt]*(wt)+[0]*(4-wt)
    score = bleu.get_bleu( candidate.split(' '), ref.split(" "))
    print(score)
    return score
 
def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    return w



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
            self.generate=lambda x:self.decoder.greedy(x,max_len=self.max_len,to_string=True)
        if self.mode=="beam" and beam_size!=None:
            self.decoder=BeamDecoder(model=model,inpLang=inpLang,optLang=optLang)
            self.generate=lambda x:self.decoder.beam(x,max_len=self.max_len,beam_width=self.beam_size, to_string=True)
    def predict(self,inp):
        return self.generate(inp)

    def generate_metrics(self,df):
        df["pred"]=df['input'].progress_apply(self.predict)
        df["pred"]=df["pred"].apply(lambda x: preprocess_sentence(x))
        df["output"]=df["output"].apply(lambda x: preprocess_sentence(x))
        df["bleu"]=df.apply(lambda x:bleu_metric(x["output"],x["pred"],wt=1) ,axis=1)
        print(df.head())

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
    parser.add_argument("--version",type=int)
    parser.add_argument("--mode",type=str)
    args=parser.parse_args()
    
    model,inpLang,optLang,hp = load_model_from_version(args.model_name,args.version)
    if args.mode==None:
        args.mode='greedy'
    test_df=load_test_df(args.model_name,args.version)
    tester=Model_tester(model,inpLang,optLang,max_len=100)
    tester.set_inference_mode(args.mode)
    tester.generate_metrics(test_df.head())
