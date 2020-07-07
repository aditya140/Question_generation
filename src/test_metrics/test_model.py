import sys

sys.path.append("./src/")
import pandas as pd
import os
import torch
from utils import load_model, load_test_df, save_test_df, get_torch_device, save_metrics
from models.seq2seq import Seq2seq
from models.transformer import transformer
from inference.inference_helpers import GreedyDecoder, BeamDecoder
import argparse
import unicodedata
import re
from tqdm import tqdm
# from .bleu import get_bleu
from .pycocoevalcap.bleu import bleu
from .pycocoevalcap.meteor import meteor
from .pycocoevalcap.cider import cider
from .pycocoevalcap.rouge import rouge

tqdm.pandas()



def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
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


def load_model_from_version(name, version):
    state_dict, inpLang, optLang, hp = load_model(name=name, version=version)
    if name == "seq2seq":
        model = Seq2seq(**hp)
    elif name == "transformer":
        src_pad_idx = inpLang.word2idx[inpLang.special["pad_token"]]
        trg_pad_idx = optLang.word2idx[optLang.special["pad_token"]]
        model = transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, **hp)
    model.load_state_dict(state_dict)
    return model, inpLang, optLang, hp


class Model_tester:
    def __init__(self, model, inpLang, optLang, max_len):
        super().__init__()
        self.model = model
        self.inpLang = inpLang
        self.optLang = optLang
        self.mode = None
        self.max_len = max_len

    def set_inference_mode(self, mode, beam_size=None):
        self.mode = mode
        if self.mode == "greedy":
            self.decoder = GreedyDecoder(
                model=self.model, inpLang=self.inpLang, optLang=self.optLang
            )
            self.decoder.to(get_torch_device())
            self.generate = lambda x: self.decoder.greedy(
                x, max_len=self.max_len, to_string=True
            )[0]
        if self.mode == "beam" and beam_size != None:
            self.decoder = BeamDecoder(model=model, inpLang=inpLang, optLang=optLang)
            self.decoder.to(get_torch_device())
            self.generate = lambda x: self.decoder.beam(
                x, max_len=self.max_len, beam_width=self.beam_size, to_string=True
            )

    def predict(self, inp):
        return self.generate(inp)

    def generate_metrics(self, df):
        print(df.head())
        df["pred"] = df["input"].progress_apply(self.predict)
        print(df["pred"].head())
        df["pred"] = df["pred"].apply(lambda x: preprocess_sentence(x))
        print(df["pred"].head())
        df["output"] = df["output"].apply(lambda x: preprocess_sentence(x))
        
        def to_list(x):
            return [x]
        df["output"]=df["output"].apply(to_list)
        df["pred"]=df["pred"].apply(to_list)

        x=df.to_dict()
        
        gts=x["output"]
        res=x["pred"]
        
        bleu_scorer=bleu.Bleu()
        meteor_scorer=meteor.Meteor()
        cider_scorer=cider.Cider()
        rouge_scorer=rouge.Rouge()
        
        bleu_score=bleu_scorer.compute_score(gts,res)
        meteor_score=meteor_scorer.compute_score(gts,res)
        cider_score=cider_scorer.compute_score(gts,res)
        rouge_score=rouge_scorer.compute_score(gts,res)
        
        s=f"Bleu1 : {bleu_score[0][0]} \nBleu2 : {bleu_score[0][1]} \nBleu3 : {bleu_score[0][2]} \nBleu4 : {bleu_score[0][3]} \n"
        s+=f"Meteor : {meteor_score[0]}\n"
        s+=f"Cider : {cider_score[0]}\n"
        s+=f"ROUGE : {rouge_score[0]}\n"
        metrics = s
        return df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing framework for question generation"
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--version", type=int)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    model, inpLang, optLang, hp = load_model_from_version(args.model_name, args.version)
    if args.mode == None:
        args.mode = "greedy"
    test_df = load_test_df(args.model_name, args.version)
    tester = Model_tester(model, inpLang, optLang, max_len=100)
    tester.set_inference_mode(args.mode)
    df, metrics = tester.generate_metrics(test_df)
    save_test_df(df, args.model_name, args.version)
    save_metrics(metrics, args.model_name, args.version)
