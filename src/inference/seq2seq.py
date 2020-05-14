import sys
sys.path.append("./src/")
from models.seq2seq import Seq2seq
import torch
from utils import load_model


state_dict,inpLang,optLang,hp=load_model(name="seq2seq",version=2)
model=Seq2seq(**hp)
model.load_state_dict(state_dict)
print(model)
print(model.decode("Hello how are you",inpLang,optLang))