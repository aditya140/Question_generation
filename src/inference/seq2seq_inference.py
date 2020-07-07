import sys

sys.path.append("./src/")
import torch

from models.seq2seq import Seq2seq
from utils import load_model
from inference_helpers import GreedyDecoder, BeamDecoder


state_dict, inpLang, optLang, hp = load_model(name="seq2seq", version=3)
model = Seq2seq(**hp)
model.load_state_dict(state_dict)
inference = GreedyDecoder(model=model, inpLang=inpLang, optLang=optLang)

inference.to(torch.device("cuda"))

print(inference.greedy("my name is john")[0])


state_dict, inpLang, optLang, hp = load_model(name="seq2seq", version=3)
model = Seq2seq(**hp)
model.load_state_dict(state_dict)
inference = BeamDecoder(model=model, inpLang=inpLang, optLang=optLang)

inference.to(torch.device("cuda"))


print(inference.beam("my name is john", beam_width=1))
