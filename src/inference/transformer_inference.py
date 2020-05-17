import sys

sys.path.append("./src/")
import torch
import time
from models.transformer import transformer
from utils import load_model,epoch_time
from inference_helpers import GreedyDecoder, BeamDecoder


state_dict, inpLang, optLang, hp = load_model(name="transformer", version=1)
src_pad_idx=inpLang.word2idx[inpLang.special["pad_token"]]
trg_pad_idx=optLang.word2idx[optLang.special["pad_token"]]

model = transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,**hp)
model.load_state_dict(state_dict)
inference = GreedyDecoder(model=model, inpLang=inpLang, optLang=optLang)
inference.to(torch.device("cuda"))
print(" ".join(inference.greedy("one ticket to raleigh.",max_len=50)))


# state_dict, inpLang, optLang, hp = load_model(name="transformer", version=0)
# src_pad_idx=inpLang.word2idx[inpLang.special["pad_token"]]
# trg_pad_idx=optLang.word2idx[optLang.special["pad_token"]]

# model = transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,**hp)
# model.load_state_dict(state_dict)
# inference = BeamDecoder(model=model, inpLang=inpLang, optLang=optLang)

# inference.to(torch.device("cuda"))


# print(inference.beam("my name is john", beam_width=1))