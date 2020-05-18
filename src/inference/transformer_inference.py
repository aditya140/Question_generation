import sys

sys.path.append("./src/")
import torch
import time
from models.transformer import transformer
from utils import load_model,epoch_time
from inference_helpers import GreedyDecoder, BeamDecoder


state_dict, inpLang, optLang, hp = load_model(name="transformer", version=3)
src_pad_idx=inpLang.word2idx[inpLang.special["pad_token"]]
trg_pad_idx=optLang.word2idx[optLang.special["pad_token"]]

model = transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,**hp)
model.load_state_dict(state_dict)
inference = GreedyDecoder(model=model, inpLang=inpLang, optLang=optLang)
inference.to(torch.device("cuda"))
print(" ".join(inference.greedy("john used to live in canada",max_len=50)))



state_dict, inpLang, optLang, hp = load_model(name="transformer", version=3)
src_pad_idx=inpLang.word2idx[inpLang.special["pad_token"]]
trg_pad_idx=optLang.word2idx[optLang.special["pad_token"]]

model = transformer(src_pad_idx=src_pad_idx,trg_pad_idx=trg_pad_idx,**hp)
model.load_state_dict(state_dict)
inference = BeamDecoder(model=model, inpLang=inpLang, optLang=optLang)
inference.to(torch.device("cuda"))
outputs = inference.beam("where dows the ",max_len=50,beam_width=3)
for i in outputs:
    print(" ".join(i[1]),f" Score : {i[0]}",sep="\t|")
