import warnings
warnings.filterwarnings("ignore")
from models.seq2seq import Beam,Seq2seq
from params import SEQ2SEQ_PARAMS
import torch
a=torch.randint(10,(1,4))

model=Seq2seq(**vars(SEQ2SEQ_PARAMS))
a=model.beam(a,0,1)
print(a)
