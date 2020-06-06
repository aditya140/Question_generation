import warnings

warnings.filterwarnings("ignore")
from models.transformer import transformer
from params import TRANSFORMER_PARAMS
import torch


model = transformer(**vars(TRANSFORMER_PARAMS)).cuda()
print(model)
print(model(torch.randint(10, (1, 100)).cuda(), torch.randint(10, (1, 100)).cuda()))
