import torch
import transformers
import torch.nn as nn
import torch.nn
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM



class bert_qg(nn.Module):
    def __init__(self,hp):
        self.bert=BertModel.from_pretrained('bert-base-uncased')
        self.fc = 
