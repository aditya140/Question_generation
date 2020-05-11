import torch
from torch import nn
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
import os
from params import *
from data_loader import *
from languageField import *
from sklearn.model_selection import train_test_split
import pickle 
from utils import *
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from torch import optim

class Encoder(pl.LightningModule):
    def __init__(self,input_dim,emb_dim,hid_dim,n_layers,dropout,embedding=None):
        super(Encoder,self).__init__()
        self.hid_dim=hid_dim
        self.n_layers=n_layers
        self.embedding=nn.Embedding(input_dim,emb_dim)
        if embedding!=None:
            self.embedding.load_state_dict({'weight': embedding})
            self.embedding.weight.requires_grad = False
        self.rnn=nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout)
        self.dropout=nn.Dropout(dropout)
    def forward(self,src):
        src=src.transpose(0,1)
        embedded=self.dropout(self.embedding(src))
        output,(hidden,cell)=self.rnn(embedded)
        return hidden,cell

class Decoder(pl.LightningModule):
    def __init__(self,output_dim,emb_dim,hid_dim,n_layers,dropout,embedding=None):
        super(Decoder,self).__init__()
        self.output_dim=output_dim
        self.hid_dim=hid_dim
        self.n_layers=n_layers
        self.embedding=nn.Embedding(output_dim,emb_dim)
        if embedding!=None:
            self.embedding.load_state_dict({'weight': embedding})
            self.embedding.weight.requires_grad = False
        self.rnn=nn.LSTM(emb_dim,hid_dim,n_layers,dropout=dropout)
        self.fc=nn.Linear(hid_dim,output_dim)
        # self.softmax=nn.Softmax(dim=1)
        self.dropout=nn.Dropout(dropout)
    def forward(self,input,hidden,cell):
        input=input.unsqueeze(0)
        embedded=self.dropout(self.embedding(input))
        output,(hidden,cell)=self.rnn(embedded,(hidden,cell))
        prediction=self.fc(output.squeeze(0))
        # prediction=self.softmax(prediction)
        return prediction,hidden,cell


class Seq2seq(pl.LightningModule):
    class MyData(Dataset):
        def __init__(self, X, y):
            self.data = X
            self.target = y
            # TODO: convert this into torch code is possible
            self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]

        def __getitem__(self, index):
            x = self.data[index]
            y = self.target[index]
            x_len = self.length[index]
            return x,y,x_len

        def __len__(self):
            return len(self.data)
    class TestData(Dataset):
        def __init__(self, X, y):
            self.data = X
            self.target = y

        def __getitem__(self, index):
            x = self.data[index]
            y = self.target[index]
            return x,y

        def __len__(self):
            return len(self.data)
    
    def __init__(self,INPUT_VOCAB,ENC_EMB_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,
                OUTPUT_VOCAB,DEC_EMB_DIM,DEC_DROPOUT,
                enc_embedding=None,
                dec_embedding=None):
        super(Seq2seq,self).__init__()
        self.INPUT_VOCAB=INPUT_VOCAB
        self.OUTPUT_VOCAB=OUTPUT_VOCAB
        self.encoder=Encoder(INPUT_VOCAB,ENC_EMB_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,embedding=enc_embedding)
        self.decoder=Decoder(OUTPUT_VOCAB,DEC_EMB_DIM,HID_DIM,N_LAYERS,DEC_DROPOUT,embedding=dec_embedding)
        self.device=device
        self.loss_func=None
    def forward(self,src,trg,teacher_forcing_ratio=0.5):
        batch_size=trg.shape[0]
        trg_len=trg.shape[1]
        trg_vocab_size=self.decoder.output_dim
        outputs=torch.zeros(trg_len,batch_size,trg_vocab_size).to(self.device)
        hidden,cell=self.encoder(src)
        input=trg[:,0]
        for t in range(1,trg_len):
            output,hidden,cell=self.decoder(input,hidden,cell)
            outputs[t]=output
            teacher_force=random.random()<teacher_forcing_ratio
            top1=output.argmax(1)
            input = trg[:,t] if teacher_force else top1
        return outputs
    def decode(self,input,max_len=10):
        stop_token=self.optLang.word2idx[self.optLang.special["eos_token"]]
        start_token=self.optLang.word2idx[self.optLang.special["init_token"]]
        src=torch.tensor(self.inpLang.encode(input),device=self.device).unsqueeze(1).transpose(0,1)
        batch_size=src.shape[1]
        trg_vocab_size=self.decoder.output_dim
        hidden,cell=self.encoder(src)
        input=torch.tensor(start_token,device=self.device).unsqueeze(0)
        stop=False
        outputs=[]
        while not stop:
            output,hidden,cell=self.decoder(input,hidden,cell)
            top1=output.argmax(1)
            topk=torch.topk(output,5)
            input=top1
            if top1.item()==stop_token or len(outputs)>max_len:
                stop=True
            outputs.append(top1.item())
        return " ".join(self.optLang.decode(outputs))

        
    def batch_decode(self,input,max_len=10):
        stop_token=self.optLang.word2idx[self.optLang.special["eos_token"]]
        start_token=self.optLang.word2idx[self.optLang.special["init_token"]]
        src=torch.tensor(self.inpLang.encode_batch(input),device=self.device)
        batch_size=src.shape[0]
        trg_vocab_size=self.decoder.output_dim
        hidden,cell=self.encoder(src)
        input=torch.tensor([start_token]*batch_size,device=self.device)
        stop=False
        outputs=[]
        while not stop:
            output,hidden,cell=self.decoder(input,hidden,cell)
            top1=output.argmax(1)
            topk=torch.topk(output,5)
            input=top1
            if len(outputs)>max_len:
                stop=True
            outputs.append(top1.cpu().tolist())
        outputs=(np.array(outputs).transpose().tolist())
        return [" ".join(i) for i in self.optLang.decode_batch(outputs)]

    def prepare_data(self):
        qg=QGenDataset(test_nmt=testNMT)
        if testNMT:
            input_,output_=qg.get_NMT(sample=SAMPLE)
        else:
            input_,output_=qg.get_AQ(sample=SAMPLE,max_len=MAX_LEN)
        train_set_input,test_set_input,train_set_output,test_set_output=train_test_split(input_,output_,test_size=0.2)
        input_train,input_test,output_train,output_test=train_test_split(train_set_input,train_set_output,test_size=0.1)

        self.inpLang=LanguageIndex(input_train,vocab_size=self.INPUT_VOCAB,max_len=MAX_LEN,tokenizer=TOKENIZER)
        self.optLang=LanguageIndex(output_train,vocab_size=self.OUTPUT_VOCAB,max_len=MAX_LEN,tokenizer=TOKENIZER)

        if USE_PRETRAINED:
            with open(GLOVE_PATH,'rb') as f:
                glove=pickle.load(f)

        if USE_PRETRAINED:
            glove_opt=load_glove_embedding(self.optLang,glove)
            glove_inp=load_glove_embedding(self.inpLang,glove)
            OUTPUT_VOCAB=glove_opt.shape[0]
            INPUT_VOCAB=glove_inp.shape[0]
        else:
            glove_inp=None
            glove_opt=None
        input_train_tokens=self.inpLang.encode_batch(input_train)
        input_test_tokens=self.inpLang.encode_batch(input_test)
        ouptut_train_tokens=self.optLang.encode_batch(output_train)
        output_test_tokens=self.optLang.encode_batch(output_test)
        TEST_BATCH_SIZE=20
        self.final_test_dataset=self.TestData(test_set_input,test_set_output)
        # final_dataloader=DataLoader(self.final_test_dataset,batch_size=TEST_BATCH_SIZE)  
        self.train_dataset = self.MyData(input_train_tokens,ouptut_train_tokens)
        self.test_dataset = self.MyData(input_test_tokens, output_test_tokens)
        # train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
        #                     drop_last=True,
        #                     shuffle=True)
        # test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, 
        #                     drop_last=True,
        #                     shuffle=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = BATCH_SIZE, 
                            drop_last=True,
                            shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = BATCH_SIZE, 
                            drop_last=True,
                            shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(self.final_test_dataset,batch_size=TEST_BATCH_SIZE) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer 
    
    def cross_entropy_loss(self, input, target):
        if self.loss_func==None:
            self.loss_func=nn.CrossEntropyLoss(ignore_index = self.optLang.word2idx[self.optLang.special["pad_token"]])
        return self.loss_func(input, target)
        
    def training_step(self, train_batch, batch_idx):
        x,y,_= train_batch
        src=x.to(device)
        trg=y.to(device)
        output=self.forward(src,trg)
        trg=trg.transpose(0,1)
        output_dim=output.shape[-1]
        output=output[1:].view(-1,output_dim)
        trg=trg[1:].contiguous().view(-1)
        loss = self.cross_entropy_loss(output,trg)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x,y,_= val_batch
        src=x.to(device)
        trg=y.to(device)
        output=self.forward(src,trg)
        trg=trg.transpose(0,1)
        output_dim=output.shape[-1]
        output=output[1:].view(-1,output_dim)
        trg=trg[1:].contiguous().view(-1)
        loss = self.cross_entropy_loss(output,trg)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

model = Seq2seq(INPUT_VOCAB,EMBEDDING_DIM,HIDDEN_SIZE,LAYERS,0.3,OUTPUT_VOCAB,EMBEDDING_DIM,0.3,enc_embedding=None,dec_embedding=None)
trainer = pl.Trainer(gpus=1)

trainer.fit(model)

