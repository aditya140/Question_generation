from models.seq2seq import Seq2seq
import pytorch_lightning as pl
import argparse
from dataPrep.data_loader import QGenDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

params={

    "lr":1e-3,
    "input_vocab":10000,
    "output_vocab":10000,
    "embedding_dim":300,
    "rnn_units":1,
    "hidden_size":128,
    "batch_size":64,
    "teacher_forcing":0.8,
    "bidirectional":False,
    "squad":True,
    "tokenizer":"spacy",
    "max_len":60,
    "sample":False,
}

hparams=argparse.Namespace(**params)

class Seq2seq_pl(pl.LightningModule):
    def __init__(self,hparams):
        super(Seq2seq_pl,self).__init__()
        self.hparams=hparams
        self.model=Seq2seq(input_vocab=hparams.input_vocab,
                        output_vocab=hparams.input_vocab,
                        embedding_dim=hparams.embedding_dim,
                        rnn_units=hparams.rnn_units,
                        hidden_size=hparams.hidden_size,
                        teacher_forcing=hparams.teacher_forcing,
                        bidirectional=hparams.bidirectional)


    def forward(self,input,target,teacher_forcing):
        return self.model.forward(input,target,teacher_forcing)

    def prepare_data(self):
        QGen=QGenDataset()

        (self.train_data_set,
        self.val_data_set,
        self.test_data_set,
        self.inpLang,self.optLang)=QGen.getData(input_vocab=self.hparams.input_vocab,
                                    output_vocab=self.hparams.output_vocab,
                                    max_len=self.hparams.max_len,
                                    tokenizer=self.hparams.tokenizer,
                                    sample=self.hparams.sample,
                                    batch_size=self.hparams.batch_size,
                                    squad=self.hparams.squad)


    def train_dataloader(self):
        return DataLoader(self.train_data_set,batch_size=self.hparams.batch_size,num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_data_set,batch_size=self.hparams.batch_size,num_workers=10)
    
    def validation_step(self, val_batch , idx):
        x,y,x_l = val_batch
        src=x
        trg=y
        output=self.forward(src,trg,teacher_forcing=False)
        output_dim = output.shape[-1]
        output = output[1:,:].contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = self.cross_entropy_loss(output,trg)
        return {'val_loss': loss}

    def training_step(self,train_batch,idx):
        x,y,x_l= train_batch
        src=x
        trg=y
        output=self.forward(src,trg,teacher_forcing=True)
        output_dim = output.shape[-1]
        output = output[1:,:].contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = self.cross_entropy_loss(output,trg)
        logs = {'train_loss': loss}
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def cross_entropy_loss(self, input, target):
        return F.cross_entropy(input,target)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer 


def main():
    model=Seq2seq_pl(hparams)
    model.prepare_data()
    trainer=pl.Trainer(max_epochs=2,gpus=1,)
    trainer.fit(model)

if __name__ == '__main__':
    main()
    