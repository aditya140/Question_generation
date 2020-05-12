from models.seq2seq import Seq2seq
import pytorch_lightning as pl
import argparse
from dataPrep.data_loader import QGenDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger('lightning_logs', name="mnist")
import torch

from torch.utils.tensorboard import SummaryWriter

params={

    "lr":1e-3,
    "input_vocab":10000,
    "output_vocab":10000,
    "embedding_dim":300,
    "rnn_units":1,
    "hidden_size":128,
    "batch_size":64,
    "squad":True,
    "tokenizer":"spacy",
    "max_len":60,
    "sample":False,
    "dropout":0.3,
}

hparams=argparse.Namespace(**params)

class Seq2seq_pl(pl.LightningModule):
    def __init__(self,hparams):
        super(Seq2seq_pl,self).__init__()
        self.hparams=hparams
        self.model=Seq2seq(INPUT_VOCAB=hparams.input_vocab,
        OUTPUT_VOCAB=hparams.output_vocab,
        ENC_EMB_DIM=hparams.embedding_dim,
        DEC_EMB_DIM=hparams.embedding_dim,
        HID_DIM=hparams.hidden_size,
        N_LAYERS=hparams.rnn_units,
        ENC_DROPOUT=hparams.dropout,
        DEC_DROPOUT=hparams.dropout,
        )


    def forward(self,input,target):
        return self.model.forward(input,target)

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
    # writer = SummaryWriter("tb_logs/mnist/")
    test_inp=next(iter(model.val_dataloader()))
    print(test_inp[0].shape,test_inp[1].shape)
    model.eval()
    print(model(test_inp[0],test_inp[1]))
    traced_script_module = torch.jit.trace(model, (test_inp[0],test_inp[1]))
    # writer.add_graph(model,(test_inp[0],test_inp[1]),verbose=True)
    # writer.close()

    trainer=pl.Trainer(max_epochs=2,gpus=1,logger=logger)
    trainer.fit(model)

if __name__ == '__main__':
    main()
    