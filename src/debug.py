from params import SEQ2SEQ_PARAMS
from models.seq2seq import Seq2seq
import pytorch_lightning as pl
from utils import save_model,get_torch_device,epoch_time
from dataloader import SimpleDataloader
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger('lightning_logs', name="mnist")
import torch

hp = SEQ2SEQ_PARAMS
device=get_torch_device()



class Seq2seq_pl(pl.LightningModule):
    def __init__(self,hparams):
        super(Seq2seq_pl,self).__init__()
        self.hparams=hparams
        self.model=Seq2seq(**hparams)


    def forward(self,input,target):
        return self.model.forward(input,target)

    def prepare_data(self):
        data = SimpleDataloader(**vars(self.hp))


    def train_dataloader(self):
        return self.data.get_train_dataloader()

    def val_dataloader(self):
        return self.data.get_val_dataloader()
    
    def validation_step(self, val_batch , idx):
        x,y,x_l = val_batch
        src=x
        trg=y
        output=self.forward(src,trg)
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
        return F.cross_entropy(input,target,ignore_index=0)

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
    