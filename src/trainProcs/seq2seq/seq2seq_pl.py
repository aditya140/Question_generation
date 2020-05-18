import sys
import glob
sys.path.append("./src/")




from params import SEQ2SEQ_PARAMS
from models.seq2seq import Seq2seq
import pytorch_lightning as pl
from utils import save_model, get_torch_device, epoch_time
from dataloader import SimpleDataloader
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback

hp = SEQ2SEQ_PARAMS
device = get_torch_device()

version = 0
versions = [int(i.split("/")[-2]) for i in glob.glob(hp.save_path + "/*/")]
if versions != []:
    version = max(versions) + 1

MODEL_DIR = hp.trial_path
PERCENT_VALID_EXAMPLES = 0.1


class Seq2seq_pl(pl.LightningModule):
    def __init__(self, hparams):
        super(Seq2seq_pl, self).__init__()
        self.hparams = hparams
        self.model = Seq2seq(**vars(hparams))

    def forward(self, input, target):
        return self.model.forward(input, target)

    def prepare_data(self):
        self.data = SimpleDataloader(**vars(self.hparams))

    def train_dataloader(self):
        return self.data.get_train_dataloader()

    def val_dataloader(self):
        return self.data.get_val_dataloader()

    def validation_step(self, val_batch, idx):
        x, y, x_l = val_batch
        src = x
        trg = y
        pred = self.forward(src, trg)
        pred = pred.permute(1, 2, 0)
        loss = self.cross_entropy_loss(pred, trg)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        return {"log": {"val_loss": loss}}

    def training_step(self, train_batch, idx):
        x, y, x_l = train_batch
        src = x
        trg = y
        pred = self.forward(src, trg)
        pred = pred.permute(1, 2, 0)
        loss = self.cross_entropy_loss(pred, trg)
        logs = {"train_loss": loss}
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def cross_entropy_loss(self, input, target):
        return F.cross_entropy(input, target, ignore_index=0)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def train_model():
    logger = TensorBoardLogger("lightning_logs", name="seq2seq")
    model = Seq2seq_pl(hp)
    trainer = pl.Trainer(max_epochs=hp.epochs, gpus=1, logger=logger, auto_lr_find=True)
    trainer.fit(model)
    to_save = {
            "model": model.model,
            "inpLang": model.data.inpLang,
            "optLang": model.data.optLang,
            "params": vars(hp),
            "version": version,
        }
    save_model(path=hp.save_path, name="seq2seq.pt", **to_save)


if __name__ == "__main__":
    train_model()
