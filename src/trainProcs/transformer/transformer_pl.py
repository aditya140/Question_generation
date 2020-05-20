import sys
import glob
sys.path.append("./src/")




from params import TRANSFORMER_PARAMS
from models.transformer import transformer
import pytorch_lightning as pl
from utils import save_model, epoch_time
from dataloader import SimpleDataloader
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger
import torch
import os


SRC_PAD_IDX = 0
TRG_PAD_IDX = 0

hp = TRANSFORMER_PARAMS

version = 0
versions = [int(i.split("/")[-2]) for i in glob.glob(hp.save_path + "/*/")]
if versions != []:
    version = max(versions) + 1


class transformer_pl(pl.LightningModule):
    def __init__(self, hparams):
        super(transformer_pl, self).__init__()
        self.hparams = hparams
        self.model = transformer(src_pad_idx=SRC_PAD_IDX,
                                trg_pad_idx=TRG_PAD_IDX,
                                **vars(hparams))
        self.model.initialize_weights()

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
        pred, _ = self.forward(src, trg[:,:-1])
        pred = pred.permute(0, 2, 1)
        loss = self.cross_entropy_loss(pred, trg[:,1:])
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        return {"log": {"val_loss": loss}}

    def training_step(self, train_batch, idx):
        x, y, x_l = train_batch
        src = x
        trg = y
        pred, _ = self.forward(src, trg[:,:-1])
        pred = pred.permute(0, 2, 1)
        loss = self.cross_entropy_loss(pred, trg[:,1:])
        logs = {"train_loss": loss}
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def cross_entropy_loss(self, input, target):
        return F.cross_entropy(input, target, ignore_index=TRG_PAD_IDX)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {'scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1),
                    'name': 'lr'}
        return [optimizer],[scheduler]


def train_model():
    logger = TensorBoardLogger("lightning_logs", name="transformer")
    lr_logger = LearningRateLogger()
    model = transformer_pl(hp)
    trainer = pl.Trainer(max_epochs=hp.epochs, gpus=1, logger=logger, auto_lr_find=False,callbacks=[lr_logger])
    trainer.fit(model)
    to_save = {
            "model": model.model,
            "inpLang": model.data.inpLang,
            "optLang": model.data.optLang,
            "params": vars(hp),
            "version": version,
        }
    save_model(path=hp.save_path, name="transformer.pt", **to_save)


if __name__ == "__main__":
    train_model()
