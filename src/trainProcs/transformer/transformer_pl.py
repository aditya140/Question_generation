import sys
import glob

sys.path.append("./src/")


import argparse

from params import TRANSFORMER_PARAMS
from models.transformer import transformer
import pytorch_lightning as pl
from utils import (
    save_model,
    get_torch_device,
    epoch_time,
    arg_copy,
    save_to_artifact,
    save_test_df,
    save_metrics,
)
from dataloader import SimpleDataloader
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger
import torch
import os
import pandas as pd
from test_metrics.test_model import Model_tester

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
        self.model = transformer(
            src_pad_idx=SRC_PAD_IDX, trg_pad_idx=TRG_PAD_IDX, **vars(hparams)
        )
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
        pred, _ = self.forward(src, trg[:, :-1])
        pred = pred.permute(0, 2, 1)
        loss = self.cross_entropy_loss(pred, trg[:, 1:])
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        return {"log": {"val_loss": loss}}

    def training_step(self, train_batch, idx):
        x, y, x_l = train_batch
        src = x
        trg = y
        pred, _ = self.forward(src, trg[:, :-1])
        pred = pred.permute(0, 2, 1)
        loss = self.cross_entropy_loss(pred, trg[:, 1:])
        logs = {"train_loss": loss}
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def cross_entropy_loss(self, input, target):
        return F.cross_entropy(input, target, ignore_index=TRG_PAD_IDX)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1),
            "name": "lr",
        }
        return [optimizer], [scheduler]

    def create_test_df(self):
        data = []
        for idx, batch in enumerate(self.data.get_test_dataloader()):
            inp, opt = batch
            for i in range(len(inp)):
                data_dict = {"input": inp[i], "output": opt[i]}
                data.append(data_dict)

        df = pd.DataFrame(data)
        return df

    def test_model(self, test_df):
        tester = Model_tester(
            self.model, self.data.inpLang, self.data.optLang, max_len=100
        )
        tester.set_inference_mode("greedy")
        return tester.generate_metrics(test_df)


def train_model(hp):
    logger = TensorBoardLogger("lightning_logs", name="transformer")
    lr_logger = LearningRateLogger()
    model = transformer_pl(hp)
    trainer = pl.Trainer(
        max_epochs=hp.epochs,
        gpus=1,
        logger=logger,
        auto_lr_find=False,
        callbacks=[lr_logger],
    )
    trainer.fit(model)
    to_save = {
        "model": model.model,
        "inpLang": model.data.inpLang,
        "optLang": model.data.optLang,
        "params": vars(hp),
        "version": version,
    }
    save_model(path=hp.save_path, name="transformer.pt", **to_save)
    test_df = model.create_test_df()
    print("Generating Test Metrics")
    df, metrics = model.test_model(test_df)
    save_test_df(df, "transformer", version)
    save_metrics(metrics, "transformer", version)
    print(metrics)
    if hp.to_artifact:
        save_to_artifact("transformer", version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Training")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--inp_vocab", type=int)
    parser.add_argument("--out_vocab", type=int)
    parser.add_argument("--att_heads", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--pf_dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--NMT", action="store_true", help="Neural Machine Translation")
    parser.add_argument("--QGEN", action="store_true", help="Question Generation")
    parser.add_argument(
        "--to_artifact", action="store_true", help="Save to artifacts folder"
    )
    parser.add_argument(
        "--auto_lr_find",
        action="store_true",
        help="Auto LR finder from pytorch lightning",
    )

    args = parser.parse_args()
    hp = TRANSFORMER_PARAMS
    hp.epochs = arg_copy(args.epochs, hp.epochs)
    hp.input_vocab = arg_copy(args.inp_vocab, hp.input_vocab)
    hp.output_vocab = arg_copy(args.out_vocab, hp.output_vocab)
    hp.dec_layers = arg_copy(args.layers, hp.dec_layers)
    hp.enc_layers = arg_copy(args.layers, hp.enc_layers)
    hp.enc_heads = arg_copy(args.att_heads, hp.enc_heads)
    hp.dec_heads = arg_copy(args.att_heads, hp.dec_heads)
    hp.hidden_dim = arg_copy(args.hidden_dim, hp.hidden_dim)
    hp.tokenizer = arg_copy(args.tokenizer, hp.tokenizer)
    hp.to_artifact = arg_copy(args.to_artifact, hp.to_artifact)
    hp.auto_lr_find = arg_copy(args.auto_lr_find, hp.auto_lr_find)
    if args.QGEN:
        hp.squad = True
    if args.NMT:
        hp.squad = False
    train_model(hp)
