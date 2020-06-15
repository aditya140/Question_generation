import sys
import glob

sys.path.append("./src/")

import argparse
from params import SEQ2SEQ_PARAMS
from models.seq2seq import Seq2seq
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
import torch
import os
import optuna
import pandas as pd
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback
from test_metrics.test_model import Model_tester

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
    test_df = model.create_test_df()
    print("Generating Test Metrics")
    df, metrics = model.test_model(test_df)
    save_test_df(df, "seq2seq", version)
    save_metrics(metrics, "seq2seq", version)
    print(metrics)
    if hp.to_artifact:
        save_to_artifact("seq2seq", version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2seq Training")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--inp_vocab", type=int)
    parser.add_argument("--out_vocab", type=int)
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--NMT", action="store_true", help="Neural Machine Translation")
    parser.add_argument("--QGEN", action="store_true", help="Question Generation")
    parser.add_argument("--sample",action="store_true",help="Sample")
    parser.add_argument(
        "--to_artifact", action="store_true", help="Save to artifacts folder"
    )


    args = parser.parse_args()
    hp = SEQ2SEQ_PARAMS
    hp.epochs = arg_copy(args.epochs, hp.epochs)
    hp.input_vocab = arg_copy(args.inp_vocab, hp.input_vocab)
    hp.output_vocab = arg_copy(args.out_vocab, hp.output_vocab)
    hp.embedding_dim = arg_copy(args.emb_dim, hp.embedding_dim)
    hp.rnn_units = arg_copy(args.layers, hp.rnn_units)
    hp.hidden_size = arg_copy(args.hidden_dim, hp.hidden_size)
    hp.tokenizer = arg_copy(args.tokenizer, hp.tokenizer)
    hp.to_artifact = arg_copy(args.to_artifact, hp.to_artifact)
    hp.sample = arg_copy(args.sample, hp.sample)
    if args.QGEN:
        hp.squad = True
    if args.NMT:
        hp.squad = False
    train_model(hp)
