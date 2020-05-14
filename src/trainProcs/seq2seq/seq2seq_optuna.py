
from params import SEQ2SEQ_PARAMS
import sys
sys.path.append("./src/")

from models.seq2seq import Seq2seq
import pytorch_lightning as pl
from utils import save_model,get_torch_device,epoch_time
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
device=get_torch_device()

MODEL_DIR = hp.trial_path
PERCENT_VALID_EXAMPLES = 0.1

class Seq2seq_optuna(pl.LightningModule):
    def __init__(self, trial, hparams):
        super(Seq2seq_optuna,self).__init__()
        self.hparams=hparams
        dropout = trial.suggest_uniform("dropout", 0.2, 0.5)
        emb_dim = trial.suggest_int("emb_dim",100,200)
        hid_dim = trial.suggest_int("hid_dim",50,100)
        self.model=Seq2seq(input_vocab=hparams.input_vocab,
                        output_vocab=hparams.output_vocab,
                        enc_emb_dim=emb_dim,
                        dec_emb_dim=emb_dim,
                        hidden_size=hid_dim,
                        rnn_units=hparams.rnn_units,
                        enc_dropout=dropout,
                        dec_dropout=dropout,)
    def forward(self,input,target):
        return self.model.forward(input,target)

    def prepare_data(self):
        self.data = SimpleDataloader(**vars(self.hparams))

    def train_dataloader(self):
        return self.data.get_train_dataloader()

    def val_dataloader(self):
        return self.data.get_val_dataloader()
    
    def validation_step(self, val_batch , idx):
        x,y,x_l = val_batch
        src=x
        trg=y
        pred=self.forward(src,trg)
        pred=pred.permute(1,2,0)
        loss = self.cross_entropy_loss(pred,trg)
        return {'val_loss': loss}

    def training_step(self,train_batch,idx):
        x,y,x_l= train_batch
        src=x
        trg=y
        pred=self.forward(src,trg)
        pred=pred.permute(1,2,0)
        loss = self.cross_entropy_loss(pred,trg)
        logs = {'train_loss': loss}
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        loss = sum(x["val_loss"] for x in outputs) / len(outputs)
        return {"log": {"val_loss": loss}}

    def cross_entropy_loss(self, input, target):
        return F.cross_entropy(input,target,ignore_index=0)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer 


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
    # Filenames for each trial must be made unique in order to access each checkpoint.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_loss"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=False,
        val_percent_check=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=checkpoint_callback,
        max_epochs=hp.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss"),
    )

    model = Seq2seq_optuna(trial,hp)
    trainer.fit(model)

    return metrics_callback.metrics[-1]["val_loss"]


def hparam_tuning():
    pruner = optuna.pruners.MedianPruner() if hp.prune else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    # train_model()
    hparam_tuning()
    