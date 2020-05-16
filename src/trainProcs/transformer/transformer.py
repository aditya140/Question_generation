import sys

sys.path.append("./src/")

from dataloader import SimpleDataloader
from params import SEQ2SEQ_PARAMS
from models.transformer import transformer, count_parameters
from utils import save_model, get_torch_device, epoch_time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import sys
import time
import glob

hp = TRANSFORMER_PARAMS
device = get_torch_device()
version = 0
versions = [int(i.split("/")[-2]) for i in glob.glob(hp.save_path + "/*/")]
if versions != []:
    version = max(versions) + 1


"""
Create dataloaders
"""
data = SimpleDataloader(**vars(hp))
train_dataloader = data.get_train_dataloader()
val_dataloader = data.get_val_dataloader()
test_dataloader = data.get_test_dataloader()


"""
Create model
"""
model = transformer(**vars(hp),src_pad_idx=inpLang.word2idx[inpLang.special["pad_token"]], 
                 trg_pad_idx=optLang.word2idx[optLang.special["pad_token"]])
model.init_weights()
print(f"The model has {count_parameters(model):,} trainable parameters")
print(model)
model.to(device)


"""
Loss Function and optimizers
"""
CCE = lambda x, y: F.cross_entropy(x, y, ignore_index=0)
adamW = optim.AdamW(model.parameters(), lr=1e-3)


"""
Train Function
"""


def train(model, dataloader, optimizer, loss_fn, device, print_freq=100):
    model.train()
    losses = []
    print(f"\tTraining for {len(dataloader)} iter")
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        src, trg, src_len = batch
        src = src.to(device)
        trg = trg.to(device)
        pred = model(src, trg[:,:-1])
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, trg)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if idx % print_freq == 0:
            print(f"\t\tIter:{idx}", f"loss:{loss.item()}", sep="----")
    return losses


"""
Evaluate Function
"""


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    for idx, batch in enumerate(dataloader):
        src, trg, src_len = batch
        src = src.to(device)
        trg = trg.to(device)
        with torch.no_grad():
            pred = model(src, trg[:,:-1])
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, trg)
        losses.append(loss.item())
    return np.mean(losses)


"""
Main Loop
"""
best_model_loss = float("inf")
val_loss = evaluate(model, val_dataloader, CCE, device)
print("Starting Loss: ", val_loss)
for ep in range(hp.epochs):
    print("-" * 10)
    print(f"Epoch : {ep}")
    st_time = time.time()
    train_loss = train(model, train_dataloader, adamW, CCE, device)
    val_loss = evaluate(model, val_dataloader, CCE, device)
    if val_loss < best_model_loss:
        to_save = {
            "model": model,
            "inpLang": data.inpLang,
            "optLang": data.optLang,
            "params": vars(hp),
            "version": version,
        }
        save_model(path=hp.save_path, name="transformer.pt", **to_save)
        best_model_loss = val_loss
    e_time = time.time()
    epoch_mins, epoch_secs = epoch_time(st_time, e_time)
    print(
        f"\tTraining Loss : {np.mean(train_loss)}",
        f"Val Perplexity : {math.exp(np.mean(train_loss))}",
        sep="\t|\t",
    )
    print(
        f"\tVal Loss      : {val_loss}",
        f"Val Perplexity : {math.exp(val_loss)}",
        sep="\t|\t",
    )
    print(f"\tTime per epoch: {epoch_mins}m {epoch_secs}s")



