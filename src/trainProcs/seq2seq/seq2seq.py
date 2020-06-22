import sys

sys.path.append("./src/")

from dataloader import SimpleDataloader
from params import SEQ2SEQ_PARAMS
from models.seq2seq import Seq2seq, count_parameters, Encoder, Decoder
from utils import (
    save_model,
    get_torch_device,
    epoch_time,
    arg_copy,
    save_to_artifact,
    save_test_df,
    save_metrics,
)
from test_metrics.test_model import Model_tester

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import sys
import time
import glob
import argparse
import pandas as pd


def main(hp):
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
    Embedding weight matrix if needed
    """
    inp_emb, opt_emb = data.get_weight_matrix()
    """
    Create model
    """
    model = Seq2seq(**vars(hp))
    model.init_weights()
    model.create_embeddings(inp_emb,opt_emb)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    print(model)
    model.to(device)

    """
    Loss Function and optimizers
    """
    CCE = lambda x, y: F.cross_entropy(x, y, ignore_index=0)
    adamW = optim.AdamW(model.parameters(), lr=hp.lr)

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
            pred = model(src, trg)
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
                pred = model(src, trg)
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, trg)
            losses.append(loss.item())
        return np.mean(losses)

    """
    Generate Test Dataframe
    """

    def create_test_df(dataloader):
        data = []
        for idx, batch in enumerate(dataloader):
            inp, opt = batch
            for i in range(len(inp)):
                data_dict = {"input": inp[i], "output": opt[i]}
                data.append(data_dict)

        df = pd.DataFrame(data)
        return df

    def test_model(model, inpLang, optLang, test_df):
        tester = Model_tester(model, inpLang, optLang, max_len=40)
        tester.set_inference_mode("greedy")
        return tester.generate_metrics(test_df)

    """
    Main Loop
    """
    test_df = create_test_df(test_dataloader)
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
                "test_df": test_df,
                "params": vars(hp),
                "version": version,
            }
            save_model(path=hp.save_path, name="seq2seq.pt", **to_save)
            best_model_loss = val_loss
        e_time = time.time()
        epoch_mins, epoch_secs = epoch_time(st_time, e_time)
        print(
            f"\tTraining Loss : {np.mean(train_loss)}",
            f"Train Perplexity : {math.exp(np.mean(train_loss))}",
            sep="\t|\t",
        )
        print(
            f"\tVal Loss      : {val_loss}",
            f"Val Perplexity : {math.exp(val_loss)}",
            sep="\t|\t",
        )
        print(f"\tTime per epoch: {epoch_mins}m {epoch_secs}s")
    print("Generating Test Metrics")
    df, metrics = test_model(model, data.inpLang, data.optLang, test_df)
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
    parser.add_argument("--sample", action="store_true", help="Sample")
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
    main(hp)
