import sys

sys.path.append("./src/")

import tpu_setup

import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

from dataloader import SimpleDataloader
from params import SEQ2SEQ_PARAMS
from models.seq2seq import Seq2seq, count_parameters, Encoder, Decoder
from utils import save_model, get_torch_device, epoch_time

hp = SEQ2SEQ_PARAMS
device = get_torch_device()


def train_seq2seq():
    torch.manual_seed(1)
    if not xm.is_master_ordinal():
        # Barrier: Wait until master is done downloading
        xm.rendezvous("download_only_once")

    # Get and shard dataset into dataloaders
    data = SimpleDataloader(**vars(hp))

    if xm.is_master_ordinal():
        # Barrier: Master done downloading, other workers can proceed
        xm.rendezvous("download_only_once")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        data.train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
    )
    train_loader = torch.utils.data.DataLoader(
        data.train_dataset,
        batch_size=data.batch_size,
        sampler=train_sampler,
        num_workers=4,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        data.val_dataset,
        batch_size=data.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    # Scale learning rate to world size
    lr = hp.lr * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()
    model = Seq2seq(**vars(hp))
    model.init_weights()
    model.to(device)

    """
    Loss Function and optimizers
    """
    loss_fn = F.cross_entropy
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, (data, target, _) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data, target)
            output = output.permute(1, 2, 0)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(hp.batch_size)
            if x % 100 == 0:
                print(
                    "[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}".format(
                        xm.get_ordinal(),
                        x,
                        loss.item(),
                        tracker.rate(),
                        tracker.global_rate(),
                        time.asctime(),
                    ),
                    flush=True,
                )

    def test_loop_fn(loader):
        total_samples = 0
        losses = 0
        model.eval()
        data, pred, target = None, None, None
        for (data, target, _) in loader:
            output = model(data, target)
            output = output.permute(1, 2, 0)
            loss = loss_fn(output, target)
            losses += loss.item()
            total_samples += data.size()[0]
        val_loss = losses / total_samples
        print("[xla:{}] Loss={:.2f}%".format(xm.get_ordinal(), val_loss), flush=True)
        return val_loss, data, pred, target

    # Train and eval loops
    loss = 0.0
    data, pred, target = None, None, None
    for epoch in range(1, hp.epochs + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

        para_loader = pl.ParallelLoader(test_loader, [device])
        loss, data, pred, target = test_loop_fn(para_loader.per_device_loader(device))

        # xm.master_print(met.metrics_report(), flush=True)

    return loss, data, pred, target


# Start training processes
def _mp_fn(rank, flags):
    # global FLAGS
    # FLAGS = flags
    # torch.set_default_tensor_type('torch.FloatTensor')
    accuracy, data, pred, target = train_seq2seq()
    if rank == 0:
        # Retrieve tensors that are on TPU core 0 and plot.
        plot_results(data.cpu(), pred.cpu(), target.cpu())


xmp.spawn(_mp_fn, args=(hp,), nprocs=8, start_method="fork")
