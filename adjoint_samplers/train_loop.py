# Copyright (c) Meta Platforms, Inc. and affiliates.

from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.aggregation import MeanMetric

import adjoint_samplers.utils.train_utils as train_utils
from adjoint_samplers.components.matcher import Matcher

import copy

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train_one_epoch(
    matcher: Matcher,
    model: torch.nn.Module,
    source: torch.nn.Module,
    optimizer: Optimizer,
    lr_schedule: LRScheduler | None,
    epoch: int,
    device: str,
    cfg: DictConfig,
):
    # build dataloader
    B = cfg.resample_batch_size
    M = matcher.resample_size // (B * cfg.world_size)
    loss_scale = matcher.loss_scale

    is_asbs_init_stage = train_utils.is_asbs_init_stage(epoch, cfg)

    for _ in range(M):
        x0 = source.sample([B,]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        matcher.populate_buffer(x0, timesteps, is_asbs_init_stage)

    dataloader = matcher.build_dataloader(cfg.train_batch_size)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    loader = iter(cycle(dataloader))

    eta = cfg.get("damping", 0.0)
    prev_model = None
    if eta > 0:
        base = model.module if hasattr(model, "module") else model   # DDP
        prev_model = copy.deepcopy(base).eval()
        for p in prev_model.parameters():
            p.requires_grad_(False)

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        optimizer.zero_grad()

        data = next(loader)

        input, target = matcher.prepare_target(data, device)
        output = model(*input)

        loss = ((output - target)**2).mean()
        if prev_model is not None:
            with torch.no_grad():
                prev_output = prev_model(*input)
            loss = loss + eta * ((output - prev_output)**2).mean()
        loss = loss_scale * loss
        loss.backward()

        if cfg.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e20)

        optimizer.step()

        epoch_loss.update(loss.item())
        if lr_schedule:
            lr_schedule.step()

    return float(epoch_loss.compute().detach().cpu())
