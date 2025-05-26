#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/05/12 11:39:49


import numpy as np
import torch
import torch.nn.functional as F

from utils import AverageMeter, compute_matric


def train_epoch(
    epoch,
    model,
    loss_fn,
    train_loader,
    optim,
    logger,
    device,
    args,
    etf_head=False,
    optim_flow=None,
):
    model.train()
    preds = np.array([])
    labels = np.array([])

    losses = AverageMeter("loss", logger)
    mse_losses = AverageMeter("mse", logger)
    tri_losses = AverageMeter("tri", logger)

    for i, a in enumerate(train_loader):

        video_feat, label = a
        video_feat = video_feat.to(device)  # (b, t, c)
        label = label.float().to(device)
        out = model(video_feat, parse="train")
        pred = out["output"]

        if etf_head:
            new_label = model.get_proj_class(label, pred)
        else:
            new_label = label

        loss, mse, tri = loss_fn(pred, new_label, out["embed"])
        if type(loss) is list:
            floss = loss[1]
            loss = loss[0]

        optim.zero_grad()

        loss.backward()
        optim.step()

        if optim_flow is not None:
            optim_flow.zero_grad()
            p_list = out["flow"]
            loss_flow = model.flow_match(p_list)
            loss_flow = torch.stack(loss_flow).mean()
            loss_flow.backward()
            optim_flow.step()

        losses.update(loss, label.shape[0])
        mse_losses.update(mse, label.shape[0])
        tri_losses.update(tri, label.shape[0])
        
        if etf_head:
            pred = model.get_score(pred)
            
        if len(preds) == 0:
            preds = pred.cpu().detach().numpy()
            labels = label.cpu().detach().numpy()
        else:
            preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
            labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)

    coef, p, L2, RL2 = compute_matric(preds, labels)
    if logger is not None:
        logger.add_scalar("train coef", coef, epoch)

    avg_loss = losses.done(epoch)
    mse_losses.done(epoch)
    tri_losses.done(epoch)

    return avg_loss, coef, L2, RL2
