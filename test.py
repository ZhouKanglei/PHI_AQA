#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/05/12 12:05:44


import time
import numpy as np
import torch
from torch import nn
from utils import compute_matric


def test_epoch(
    epoch,
    model,
    test_loader,
    logger,
    device,
    args,
    etf_head=False,
):

    model.eval()

    preds = np.array([])
    labels = np.array([])

    with torch.no_grad():
        for i, a in enumerate(test_loader):

            video_feat, label = a
            video_feat = video_feat.to(device)

            label = label.float().to(device)
            out = model(video_feat)

            if etf_head:
                pred = model.get_score(out["output"])
            else:
                pred = out["output"]

            if len(preds) == 0:
                preds = pred.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)

    coef, p, L2, RL2 = compute_matric(preds, labels)

    if logger is not None:
        logger.add_scalar("Test coef", coef, epoch)
        logger.add_scalar("Test loss", L2, epoch)

    return {
        "coef": coef,
        "pred": preds,
        "label": labels,
        "L2": L2,
        "RL2": RL2,
    }
