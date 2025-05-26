#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/05/11 15:45:31

import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--activate-type",
    type=int,
    default=3,
    help="0:l1 1:l1 smooth 2:sigmoid 3:arcl1",
)
parser.add_argument("--action-type", type=str, default="Ball")
parser.add_argument("--alpha", type=float, default=0.0)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--clip-num", type=int, default=68)
parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument("--dataset", type=str, default="FS1000")
parser.add_argument(
    "--decay-rate",
    type=float,
    default=0.1,
    help="lr decay rate",
)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--epoch", type=int, default=400)
parser.add_argument("--exp-name", type=str, default=None, help="exp name")
parser.add_argument("--flow_hidden_dim", type=int, default=256, help="flow step")
parser.add_argument("--gpus", nargs="+", type=int, help="gpu ids", default=[0])
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--in_dim", type=int, default=1024)
parser.add_argument(
    "--loss_align",
    type=int,
    default=1,
    help="0:GDLK 1:short 2:long",
)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument(
    "--lr-decay",
    type=str,
    default=None,
    help="use what decay scheduler",
)
parser.add_argument("--margin", type=float, default=0.0)

parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--multmoding", type=bool, default=False)
parser.add_argument("--n_decoder", type=int, default=1)
parser.add_argument("--n_encoder", type=int, default=1)
parser.add_argument("--n_head", type=int, default=1)
parser.add_argument("--n_query", type=int, default=1)
parser.add_argument("--optim", type=str, default="sgd")
parser.add_argument("--score-type", type=str, default="Total_Score")
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="random seed",
)
parser.add_argument("--model-name", type=str, default="model", help="model name")
parser.add_argument(
    "--test",
    action="store_true",
    help="only evaluate, don't train",
)
parser.add_argument(
    "--test-label-path",
    type=str,
    default="../action_assessment/rg_feat/test.txt",
)
parser.add_argument(
    "--train-label-path",
    type=str,
    default="../action_assessment/rg_feat/train.txt",
)
parser.add_argument(
    "--video-path",
    type=str,
    default="../action_assessment/rg_feat/swintx_avg_fps25_clip32",
)
parser.add_argument("--warmup", type=int, default=0, help="warmup epoch")
parser.add_argument("--weight-decay", type=float, default=1e-4)
