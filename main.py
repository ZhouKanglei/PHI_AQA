#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/05/11 15:09:19

import os
import csv
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


import options
from logger import Logger
from datasets import RGDataset
from models import loss
from train import train_epoch
from test import test_epoch


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adam":
        optim = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optim == "adamw":
        optim = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optim == "rmsprop":
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=args.epoch - args.warmup,
                eta_min=args.lr * args.decay_rate,
            )
        elif args.lr_decay == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optim, milestones=[args.epoch - 30], gamma=args.decay_rate
            )
        else:
            raise Exception("Unknown scheduler")
    else:
        scheduler = None
    return scheduler


def build_model(args):
    if args.model_name == "phi":
        from models.model_phi import PHI as Model

        print("PHI...")
        model = Model(
            args.in_dim,
            args.hidden_dim,
            args.n_head,
            args.n_encoder,
            args.n_decoder,
            args.n_query,
            args.dropout,
            args.activate_type,
        )

        from models.model_phi import FlowMatch

        flow_model = FlowMatch(args.hidden_dim, 1, 1, 2, args.flow_hidden_dim, 0.3)

        model.flow_model = flow_model

        return model

    else:
        assert False, "Unknown model name"


def save_history(save_path, best_coef, best_epoch, rl2):
    history_file = os.path.join("/home/contlrn/zkl/Codes/PHI_AQA/outputs/history.csv")
    with open(history_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(history_file).st_size == 0:
            writer.writerow(["Best Epoch", "Coef", "RL2", "Output Directory"])
        writer.writerow([best_epoch, f"{best_coef:.3f}", f"{rl2:.3f}", save_path])
        # sorder with the third column
        with open(history_file, mode="r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # check if the first row is ["Best Epoch", "Coef", "RL2", "Output Directory"]
            new_rows = []
            for row in rows:
                if row[-2] == "RL2":
                    pass
                else:
                    new_rows.append(row)

            new_rows.sort(key=lambda x: x[-1], reverse=True)
        # write the sorted rows back to the file
        with open(history_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Best Epoch", "Coef", "RL2", "Output Directory"])
            for row in new_rows:
                writer.writerow(row)

        f.close()


if __name__ == "__main__":
    """
    Step 1: init
    """

    # parse args
    args = options.parser.parse_args()
    # set logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.test:
        timestamp = "test-" + timestamp
    print("********* Timestamp: ", timestamp)

    if args.exp_name is not None:
        save_path = os.path.join(
            "outputs", args.model_name, args.action_type, args.exp_name
        )
    else:
        save_path = os.path.join(
            "outputs", args.model_name, args.action_type, timestamp
        )
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, "log.txt")

    logger = Logger(
        file_name=log_file,
        file_mode="w+",
        should_flush=True,
    )

    print(" Init ".center(30, "-"))
    print("Log file:", log_file)
    print("Args: ", args)
    # set random seed
    setup_seed(args.seed)
    print(" Random seed:", args.seed)
    # set gpu
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # using etf
    if "phi" in args.model_name:
        use_etf = True
    else:
        use_etf = False

    """
    Step 2: load data
    """
    # load training dataset
    print(" Loading data ".center(30, "-"))
    train_data = RGDataset(
        args.video_path,
        args.train_label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    # load testing dataset
    test_data = RGDataset(
        args.video_path,
        args.test_label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
        train=False,
    )
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=8, drop_last=True
    )

    print("Train data size:", len(train_data))
    print(" Test data size:", len(test_data))

    """
    Step 3: load model
    """
    print(" Loading model ".center(30, "-"))

    model = build_model(args).to(device)

    # load pretrained model
    if args.ckpt is not None:
        print("Load pretrained model from ", args.ckpt)
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint, strict=False)

    # record model
    tb_path = os.path.join("outputs/tensorboard", args.model_name, timestamp)
    logger = SummaryWriter(tb_path)
    best_coef, best_epoch, rl2 = -1, -1, -1
    final_train_loss, final_train_coef = 0, 0
    final_test_loss, final_test_coef = 0, 0

    """
    Step 4: test
    """
    if args.test:
        print(" Testing ".center(30, "-"))
        out = test_epoch(0, model, test_loader, logger, device, args, use_etf)
        print(
            f"Test Loss: {out['L2']:.4f}, Test Coef: {out['coef']:.3f}, "
            f"L2: {out['L2']:.3f}, RL2: {out['RL2']:.3f}"
        )

        raise SystemExit

    """
    Step 5: train
    """
    # optimizer
    optim = get_optim(model, args)
    optim_flow = get_optim(model.flow_model, args)
    # scheduler
    scheduler = get_scheduler(optim, args)
    # loss
    loss_fn = loss.LossFun(
        args.alpha, args.margin, True, args.loss_align, beta=args.beta
    )
    # learning rate warmup
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda t: t / args.warmup
        )
    else:
        warmup = None

    # training loop
    print(" Training ".center(30, "-"))
    for epc in range(args.epoch):
        if args.warmup and epc < args.warmup:
            warmup.step()
        # print("Learning rate: ", optim.state_dict()["param_groups"][0]["lr"])
        avg_loss, train_coef, L2, RL2 = train_epoch(
            epc,
            model,
            loss_fn,
            train_loader,
            optim,
            logger,
            device,
            args,
            use_etf,
            optim_flow,
        )
        # scheduler
        if scheduler is not None and (args.lr_decay != "cos" or epc >= args.warmup):
            scheduler.step()
        # evaluation
        out = test_epoch(epc, model, test_loader, logger, device, args, use_etf)
        if out["coef"] > best_coef:
            best_coef, best_epoch, rl2 = out["coef"], epc, out["RL2"]
            print(f"**** Best model saved, coef: {best_coef:.3f}, rl2: {rl2:.3f}")
            # save model
            torch.save(model.state_dict(), save_path + "/best.pkl")

        print(
            f"Epoch: {epc}\tLoss: {avg_loss:.4f}\t"
            f"Train Coef: {train_coef:.3f}\tTest Loss: {out['L2']:.4f}\t"
            f"Test Coef: {out['coef']:.3f}\tL2: {out['L2']:.3f}\tRL2: {out['RL2']:.3f}"
        )
        if epc == args.epoch - 1:
            final_train_loss, final_train_coef = avg_loss, train_coef
            final_test_loss, final_test_coef = out["L2"], out["coef"]

    print(f"Best Test Coef: {best_coef:.3f}, Epoch: {best_epoch}, RL2: {rl2:.3f}")

    # write a history file: output dir, best epoch, best coef, best rl2
    save_history(save_path, best_coef, best_epoch, rl2)
