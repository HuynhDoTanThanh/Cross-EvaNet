#!/usr/bin/env python3
"""
Phase-2 training script for multi-view CheXray (Triple-Branch EVA).
Supports TPU (Kaggle/Colab) via torch_xla, GPU and CPU.

E_s is loaded from the Phase-1 checkpoint (--pretrained, required) and frozen;
E_m is initialised from EVA-X weights (--mv-init-ckpt, required). Training runs
the full epoch budget and saves the final-epoch checkpoint; validation AUC is
logged each epoch for monitoring only.

Usage:
  # TPU (set TPU env or run on Kaggle)
  python -m scripts.train --config configs/phase2_B_matched_apl.json --use-tpu

  # GPU / CPU
  python -m scripts.train --train-csv data/train_mv.csv --train-dir data/train \
    --split-csv configs/split_B.csv --pretrained outputs/phase1_B_448.pth \
    --mv-init-ckpt weights/eva_x_base_patch16_merged520k_mim.pt
"""

import argparse
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.config import TrainConfig, load_json_config
from src.constants import NUM_LABELS, seed_everything
from src.dataset import build_transforms, make_loaders, read_train_csv, split_by_patient
from src.losses import LOSS_NAMES, build_loss
from src.models import FUSION_HEAD_TYPES, build_triple_branch_model
from src.precision import check_xla_mixed_precision_env, make_grad_scaler
from src.train import build_scheduler, evaluate, train_one_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: train the multi-view CheXray model")
    p.add_argument("--config", default=None, help="JSON of TrainConfig fields; CLI flags override it")
    p.add_argument("--train-csv", help="Path to train CSV (Patient_ID, Study, Image_name, ...)")
    p.add_argument("--train-dir", help="Directory containing training images")
    p.add_argument("--val-split", type=float, help="Validation fraction when creating the split")
    p.add_argument("--split-csv", help="Persisted patient split, e.g. configs/split_B.csv (created on first use)")
    p.add_argument("--pretrained", help="Phase-1 single-view checkpoint theta_s* for E_s (required)")
    p.add_argument(
        "--mv-init-ckpt",
        "--pretrained-2",
        dest="mv_init_ckpt",
        help="EVA-X weights initialising E_m: public MIM checkpoint or a Phase-1 "
        "checkpoint (required; --pretrained-2 is a deprecated alias)",
    )
    p.add_argument("--save-path", help="Where to save the final-epoch checkpoint")
    p.add_argument("--img-size", type=int)
    p.add_argument("--num-views", type=int, choices=[2])
    p.add_argument("--drop-path-rate", type=float)
    p.add_argument("--fusion-head", dest="fusion_head_type", choices=FUSION_HEAD_TYPES)
    p.add_argument(
        "--single-view-route",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Score single-image validation studies with the frozen single-view logits",
    )
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--warmup-pct", type=float)
    p.add_argument("--grad-accum-steps", type=int)
    p.add_argument("--clip-grad", type=float)
    p.add_argument("--loss", choices=LOSS_NAMES)
    p.add_argument("--seed", type=int)
    p.add_argument("--use-tpu", action="store_true", help="Use TPU (torch_xla)")
    return p.parse_args()


def build_config(args) -> TrainConfig:
    """TrainConfig from defaults, then the JSON config, then CLI flags."""
    if any(a == "--pretrained-2" or a.startswith("--pretrained-2=") for a in sys.argv):
        print("WARNING: --pretrained-2 is deprecated; use --mv-init-ckpt.")
    values = load_json_config(args.config) if args.config else {}
    field_names = {f.name for f in fields(TrainConfig)}
    values.update({k: v for k, v in vars(args).items() if k in field_names and v is not None})
    cfg = TrainConfig(**values)
    required = {
        "train_dir": "--train-dir",
        "split_csv": "--split-csv",
        "pretrained": "--pretrained",
        "mv_init_ckpt": "--mv-init-ckpt",
    }
    missing = [flag for name, flag in required.items() if not getattr(cfg, name)]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)

    use_xla = args.use_tpu
    if use_xla:
        check_xla_mixed_precision_env()
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        device = xm.xla_device()
        log = xm.master_print
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log = print
    log(f"Using device: {device}")
    log(f"Config: {asdict(cfg)}")

    # Seed
    seed_everything(cfg.seed, use_xla=use_xla)

    # Data
    df = read_train_csv(cfg.train_csv)
    train_df, val_df = split_by_patient(df, cfg.split_csv, val_frac=cfg.val_split, seed=cfg.seed)
    log(f"Train Rows: {len(train_df)}, Val Rows: {len(val_df)}")

    model = build_triple_branch_model(NUM_LABELS, cfg)
    model = model.to(device)

    train_tf, val_tf = build_transforms(cfg.img_size)
    train_loader, val_loader = make_loaders(train_df, val_df, cfg, train_tf, val_tf)
    steps_per_epoch = len(train_loader)
    has_val = len(val_loader.dataset) > 0
    if not has_val:
        log("WARNING: the split has no validation studies; per-epoch monitoring is skipped.")
    if use_xla:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = build_loss(cfg.loss)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)
    scaler = make_grad_scaler(device)

    for epoch in range(1, cfg.epochs + 1):
        log(f"\n===== Epoch {epoch}/{cfg.epochs} =====")
        total_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg,
            use_xla=use_xla, scaler=scaler,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        log(f"total_loss: {total_loss:.4f} | LR: {current_lr:.6f}")

        # Monitoring only: no checkpoint is selected on validation AUC.
        if not has_val:
            continue
        metrics = evaluate(model, val_loader, device, use_xla=use_xla)
        log(
            f"val per-image macro ROC-AUC ({metrics['n_images']} images, {metrics['n_studies']} studies):"
            f" {metrics['macro_auc']:.4f}"
        )
        log(f"per_class_auc: {[round(a, 4) for a in metrics['per_class_auc']]}")
        log(
            f"paired studies ({metrics['n_paired']} studies, {metrics['n_paired_images']} images):"
            f" fused {metrics['paired_fused_macro_auc']:.4f}"
            f" | single-view per image {metrics['paired_single_macro_auc']:.4f}"
        )

    Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": model.state_dict(), "config": asdict(cfg), "epoch": cfg.epochs}
    if use_xla:
        xm.save(checkpoint, cfg.save_path)
    else:
        torch.save(checkpoint, cfg.save_path)
    log(f"Training finished. Final-epoch checkpoint saved to {cfg.save_path}")


if __name__ == "__main__":
    main()
