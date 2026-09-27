#!/usr/bin/env python3
"""
Phase-1 training script: single-view fine-tuning of the EVA-X ViT-B encoder.
Supports TPU (Kaggle/Colab) via torch_xla, GPU and CPU.

The encoder is initialised from the public EVA-X masked-image-modelling
weights (--init-ckpt; pos_embed resampled to the input grid) and fine-tuned
on individual radiographs with per-image labels, Strong Augment and APL
(--loss selects another objective). Training runs the full epoch budget and
saves the final-epoch checkpoint theta_s*, which Phase 2 loads strictly as
E_s (--pretrained); validation AUC is logged each epoch for monitoring only.

Regimes (one entry per division in --train-csv / --train-dir / --split-csv):
  matched-data: the target division only.
  full data:    both divisions merged; every division's validation patients
                are excluded from training.

Usage:
  # Matched-data regime, Division B, 448 (TPU)
  python -m scripts.train_phase1 --config configs/phase1_B_matched_448.json --use-tpu

  # Full-data regime (Divisions A + B), GPU / CPU
  python -m scripts.train_phase1 --config configs/phase1_full_448.json

  # Single-view baseline at 224 under ASL
  python -m scripts.train_phase1 --config configs/phase1_B_matched_448.json \
    --img-size 224 --loss asl --save-path outputs/phase1_B_matched_224_asl.pth
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.constants import NUM_LABELS, seed_everything
from src.dataset import (
    SingleViewXRayDataset,
    build_transforms,
    read_train_csv,
    split_by_patient,
    val_patients_from_splits,
)
from src.losses import LOSS_NAMES, build_loss
from src.models import eva_x_base_patch16, load_evax_init_weights
from src.precision import check_xla_mixed_precision_env, make_grad_scaler
from src.train import build_scheduler, evaluate_single_view, train_one_epoch


@dataclass
class Phase1Config:
    """Configuration for Phase-1 single-view fine-tuning (paper Sec. 5.1)."""

    # Data: one entry per division (1 = matched-data, 2 = full data)
    train_csv: List[str] = field(default_factory=list)
    train_dir: List[str] = field(default_factory=list)
    # Persisted patient splits shared with Phase 2, e.g. configs/split_B.csv
    # (created from the matching train CSV on first use).
    split_csv: List[str] = field(default_factory=list)
    val_split: float = 0.1

    # Checkpoints
    init_ckpt: Optional[str] = None  # public EVA-X ViT-B MIM weights
    save_path: str = "outputs/crossevanet_phase1.pth"  # final-epoch theta_s*

    # Model
    img_size: int = 448
    drop_path_rate: float = 0.2

    # Training
    batch_size: int = 16
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 10
    warmup_pct: float = 0.05
    grad_accum_steps: int = 1
    clip_grad: float = 1.0
    loss: str = "apl"  # bce, focal, twoway, zlpr, asl, apl
    seed: int = 1337

    def __post_init__(self) -> None:
        for name in ("train_csv", "train_dir", "split_csv"):
            value = getattr(self, name)
            setattr(self, name, [value] if isinstance(value, str) else list(value))
        lengths = {len(v) for v in (self.train_csv, self.train_dir, self.split_csv) if v}
        if len(lengths) > 1:
            raise ValueError(
                "train_csv, train_dir and split_csv need one entry per division, got "
                f"{len(self.train_csv)}, {len(self.train_dir)} and {len(self.split_csv)}"
            )


def load_phase1_json(path: str) -> Dict[str, Any]:
    """Read a per-experiment JSON config whose keys are Phase1Config fields."""
    with open(path) as f:
        values = json.load(f)
    unknown = sorted(set(values) - {f.name for f in fields(Phase1Config)})
    if unknown:
        raise ValueError(f"Unknown Phase1Config fields in {path}: {unknown}")
    return values


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: single-view fine-tuning of EVA-X")
    p.add_argument("--config", default=None, help="JSON of Phase1Config fields; CLI flags override it")
    p.add_argument("--train-csv", nargs="+", help="Train CSV per division (Patient_ID, Study, Image_name, ...)")
    p.add_argument("--train-dir", nargs="+", help="Image directory per division")
    p.add_argument("--split-csv", nargs="+", help="Persisted patient split per division, e.g. configs/split_B.csv")
    p.add_argument("--val-split", type=float, help="Validation fraction when creating a split")
    p.add_argument("--init-ckpt", help="Public EVA-X ViT-B MIM checkpoint (required)")
    p.add_argument("--save-path", help="Where to save the final-epoch checkpoint")
    p.add_argument("--img-size", type=int, help="Input resolution (448; 224 for the single-view baseline)")
    p.add_argument("--drop-path-rate", type=float)
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


def build_config(args) -> Phase1Config:
    """Phase1Config from defaults, then the JSON config, then CLI flags."""
    values = load_phase1_json(args.config) if args.config else {}
    field_names = {f.name for f in fields(Phase1Config)}
    values.update({k: v for k, v in vars(args).items() if k in field_names and v is not None})
    cfg = Phase1Config(**values)
    required = {
        "train_csv": "--train-csv",
        "train_dir": "--train-dir",
        "split_csv": "--split-csv",
        "init_ckpt": "--init-ckpt",
    }
    missing = [flag for name, flag in required.items() if not getattr(cfg, name)]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")
    return cfg


def build_single_view_model(
    img_size: int, drop_path_rate: float, init_ckpt: Optional[str] = None
) -> Tuple[nn.Module, Optional[str]]:
    """EVA-X ViT-B classifier for Phase 1, optionally initialised from ``init_ckpt``.

    Returns the model and the detected checkpoint source ("mim" / "single_view").
    """
    model = eva_x_base_patch16(
        pretrained=False,
        drop_path_rate=drop_path_rate,
        img_size=img_size,
        num_classes=NUM_LABELS,
    )
    source = load_evax_init_weights(model, init_ckpt) if init_ckpt else None
    return model, source


def build_datasets(cfg: Phase1Config, train_tf, val_tf, log=print):
    """Per-image train / val datasets over the configured divisions.

    Each division is split with its persisted patient split; the validation
    patients of every division are then removed from all training rows, so
    that in the full-data regime no Phase-2 validation patient is seen in
    Phase 1.
    """
    divisions = []
    for train_csv, split_csv in zip(cfg.train_csv, cfg.split_csv):
        df = read_train_csv(train_csv)
        train_df, val_df = split_by_patient(df, split_csv, val_frac=cfg.val_split, seed=cfg.seed)
        divisions.append((train_df, val_df))
    val_ids = val_patients_from_splits(cfg.split_csv)

    train_sets, val_sets = [], []
    for (train_df, val_df), train_dir, train_csv in zip(divisions, cfg.train_dir, cfg.train_csv):
        is_val = train_df["Patient_ID"].astype(str).isin(val_ids)
        if is_val.any():
            log(
                f"{train_csv}: excluded {int(is_val.sum())} training images of "
                f"{train_df.loc[is_val, 'Patient_ID'].nunique()} patients that are "
                "validation patients of another division"
            )
        train_df = train_df[~is_val].reset_index(drop=True)
        log(f"{train_csv}: train images {len(train_df)}, val images {len(val_df)}")
        train_sets.append(SingleViewXRayDataset(train_df, train_dir, transform=train_tf))
        val_sets.append(SingleViewXRayDataset(val_df, train_dir, transform=val_tf))
    return ConcatDataset(train_sets), ConcatDataset(val_sets)


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
    log(f"Regime: {'matched-data' if len(cfg.train_csv) == 1 else 'full data'} ({len(cfg.train_csv)} division(s))")

    # Seed
    seed_everything(cfg.seed, use_xla=use_xla)

    # Data
    train_tf, val_tf = build_transforms(cfg.img_size)
    train_ds, val_ds = build_datasets(cfg, train_tf, val_tf, log=log)
    log(f"Train Images: {len(train_ds)}, Val Images: {len(val_ds)}")
    if len(val_ds) == 0:
        log("WARNING: the split has no validation images; per-epoch monitoring is skipped.")

    model, source = build_single_view_model(cfg.img_size, cfg.drop_path_rate, cfg.init_ckpt)
    if source != "mim":
        log(f"WARNING: {cfg.init_ckpt} is not a MIM checkpoint; Phase 1 starts from public EVA-X MIM weights.")
    model = model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )
    steps_per_epoch = len(train_loader)
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
        if len(val_ds) == 0:
            continue
        metrics = evaluate_single_view(model, val_loader, device)
        log(f"val per-image macro ROC-AUC: {metrics['macro_auc']:.4f}")
        log(f"per_class_auc: {[round(a, 4) for a in metrics['per_class_auc']]}")

    # Same format as the Phase-2 checkpoint; Phase 2 loads "model" strictly as E_s.
    Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": model.state_dict(), "config": asdict(cfg), "epoch": cfg.epochs}
    if use_xla:
        xm.save(checkpoint, cfg.save_path)
    else:
        torch.save(checkpoint, cfg.save_path)
    log(f"Training finished. Final-epoch checkpoint saved to {cfg.save_path}")


if __name__ == "__main__":
    main()
