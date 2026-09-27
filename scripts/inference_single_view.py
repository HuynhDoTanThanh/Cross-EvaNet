#!/usr/bin/env python3
"""
Single-view inference: score every test image on its own with a Phase-1 checkpoint.

This writes the submission of the single-view (EVA-X) rows of Tables 5 and 7,
at 448 or 224. The unit of evaluation is the image, as on the Grand X-Ray SLAM
leaderboard: every image is scored independently by the fine-tuned
single-view encoder (sigmoid(y) of that image; no pairing, no averaging over
the images of a study, no test-time augmentation). The CSV has the layout of
scripts/inference.py (Image_name, then the 14 labels), in file-name order.

Usage:
  python -m scripts.inference_single_view \
    --checkpoint outputs/phase1_B_matched_448.pth \
    --test-dir data/division_b/test \
    --output submissions/phase1_B_matched_448.csv \
    [--img-size 224] [--use-tpu]
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

from src.constants import LABEL_COLUMNS, NUM_LABELS
from src.dataset import TestSingleViewDataset, build_transforms
from src.models import eva_x_base_patch16, load_single_view_weights
from src.precision import autocast, check_xla_mixed_precision_env


def parse_args():
    p = argparse.ArgumentParser(description="Single-view submission from a Phase-1 checkpoint")
    p.add_argument("--checkpoint", required=True, help="Phase-1 checkpoint written by scripts/train_phase1.py")
    p.add_argument("--test-dir", required=True, help="Directory of test images (*.jpg / *.png)")
    p.add_argument("--output", default="submission_single_view.csv", help="Output submission CSV path")
    p.add_argument("--img-size", type=int, default=448, help="Resolution of the checkpoint (448, or 224)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-tpu", action="store_true", help="Use TPU (torch_xla)")
    return p.parse_args()


@torch.no_grad()
def predict_single_view(model: torch.nn.Module, loader, device: torch.device) -> Tuple[List[str], np.ndarray]:
    """Image names and per-image probabilities sigmoid(y) [I, C], loader order.

    The sigmoid is applied in float64 on the host, one transfer per batch.
    """
    model.eval()
    names: List[str] = []
    probs = []
    for batch in loader:
        imgs = batch["image"].to(device)
        with autocast(device):
            logits = model(imgs)
        probs.append(expit(logits.float().cpu().numpy().astype(np.float64)))
        names.extend(batch["image_name"])
    return names, np.concatenate(probs)


def main():
    args = parse_args()
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

    _, val_tf = build_transforms(args.img_size)
    test_ds = TestSingleViewDataset(args.test_dir, transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if use_xla:
        test_loader = pl.MpDeviceLoader(test_loader, device)

    model = eva_x_base_patch16(pretrained=False, img_size=args.img_size, num_classes=NUM_LABELS)
    load_single_view_weights(model, args.checkpoint)
    model = model.to(device)

    log(f"Scoring {len(test_ds)} images independently...")
    names, probs = predict_single_view(model, test_loader, device)
    if names != test_ds.image_names:
        raise RuntimeError("Predictions are not in file order")
    submission = pd.DataFrame(probs, columns=LABEL_COLUMNS)
    submission.insert(0, "Image_name", names)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    log(f"Single-view submission ({len(submission)} images) saved to {args.output}")


if __name__ == "__main__":
    main()
