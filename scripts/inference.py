#!/usr/bin/env python3
"""
Inference script: load a Phase-2 checkpoint, run on test images, write submission CSV.

Images of a study are taken in file order; studies with more than two images
are scored on consecutive pairs whose probabilities are averaged, and studies
with a single image return the frozen single-view prediction.

Usage:
  python -m scripts.inference \
    --checkpoint outputs/crossevanet_phase2.pth \
    --test-dir data/test \
    --output submission.csv \
    [--use-tpu]
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from torch.utils.data import DataLoader

from src.constants import NUM_LABELS
from src.dataset import TestMultiViewDataset, build_transforms
from src.inference import aggregate_study_preds, build_submission, run_inference
from src.models import load_triple_branch_model
from src.precision import check_xla_mixed_precision_env


def parse_args():
    p = argparse.ArgumentParser(description="Run inference and build submission")
    p.add_argument("--checkpoint", required=True, help="Phase-2 checkpoint written by scripts/train.py")
    p.add_argument("--test-dir", required=True, help="Directory of test images (*.jpg / *.png)")
    p.add_argument("--output", default="submission.csv", help="Output submission CSV path")
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--num-views", type=int, default=2, choices=[2])
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--single-view-route",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return the frozen single-view logits for single-image studies (default: on)",
    )
    p.add_argument("--use-tpu", action="store_true", help="Use TPU (torch_xla)")
    return p.parse_args()


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
    test_ds = TestMultiViewDataset(args.test_dir, transform=val_tf, num_views=args.num_views)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    if use_xla:
        test_loader = pl.MpDeviceLoader(test_loader, device)

    model = load_triple_branch_model(
        args.checkpoint,
        NUM_LABELS,
        img_size=args.img_size,
        single_view_route=args.single_view_route,
    )
    model = model.to(device)
    model.eval()

    log("Starting inference...")
    results = run_inference(model, test_loader, device, use_xla=use_xla)
    final_preds = aggregate_study_preds(results, num_labels=NUM_LABELS)
    build_submission(args.test_dir, final_preds, output_path=args.output)
    log(f"Submission saved to {args.output}")


if __name__ == "__main__":
    main()
