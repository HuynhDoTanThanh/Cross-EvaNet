#!/usr/bin/env python3
"""
Zero-shot evaluation of a Phase-2 checkpoint on a list of paired studies.

The unit of evaluation is the image (radiograph), as on the Grand X-Ray SLAM
leaderboard, and every image carries its study's labels. One forward pass
gives, for every image:

  single_<L>  the frozen single-view (Phase-1) encoder inside the model applied
              to that image on its own: y1 for the slot-1 image, y2 for the
              slot-2 image (no averaging over the images of a study);
  fused_<L>   the study-level Cross-EvaNet logits (Eq. (1):
              0.5 * (0.5 * (y1 + y2) + y')), identical for all images of the
              study.

The CSV has one row per image (study_id, patient_id, image, slot, n_images,
label_<L>, single_<L>, fused_<L> for each label L) and is the input of
scripts/statistics.py, which resamples studies as clusters. ``--per-study``
instead writes one row per study with the earlier comparator 0.5 * (y1 + y2);
it is not the paper's protocol.

Inputs (one of):
  --study-list    CheXpert cohort from scripts/build_chexpert_cohort.py
                  (image_1 / image_2: the two selected images in metadata
                  order; -1 kept).
  --train-csv + --split-csv
                  The in-domain validation split of a division, restricted to
                  studies with >= 2 images (images in file order; a study with
                  n > 2 images is scored on its n - 1 consecutive pairs, whose
                  fused sigmoid probabilities are averaged and written back as
                  logits; each image keeps its own single-view logits).

No test-time augmentation; the validation transform is used unchanged.

Usage:
  python -m scripts.evaluate_chexpert \
    --checkpoint outputs/phase2_B_full_apl.pth \
    --study-list outputs/chexpert_cohort.csv --image-root /data \
    --output outputs/chexpert_zeroshot_logits.csv

  python -m scripts.evaluate_chexpert \
    --checkpoint outputs/phase2_B_full_apl.pth \
    --train-csv data/division_b/train_mv.csv --split-csv configs/split_B.csv \
    --image-root data/division_b/train --output outputs/divB_val_logits.csv
"""

import argparse
import contextlib
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from torch.utils.data import DataLoader

from src.constants import LABEL_COLUMNS, NUM_LABELS
from src.dataset import MultiViewEvalDataset, build_transforms, read_train_csv, split_by_patient
from src.dataset.study import STUDY_COLUMNS, study_labels
from src.inference import image_values, mean_by_study
from src.metrics import macro_auc, per_label_auc
from src.models import load_triple_branch_model
from src.precision import autocast

Studies = List[Tuple[str, List[str]]]


def parse_args():
    p = argparse.ArgumentParser(description="Per-image fused and single-view logits for statistics")
    p.add_argument("--checkpoint", required=True, help="Phase-2 checkpoint written by scripts/train.py")
    p.add_argument("--study-list", help="Study list CSV from scripts/build_chexpert_cohort.py")
    p.add_argument("--train-csv", help="Train CSV of a division (validation-split mode)")
    p.add_argument("--split-csv", help="Persisted patient split, e.g. configs/split_B.csv (validation-split mode)")
    p.add_argument("--image-root", required=True, help="Directory the image paths are relative to")
    p.add_argument("--output", required=True, help="Per-image logits CSV to write")
    p.add_argument(
        "--per-study",
        action="store_true",
        help="One row per study with the comparator 0.5 * (y1 + y2) instead (earlier protocol, not the paper's)",
    )
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision as in training/inference (default: on); --no-amp runs in fp32",
    )
    p.add_argument("--use-tpu", action="store_true", help="Use TPU (torch_xla)")
    args = p.parse_args()
    if bool(args.study_list) == bool(args.train_csv or args.split_csv):
        p.error("give either --study-list or both --train-csv and --split-csv")
    if bool(args.train_csv) != bool(args.split_csv):
        p.error("--train-csv and --split-csv go together")
    return args


def load_study_list(path: str) -> Tuple[Studies, List[str], np.ndarray]:
    """Studies (key, [slot-1 path, slot-2 path]), patient ids and labels (-1 kept)."""
    df = pd.read_csv(path, dtype={"study_key": str, "patient_id": str})
    missing = [c for c in ["study_key", "patient_id", "image_1", "image_2"] + LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if not df[LABEL_COLUMNS].isin([0, 1, -1]).all().all():
        raise ValueError(f"{path}: labels must be 1, 0 or -1 (uncertain)")
    if (df["image_1"] == df["image_2"]).any():
        raise ValueError(f"{path}: image_1 and image_2 must differ")
    studies = [(key, [a, b]) for key, a, b in zip(df["study_key"], df["image_1"], df["image_2"])]
    return studies, df["patient_id"].tolist(), df[LABEL_COLUMNS].to_numpy(dtype=np.float32)


def load_val_studies(train_csv: str, split_csv: str) -> Tuple[Studies, List[str], np.ndarray]:
    """Validation studies with >= 2 images from the persisted patient split."""
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Split file {split_csv} not found; it is written by scripts/train.py")
    _, val_df = split_by_patient(read_train_csv(train_csv), split_csv)
    study_map = val_df.groupby(STUDY_COLUMNS)["Image_name"].apply(list)
    study_map = study_map[study_map.apply(len) >= 2]
    keys = list(study_map.index)
    studies = [(f"{pid}_{study}", sorted(str(name) for name in study_map[(pid, study)])) for pid, study in keys]
    labels = study_labels(val_df, LABEL_COLUMNS).loc[keys].to_numpy(dtype=np.float32)
    return studies, [str(pid) for pid, _ in keys], labels


@torch.no_grad()
def predict_pair_logits(
    model: torch.nn.Module, loader, device: torch.device, amp: bool = True
) -> Dict[str, Any]:
    """Per-pair logits (fp32) from one forward pass.

    Returns ``study_keys`` [P], ``images`` [P] (the two image names, slot
    order), ``fused`` [P, C] (Eq. (1)) and ``views`` [P, 2, C] (frozen
    single-view logits y1, y2 of the two slots).
    """
    model.eval()
    keys: List[str] = []
    images: List[Tuple[str, str]] = []
    fused, views = [], []
    for batch in loader:
        imgs = batch["images"].to(device)
        with autocast(device) if amp else contextlib.nullcontext():
            out = model(imgs, single_mask=batch["single"].to(device), return_all=True)
        # One device-to-host transfer (on XLA each transfer executes the graph).
        packed = torch.stack([out["logits"], out["logits_1"], out["logits_2"]]).float().cpu().numpy()
        fused.append(packed[0])
        views.append(packed[1:].transpose(1, 0, 2))
        keys.extend(batch["study_key"])
        images.extend(zip(*batch["image_names"]))
    return {
        "study_keys": keys,
        "images": images,
        "fused": np.concatenate(fused),
        "views": np.concatenate(views),
    }


def study_logits(keys: Sequence[str], logits: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Per-study logits in first-seen order.

    A study scored on one pair keeps its logits; a study scored on several
    windows gets logit(mean sigmoid), the deployed probability-averaging rule.
    """
    logits = np.asarray(logits, dtype=np.float64)
    order, mean_logits = mean_by_study(keys, logits)
    _, mean_probs = mean_by_study(keys, expit(logits))
    counts = Counter(keys)
    n_pairs = np.array([counts[key] for key in order])[:, None]
    averaged = logit(np.clip(mean_probs, 1e-15, 1 - 1e-15))
    return order, np.where(n_pairs == 1, mean_logits, averaged)


def _with_scores(out: pd.DataFrame, labels: np.ndarray, single: np.ndarray, fused: np.ndarray) -> pd.DataFrame:
    """Append label_<L>, single_<L> and fused_<L> columns."""
    columns = {f"label_{name}": labels[:, c].astype(int) for c, name in enumerate(LABEL_COLUMNS)}
    for prefix, values in (("single", single), ("fused", fused)):
        columns.update({f"{prefix}_{name}": values[:, c] for c, name in enumerate(LABEL_COLUMNS)})
    return pd.concat([out, pd.DataFrame(columns, index=out.index)], axis=1)


def per_image_table(
    studies: Studies,
    patient_ids: Sequence[str],
    labels: np.ndarray,
    fused: np.ndarray,
    pairs: Dict[str, Any],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """One row per image: the study labels and fused logits, the image's own single-view logits.

    Returns the table and its label, single and fused arrays [I, C].
    """
    images, single_views = image_values(pairs["study_keys"], pairs["images"], pairs["views"])
    lookup = {image: i for i, image in enumerate(images)}
    rows, study_idx, image_idx = [], [], []
    for s, (key, names) in enumerate(studies):
        for slot, name in enumerate(names, start=1):
            rows.append((key, patient_ids[s], name, slot, len(names)))
            study_idx.append(s)
            image_idx.append(lookup[(key, name)])
    if len(rows) != len(lookup):
        raise RuntimeError("Every image of the study list must be scored exactly once")
    out = pd.DataFrame(rows, columns=["study_id", "patient_id", "image", "slot", "n_images"])
    labels_img, single_img, fused_img = labels[study_idx], single_views[image_idx], fused[study_idx]
    return _with_scores(out, labels_img, single_img, fused_img), labels_img, single_img, fused_img


def per_study_table(
    studies: Studies,
    patient_ids: Sequence[str],
    labels: np.ndarray,
    single: np.ndarray,
    fused: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """One row per study (``--per-study``): comparator 0.5 * (y1 + y2), window-averaged."""
    out = pd.DataFrame(
        {
            "study_id": [key for key, _ in studies],
            "patient_id": list(patient_ids),
            "n_images": [len(images) for _, images in studies],
        }
    )
    return _with_scores(out, labels, single, fused), labels, single, fused


def main():
    args = parse_args()
    use_xla = args.use_tpu
    if use_xla:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        device = xm.xla_device()
        log = xm.master_print
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log = print
    log(f"Using device: {device}")

    if args.study_list:
        studies, patient_ids, labels = load_study_list(args.study_list)
    else:
        studies, patient_ids, labels = load_val_studies(args.train_csv, args.split_csv)
    log(f"{len(studies)} paired studies, {sum(len(names) for _, names in studies)} images")

    _, val_tf = build_transforms(args.img_size)
    dataset = MultiViewEvalDataset(studies, args.image_root, val_tf, labels=labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if use_xla:
        loader = pl.MpDeviceLoader(loader, device)

    model = load_triple_branch_model(args.checkpoint, NUM_LABELS, img_size=args.img_size)
    model = model.to(device)

    pairs = predict_pair_logits(model, loader, device, amp=args.amp)
    study_keys = [key for key, _ in studies]
    order, fused = study_logits(pairs["study_keys"], pairs["fused"])
    if order != study_keys:
        raise RuntimeError("Predictions are not in study-list order")
    if args.per_study:
        _, single = study_logits(pairs["study_keys"], pairs["views"].mean(axis=1))
        out, labels_out, single_out, fused_out = per_study_table(studies, patient_ids, labels, single, fused)
        unit = "study"
    else:
        out, labels_out, single_out, fused_out = per_image_table(studies, patient_ids, labels, fused, pairs)
        unit = "image"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    fused_auc = macro_auc(per_label_auc(labels_out, fused_out))
    single_auc = macro_auc(per_label_auc(labels_out, single_out))
    log(f"{len(out)} rows ({unit} level), {len(studies)} studies")
    log(f"Per-{unit} macro AUC (uncertain labels excluded): fused {fused_auc:.4f}, single-view {single_auc:.4f}")
    log(f"Per-{unit} logits saved to {args.output}")


if __name__ == "__main__":
    main()
