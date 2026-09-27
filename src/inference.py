"""Inference, study-level aggregation, per-image evaluation rows and submission building.

The unit of evaluation and submission is the image (radiograph), as on the
Grand X-Ray SLAM leaderboard. Cross-EvaNet makes one prediction per study and
assigns it to every image of the study; the frozen single-view encoder scores
every image on its own.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from src.constants import LABEL_COLUMNS, NUM_LABELS
from src.dataset.study import is_test_image, test_study_key
from src.precision import autocast


def run_inference(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    use_xla: bool = False,
    with_single_view: bool = False,
) -> List[dict]:
    """Run a TripleBranchEVA over an evaluation loader, one record per pair.

    Records hold ``study_key``, ``probs`` (sigmoid of the model output, in
    float64 on the host so that large logits keep their ranking; single-image
    studies get the single-view route when the model enables it)
    and ``single``; plus, when ``with_single_view``, ``view_probs`` [2, C] =
    sigmoid(y1), sigmoid(y2) (the frozen single-view prediction of each slot's
    image, same forward pass) and ``images`` (the two image names, slot order;
    needs the loader's ``image_names``); and ``labels`` when the loader
    provides them. Precision follows ``device`` (``use_xla`` is kept for
    compatibility).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["images"].to(device)
            single = batch["single"].to(device)
            study_keys = batch["study_key"]
            with autocast(device):
                output = model(imgs, single_mask=single, return_all=with_single_view)
            if with_single_view:
                # One device-to-host transfer (on XLA each transfer executes the graph).
                packed = torch.stack([output["logits"], output["logits_1"], output["logits_2"]]).float()
                probs_np, probs_1, probs_2 = expit(packed.cpu().numpy().astype(np.float64))
                image_names = batch["image_names"]
            else:
                probs_np = expit(output.float().cpu().numpy().astype(np.float64))
            single_np = batch["single"].cpu().numpy()
            labels_np = batch["labels"].float().cpu().numpy() if "labels" in batch else None
            for i, key in enumerate(study_keys):
                record = {"study_key": key, "probs": probs_np[i], "single": bool(single_np[i])}
                if with_single_view:
                    record["view_probs"] = np.stack([probs_1[i], probs_2[i]])
                    record["images"] = (image_names[0][i], image_names[1][i])
                if labels_np is not None:
                    record["labels"] = labels_np[i]
                results.append(record)
    return results


def mean_by_study(
    study_keys: Sequence[str], values: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Average per-pair rows by study (sliding-window merge), first-seen order."""
    order = list(dict.fromkeys(study_keys))
    index = {key: i for i, key in enumerate(order)}
    idx = np.array([index[key] for key in study_keys])
    values = np.asarray(values, dtype=np.float64)
    sums = np.zeros((len(order),) + values.shape[1:])
    np.add.at(sums, idx, values)
    counts = np.bincount(idx, minlength=len(order)).reshape((-1,) + (1,) * (values.ndim - 1))
    return order, sums / counts


def image_values(
    study_keys: Sequence[str],
    pair_images: Sequence[Sequence[str]],
    view_values: np.ndarray,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """Per-image values from per-pair, per-slot values (no averaging across images).

    Args:
        study_keys: [P] study of each evaluation pair, loader order.
        pair_images: [P] the two image names of each pair, slot order.
        view_values: [P, 2, ...] value of each slot, e.g. the frozen
            single-view outputs y1, y2.

    Returns the distinct images as (study_key, image_name) and their values
    [I, ...], in first-seen order: studies in loader order, the images of a
    study in file order (consecutive windows). An image shared by two windows
    keeps its first value (the frozen encoder is deterministic in eval mode,
    so both windows give it the same value); a self-pair [a, a] gives one row.
    """
    index: Dict[Tuple[str, str], int] = {}
    rows = []
    for p, (key, names) in enumerate(zip(study_keys, pair_images)):
        for slot, name in enumerate(names):
            if (key, name) not in index:
                index[(key, name)] = len(rows)
                rows.append(view_values[p, slot])
    return list(index), np.stack(rows)


def aggregate_study_preds(
    results: List[dict],
    num_labels: int = NUM_LABELS,
) -> pd.DataFrame:
    """Average probabilities per study (sliding-window merge)."""
    prob_cols = [f"p_{i}" for i in range(num_labels)]
    keys, probs = mean_by_study(
        [r["study_key"] for r in results], np.stack([r["probs"] for r in results])
    )
    final = pd.DataFrame(probs, columns=prob_cols)
    final.insert(0, "study_key", keys)
    return final


def predict_images(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_xla: bool = False,
) -> Dict[str, Any]:
    """Per-image predictions of a TripleBranchEVA over an evaluation loader.

    The unit of evaluation is the image; every image carries its study's
    labels. From one forward pass:

    - ``probs`` [I, C]: the deployed study prediction assigned to every image
      of the study (sigmoid of Eq. (1) for a study with two images, the mean
      of the window sigmoids for n > 2, sigmoid(y1) for a single-image study
      when the model routes it);
    - ``single_probs`` [I, C]: the frozen single-view prediction sigmoid(y_k)
      of each image on its own (no averaging over the images of a study).

    Also returns ``study_keys`` and ``image_names`` (lists, [I]), ``slots``
    [I] (1-based position of the image in its study, file order),
    ``labels`` [I, C] (the study labels; None if the loader has none) and
    ``paired`` [I] bool (the image belongs to a study with >= 2 images).
    """
    results = run_inference(model, loader, device, use_xla=use_xla, with_single_view=True)
    pair_keys = [r["study_key"] for r in results]
    study_keys, study_probs = mean_by_study(pair_keys, np.stack([r["probs"] for r in results]))
    _, study_single = mean_by_study(pair_keys, np.array([r["single"] for r in results], dtype=float))
    images, single_probs = image_values(
        pair_keys, [r["images"] for r in results], np.stack([r["view_probs"] for r in results])
    )
    study_index = {key: i for i, key in enumerate(study_keys)}
    idx = np.array([study_index[key] for key, _ in images])
    labels: Optional[np.ndarray] = None
    if "labels" in results[0]:
        _, study_label = mean_by_study(pair_keys, np.stack([r["labels"] for r in results]))
        labels = study_label[idx]
    slots, seen = [], {}
    for key, _ in images:
        seen[key] = seen.get(key, 0) + 1
        slots.append(seen[key])
    return {
        "study_keys": [key for key, _ in images],
        "image_names": [name for _, name in images],
        "slots": np.array(slots),
        "probs": study_probs[idx],
        "single_probs": single_probs,
        "labels": labels,
        "paired": study_single[idx] == 0,
    }


def build_submission(
    test_dir: str,
    final_study_preds: pd.DataFrame,
    output_path: str = "submission.csv",
    label_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build the per-image submission CSV from study-level predictions.

    Every image of a study receives the study's prediction (a single-image
    study's one image receives the single-view route output).
    """
    label_columns = label_columns or LABEL_COLUMNS
    prob_cols = [f"p_{i}" for i in range(len(label_columns))]
    all_files = sorted(f for f in os.listdir(test_dir) if is_test_image(f))
    pred_lookup = final_study_preds.set_index("study_key")[prob_cols].to_dict("index")
    submission_rows = []
    for img_name in all_files:
        study_probs = pred_lookup[test_study_key(img_name)]
        submission_rows.append([img_name] + [study_probs[c] for c in prob_cols])
    submission_df = pd.DataFrame(
        submission_rows, columns=["Image_name"] + label_columns
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return submission_df
