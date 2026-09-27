"""Per-label ROC-AUC with uncertain-label masking."""

from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_label_mask(labels: np.ndarray) -> np.ndarray:
    """True where a label is a usable target (0 or 1); False for -1 (uncertain) or NaN."""
    labels = np.asarray(labels)
    return (labels == 0) | (labels == 1)


def label_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC of one label over rows with y in {0, 1}; NaN if one class remains."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = binary_label_mask(y_true)
    y_true, y_score = y_true[mask], y_score[mask]
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def per_label_auc(labels: np.ndarray, scores: np.ndarray) -> List[float]:
    """Per-label AUC for [N, C] labels (values in {0, 1, -1}) and [N, C] scores."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    return [label_auc(labels[:, c], scores[:, c]) for c in range(labels.shape[1])]


def macro_auc(per_label: List[float]) -> float:
    """Mean of the per-label AUCs, ignoring labels without both classes."""
    return float(np.nanmean(per_label))
