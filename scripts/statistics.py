#!/usr/bin/env python3
"""
Paired statistical comparison of Cross-EvaNet against its single-view comparator
(paper Sec. 5.2). The unit of evaluation is the image (radiograph), as on the
Grand X-Ray SLAM leaderboard.

Input: the per-image CSV of scripts/evaluate_chexpert.py, one row per image
with study_id, patient_id and, for each label L, label_<L> (the study's
labels), single_<L> (the frozen single-view encoder applied to that image
alone) and fused_<L> (the Cross-EvaNet study prediction, Eq. (1), identical
for all images of the study). Scores may be logits or probabilities (AUC is
rank based). The fused scores and labels must be constant within a study. A
per-study file (``--per-study``) also works; each row is then a study.

Protocol:
  - per-label AUC over the image rows; labels other than 0/1 (uncertain -1,
    NaN) are excluded per label from the AUC, the bootstrap and DeLong (the
    images of a study share its labels);
  - 95% CIs: percentile bootstrap, B = 2000 resamples (seed 42), resampling
    studies as clusters: a drawn study brings all of its images. ``--resample
    patient`` resamples patients instead (an extra, not used in the paper). The
    same replicates are used for both arms, every label and the aggregates; a
    replicate where a label has only one class is dropped for that label and
    for the aggregates containing it;
  - per label: two-sided paired DeLong test on the image rows (fast algorithm
    of Sun & Xu, 2014), z = DeLong statistic. DeLong treats the images as
    independent; for comparison the raw output also has the two-sided
    cluster-bootstrap p-value of each label (``p_boot``, with ``z_boot`` =
    delta / SE_boot);
  - Macro average (all labels) and CheXpert Core 5 (Atelectasis, Cardiomegaly,
    Consolidation, Edema, Pleural Effusion): z = delta / SE_boot and the
    empirical paired cluster-bootstrap p = 2 * min(P(delta* <= 0), P(delta* >= 0)).

The output CSV has the column layout of the paper's Table 8, followed by the
counts (evaluable images, studies, positive and negative images per label; all
images and studies on the aggregate rows). ``--raw-output`` also writes the
unformatted numbers, including Holm-adjusted DeLong p-values (``p_holm``, not
reported in the paper), ``p_boot`` / ``z_boot`` and the number of resampling
clusters.

Needs numpy, scipy, pandas and scikit-learn (through src.metrics), not torch.
Clusters are numbered in sorted-id order, so the replicates do not depend on
the row order of the input. Run it as a module from the repository root.

Usage:
  python -m scripts.statistics --predictions outputs/chexpert_zeroshot_logits.csv \
    --output outputs/chexpert_statistics.csv [--raw-output outputs/chexpert_statistics_raw.csv]

  python -m scripts.statistics --predictions outputs/divB_val_logits.csv \
    --output outputs/divB_val_statistics.csv

  python -m scripts.statistics --selftest
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from src.metrics import binary_label_mask

CORE5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
ARMS = ("single", "fused")
TABLE_COLUMNS = [
    "Pathology",
    "EVA-X (Single-View) AUC (95% CI)",
    "Cross-EvaNet (Ours) AUC (95% CI)",
    "Delta AUC (95% CI)",
    "z-score",
    "p-value",
    "Significance",
]
CORE5_ROW = "CheXpert Core 5 Benchmark Average"
COUNT_COLUMNS = ["N images", "N studies", "N positive", "N negative"]
# Study identifier of a row (per-image and --per-study files).
STUDY_COLUMN = "study_id"
BOOT_CHUNK = 100
# Standard errors at or below this are treated as 0 (floating-point noise when
# every replicate gives the same delta).
SE_TOL = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description="Bootstrap CIs and paired DeLong for fused vs single-view")
    p.add_argument("--predictions", help="Per-image CSV from scripts/evaluate_chexpert.py")
    p.add_argument("--output", help="Formatted table CSV to write")
    p.add_argument("--raw-output", help="Optional CSV of the unformatted numbers")
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resample",
        choices=["study", "patient"],
        default="study",
        help="Bootstrap cluster: study (paper; column study_id, all images of a study together) or "
        "patient (column patient_id; extra, not used in the paper)",
    )
    p.add_argument("--selftest", action="store_true", help="Run the self-check on synthetic data and exit")
    args = p.parse_args()
    if not args.selftest and not (args.predictions and args.output):
        p.error("--predictions and --output are required (or use --selftest)")
    return args


# ---------------------------------------------------------------------------
# AUC, DeLong, Holm
# ---------------------------------------------------------------------------


def _sorted_groups(score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort order of ``score`` and the start of each block of tied scores."""
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    starts = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
    return order, starts


def weighted_auc(positive: np.ndarray, weights: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """ROC-AUC for each row of case weights, ties counted 1/2.

    Args:
        positive: [n] bool, cases sorted by score.
        weights: [R, n] multiplicities in the same order (bootstrap counts;
            ones for the plain AUC).
        starts: start index of each block of tied scores.

    Returns [R] AUCs; NaN where a row has no weighted positive or negative.
    """
    w_pos = weights * positive
    w_neg = weights - w_pos
    pos = np.add.reduceat(w_pos, starts, axis=1)
    neg = np.add.reduceat(w_neg, starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    num = (pos * (neg_below + 0.5 * neg)).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def delong_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> Tuple[float, float, float, float]:
    """Two-sided paired DeLong test of AUC(b) - AUC(a) (fast algorithm, Sun & Xu 2014).

    ``y`` holds 0/1 targets only. Returns (auc_a, auc_b, z, p); z and p are
    NaN with fewer than two positives or negatives.
    """
    positive = y == 1
    m, n = int(positive.sum()), int((~positive).sum())
    scores = np.vstack([score_a, score_b]).astype(np.float64)
    x, z_neg = scores[:, positive], scores[:, ~positive]
    tx = rankdata(x, axis=1)
    ty = rankdata(z_neg, axis=1)
    tz = rankdata(np.hstack([x, z_neg]), axis=1)
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v10 = (tz[:, :m] - tx) / n  # per positive: share of negatives ranked below
    v01 = 1.0 - (tz[:, m:] - ty) / m  # per negative: share of positives ranked above
    if m < 2 or n < 2:
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    cov = np.cov(v10) / m + np.cov(v01) / n
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    delta = aucs[1] - aucs[0]
    if var <= SE_TOL**2:
        z = 0.0 if delta == 0 else float(np.sign(delta) * np.inf)
    else:
        z = float(delta / np.sqrt(var))
    p = float(2.0 * norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), z, p


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN entries are left out of the family."""
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full_like(p, np.nan)
    tested = ~np.isnan(p)
    pv = p[tested]
    order = np.argsort(pv, kind="mergesort")
    steps = (pv.size - np.arange(pv.size)) * pv[order]
    result = np.empty_like(pv)
    result[order] = np.minimum(1.0, np.maximum.accumulate(steps))
    adjusted[tested] = result
    return adjusted


def bootstrap_counts(units: np.ndarray, n_boot: int, seed: int):
    """Yield [R, n_cases] cluster-bootstrap multiplicities in chunks of replicates.

    ``units`` [n_cases] maps each case (image) to its cluster (study or
    patient, 0..U-1); clusters are drawn uniformly with replacement, U of
    them, and every case inherits the count of its cluster, so the images of
    a drawn cluster are resampled together.
    """
    rng = np.random.default_rng(seed)
    n_units = int(units.max()) + 1
    for start in range(0, n_boot, BOOT_CHUNK):
        rows = min(BOOT_CHUNK, n_boot - start)
        counts = np.stack(
            [np.bincount(rng.integers(0, n_units, n_units), minlength=n_units) for _ in range(rows)]
        )
        yield start, counts[:, units].astype(np.float64)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _ci(values: np.ndarray) -> Tuple[float, float]:
    """95% percentile interval of the finite replicates."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_test(delta: float, delta_boot: np.ndarray, se: float) -> Tuple[float, float]:
    """z = delta / SE_boot and the two-sided p = 2 * min(P(delta* <= 0), P(delta* >= 0))."""
    delta_boot = delta_boot[np.isfinite(delta_boot)]
    if not np.isfinite(se):
        return float("nan"), float("nan")
    if se > SE_TOL:
        z = delta / se
    else:
        z = 0.0 if delta == 0 else float(np.sign(delta) * np.inf)
    p = min(1.0, 2.0 * min(np.mean(delta_boot <= 0), np.mean(delta_boot >= 0)))
    return float(z), float(p)


def _summary(name: str, point: np.ndarray, boot: np.ndarray) -> Dict[str, object]:
    """Point AUCs, CIs, delta and bootstrap test for one row; ``point`` [2], ``boot`` [B, 2] (single, fused)."""
    delta_boot = boot[:, 1] - boot[:, 0]
    row: Dict[str, object] = {"row": name}
    for a, arm in enumerate(ARMS):
        row[f"auc_{arm}"] = float(point[a])
        row[f"auc_{arm}_lo"], row[f"auc_{arm}_hi"] = _ci(boot[:, a])
    row["delta"] = float(point[1] - point[0])
    row["delta_lo"], row["delta_hi"] = _ci(delta_boot)
    valid = np.isfinite(delta_boot)
    row["n_boot_valid"] = int(valid.sum())
    row["se_boot"] = float(np.std(delta_boot[valid], ddof=1)) if valid.sum() > 1 else float("nan")
    row["z_boot"], row["p_boot"] = _bootstrap_test(row["delta"], delta_boot, row["se_boot"])
    return row


def analyze(
    labels: np.ndarray,
    single: np.ndarray,
    fused: np.ndarray,
    label_names: Sequence[str],
    units: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    studies: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-label and aggregate statistics (one row each, unformatted).

    Args:
        labels: [N, C] targets of the N images (their study's labels); values
            other than 0/1 are excluded per label.
        single, fused: [N, C] scores of the comparator and of Cross-EvaNet.
        label_names: C names (Core 5 is looked up by name).
        units: [N] int bootstrap cluster of each image (0..U-1).
        studies: [N] int study of each image, for the counts (default: ``units``).
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.stack([np.asarray(single, np.float64), np.asarray(fused, np.float64)], axis=-1)
    units = np.asarray(units)
    studies = units if studies is None else np.asarray(studies)
    n_labels = labels.shape[1]

    # Per label and arm: image rows with a 0/1 target, sorted by score.
    prepared = []
    point = np.full((n_labels, 2), np.nan)
    valid_idx, counts = [], []
    for c in range(n_labels):
        mask = binary_label_mask(labels[:, c])
        idx = np.flatnonzero(mask)
        y = labels[idx, c]
        valid_idx.append(idx)
        counts.append(
            {
                "n_images": int(idx.size),
                "n_studies": int(np.unique(studies[idx]).size),
                "n_clusters": int(np.unique(units[idx]).size),
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
                "n_excluded": int((~mask).sum()),
            }
        )
        if idx.size == 0:  # no 0/1 target: the AUC stays NaN and the label is reported as n/a
            continue
        for a in range(2):
            order, starts = _sorted_groups(scores[idx, c, a])
            positive = y[order] == 1
            prepared.append((c, a, idx[order], positive, starts))
            point[c, a] = weighted_auc(positive, np.ones((1, idx.size)), starts)[0]

    boot = np.full((n_boot, n_labels, 2), np.nan)
    for start, weights in bootstrap_counts(units, n_boot, seed):
        for c, a, idx, positive, starts in prepared:
            boot[start : start + len(weights), c, a] = weighted_auc(positive, weights[:, idx], starts)

    defined = [c for c in range(n_labels) if np.isfinite(point[c]).all()]
    for c in range(n_labels):
        if c not in defined:
            warnings.warn(f"{label_names[c]}: needs both classes; left out of all statistics")

    totals = {
        "n_images": int(len(labels)),
        "n_studies": int(np.unique(studies).size),
        "n_clusters": int(np.unique(units).size),
    }
    rows: List[Dict[str, object]] = []
    core5 = [c for c in defined if label_names[c] in CORE5]
    aggregates = [(f"Macro Average (All {len(defined)} Pathologies)", defined)]
    if core5:
        if len(core5) < len(CORE5):
            warnings.warn(f"Core 5 average uses {len(core5)} of {len(CORE5)} labels")
        aggregates.append((CORE5_ROW, core5))
    for name, members in aggregates:
        # Mean over the member labels; NaN if one of them is undefined in a replicate.
        row = _summary(name, point[members].mean(axis=0), boot[:, members].mean(axis=1))
        row.update(test="paired bootstrap", n_labels=len(members), z=row["z_boot"], p=row["p_boot"])
        row.update(totals, p_holm=float("nan"))
        rows.append(row)

    label_rows = []
    for c in range(n_labels):
        row = {"row": label_names[c], "test": "DeLong", "n_labels": 1, "z": np.nan, "p": np.nan}
        if c in defined:
            row.update(_summary(label_names[c], point[c], boot[:, c]))
            idx = valid_idx[c]
            _, _, row["z"], row["p"] = delong_test(labels[idx, c], scores[idx, c, 0], scores[idx, c, 1])
        row.update(counts[c])
        label_rows.append(row)
    holm = holm_adjust([row["p"] for row in label_rows])
    for row, p_holm in zip(label_rows, holm):
        row["p_holm"] = float(p_holm)
    rows.extend(label_rows)

    columns = [
        "row", "test", "n_labels", "n_images", "n_studies", "n_clusters", "n_pos", "n_neg", "n_excluded",
        "auc_single", "auc_single_lo", "auc_single_hi",
        "auc_fused", "auc_fused_lo", "auc_fused_hi",
        "delta", "delta_lo", "delta_hi", "se_boot", "z", "p", "p_holm", "z_boot", "p_boot", "n_boot_valid",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


# ---------------------------------------------------------------------------
# Formatting and I/O
# ---------------------------------------------------------------------------


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _fmt_count(value: float) -> str:
    return str(int(value)) if np.isfinite(value) else "-"


def format_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Paper-table layout: 'AUC [lo, hi]', signed delta, z, p, stars; then the counts."""
    out = []
    for _, r in raw.iterrows():
        counts = [_fmt_count(r[k]) for k in ("n_images", "n_studies", "n_pos", "n_neg")]
        if not np.isfinite(r["auc_single"]):
            out.append([r["row"]] + ["n/a"] * (len(TABLE_COLUMNS) - 1) + counts)
            continue
        out.append(
            [
                r["row"],
                f"{r['auc_single']:.4f} [{r['auc_single_lo']:.4f}, {r['auc_single_hi']:.4f}]",
                f"{r['auc_fused']:.4f} [{r['auc_fused_lo']:.4f}, {r['auc_fused_hi']:.4f}]",
                f"{r['delta']:+.4f} [{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}]",
                f"{r['z']:.2f}" if not np.isnan(r["z"]) else "n/a",
                _fmt_p(r["p"]),
                _stars(r["p"]),
            ]
            + counts
        )
    return pd.DataFrame(out, columns=TABLE_COLUMNS + COUNT_COLUMNS)


def load_predictions(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Labels, single and fused scores [N, C] of the N rows (images), label names and the table."""
    df = pd.read_csv(path, dtype={c: str for c in (STUDY_COLUMN, "patient_id", "image")})
    names = [c[len("label_"):] for c in df.columns if c.startswith("label_")]
    missing = [f"{arm}_{n}" for n in names for arm in ARMS if f"{arm}_{n}" not in df.columns]
    if not names or missing:
        raise ValueError(f"{path}: need label_<L>, single_<L> and fused_<L> columns (missing {missing})")
    if STUDY_COLUMN in df.columns:
        # One row per image of a study (per-image file) or per study (per-study file).
        key = [STUDY_COLUMN, "image"] if "image" in df.columns else [STUDY_COLUMN]
        if df.duplicated(key).any():
            raise ValueError(f"{path}: duplicated rows for {key}")
        if "image" in df.columns:
            shared = [c for c in df.columns if c.startswith(("fused_", "label_"))]
            if df.groupby(STUDY_COLUMN)[shared].nunique(dropna=False).gt(1).any().any():
                raise ValueError(f"{path}: fused scores and labels must be constant within a study")
    labels = df[[f"label_{n}" for n in names]].to_numpy(dtype=np.float64)
    single = df[[f"single_{n}" for n in names]].to_numpy(dtype=np.float64)
    fused = df[[f"fused_{n}" for n in names]].to_numpy(dtype=np.float64)
    if not (np.isfinite(single).all() and np.isfinite(fused).all()):
        raise ValueError(f"{path}: scores must be finite")
    return labels, single, fused, names, df


def study_ids(df: pd.DataFrame) -> np.ndarray:
    """Study index of each row (0..S-1, sorted study_id order).

    Without a study_id column every row is taken as its own study.
    """
    if STUDY_COLUMN in df.columns:
        return pd.factorize(df[STUDY_COLUMN].astype(str), sort=True)[0]
    warnings.warn("no study_id column: every row is treated as its own study")
    return np.arange(len(df))


def resampling_units(df: pd.DataFrame, level: str) -> np.ndarray:
    """Bootstrap cluster of each row (image): its study, or its patient (sorted-id order)."""
    if level == "study":
        return study_ids(df)
    if "patient_id" not in df.columns:
        raise ValueError("--resample patient needs a patient_id column")
    return pd.factorize(df["patient_id"].astype(str), sort=True)[0]


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _delong_bruteforce(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """O(m n) DeLong from the pairwise kernel (reference for the self-check)."""
    pos, neg = y == 1, y == 0
    v10, v01, aucs = [], [], []
    for s in (a, b):
        psi = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
        v10.append(psi.mean(axis=1))
        v01.append(psi.mean(axis=0))
        aucs.append(psi.mean())
    cov = np.cov(np.vstack(v10)) / pos.sum() + np.cov(np.vstack(v01)) / neg.sum()
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    return aucs[0], aucs[1], (aucs[1] - aucs[0]) / np.sqrt(var)


def _synthetic(n_studies: int, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic per-image file: studies of 1-3 images and repeated patients.

    The images of a study share its labels (with -1 entries) and one fused
    score; single-view scores are per image, correlated within a study and
    rounded to create ties. 14 labels, Core 5 by name.
    """
    names = CORE5 + [f"Other {i}" for i in range(9)]
    n_labels = len(names)
    prevalence = rng.uniform(0.08, 0.45, n_labels)
    y = (rng.random((n_studies, n_labels)) < prevalence).astype(np.float64)
    sizes = rng.choice([1, 2, 3], n_studies, p=[0.2, 0.6, 0.2])
    study = np.repeat(np.arange(n_studies), sizes)
    signal = y * rng.uniform(0.3, 1.5, n_labels)
    single = signal[study] + 0.7 * rng.normal(size=(n_studies, n_labels))[study]
    single = np.round(single + 0.7 * rng.normal(size=single.shape), 1)
    study_mean = np.zeros((n_studies, n_labels))
    np.add.at(study_mean, study, single)
    study_mean /= sizes[:, None]
    gain = np.linspace(-0.2, 0.8, n_labels)
    fused = study_mean + y * gain + 0.3 * rng.normal(size=y.shape)
    y[rng.random(y.shape) < 0.1] = -1
    patients = pd.factorize(rng.integers(0, n_studies // 2, n_studies))[0]
    slot = np.concatenate([np.arange(1, n + 1) for n in sizes])
    df = pd.DataFrame(
        {
            "study_id": [f"s{i}" for i in study],
            "patient_id": [f"p{patients[i]}" for i in study],
            "image": [f"s{i}_{k}.jpg" for i, k in zip(study, slot)],
            "slot": slot,
        }
    )
    columns = {}
    for prefix, values in (("label", y[study]), ("single", single), ("fused", fused[study])):
        columns.update({f"{prefix}_{name}": values[:, c] for c, name in enumerate(names)})
    return pd.concat([df, pd.DataFrame(columns)], axis=1)


def run_selftest() -> None:
    """Check AUC, DeLong, the cluster bootstrap and Holm on a synthetic per-image file."""
    import io

    from sklearn.metrics import roc_auc_score

    from src.metrics import label_auc

    rng = np.random.default_rng(0)
    frame = _synthetic(500, rng)
    y, single, fused, names, df = load_predictions(io.StringIO(frame.to_csv(index=False)))
    studies = resampling_units(df, "study")
    patients = resampling_units(df, "patient")
    assert len(df) > studies.max() + 1 > patients.max() + 1, "need multi-image studies and patients"
    assert pd.Series(patients).groupby(studies).nunique().eq(1).all(), "a study must belong to one patient"

    # 1. Per-image AUC (masked, with ties) matches sklearn / src.metrics on the image rows.
    for c in range(y.shape[1]):
        mask = binary_label_mask(y[:, c])
        for s in (single[:, c], fused[:, c]):
            order, starts = _sorted_groups(s[mask])
            auc = weighted_auc(y[mask, c][order] == 1, np.ones((1, mask.sum())), starts)[0]
            assert np.isclose(auc, roc_auc_score(y[mask, c], s[mask])), "AUC mismatch"
            assert np.isclose(auc, label_auc(y[:, c], s)), "AUC mismatch vs src.metrics"

    # 2. Fast DeLong on image rows matches the O(mn) reference.
    for c in range(y.shape[1]):
        mask = binary_label_mask(y[:, c])
        fast = delong_test(y[mask, c], single[mask, c], fused[mask, c])
        ref = _delong_bruteforce(y[mask, c], single[mask, c], fused[mask, c])
        assert np.allclose(fast[:3], ref), f"DeLong mismatch: {fast[:3]} vs {ref}"
    assert delong_test(y[y[:, 0] >= 0, 0], single[y[:, 0] >= 0, 0], single[y[:, 0] >= 0, 0])[3] == 1.0

    # 3. Cluster bootstrap: the images of a drawn study (or patient) come together,
    #    and a replicate equals the AUC of the explicitly resampled image rows.
    c = 1
    mask = binary_label_mask(y[:, c])
    idx = np.flatnonzero(mask)
    order, starts = _sorted_groups(fused[idx, c])
    for units in (studies, patients):
        _, weights = next(bootstrap_counts(units, 3, seed=7))
        first = pd.Series(np.arange(len(units))).groupby(units).first().to_numpy()
        assert np.array_equal(weights, weights[:, first][:, units]), "images of a cluster split"
        per_study_first = pd.Series(np.arange(len(studies))).groupby(studies).first().to_numpy()
        assert np.array_equal(weights, weights[:, per_study_first][:, studies]), "images of a study split"
        rng_ref = np.random.default_rng(7)
        n_units = units.max() + 1
        draw = rng_ref.integers(0, n_units, n_units)
        rows = np.concatenate([np.flatnonzero(units == u) for u in draw])
        assert weights[0].sum() == rows.size
        rows = rows[mask[rows]]
        expected = roc_auc_score(y[rows, c], fused[rows, c])
        got = weighted_auc(y[idx[order], c] == 1, weights[:1, idx[order]], starts)[0]
        assert np.isclose(got, expected), "bootstrap replicate mismatch"

    # 4. Holm on a textbook example.
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03, 0.005, np.nan])[:4], [0.03, 0.06, 0.06, 0.02])

    # 5. End to end: layout, determinism, counts, bootstrap p, recovered effects.
    raw = analyze(y, single, fused, names, studies, n_boot=300, seed=42)
    again = analyze(y, single, fused, names, studies, n_boot=300, seed=42)
    pd.testing.assert_frame_equal(raw, again)
    table = format_table(raw)
    assert list(table.columns) == TABLE_COLUMNS + COUNT_COLUMNS and len(table) == 2 + len(names)
    assert table["Pathology"].iloc[1] == CORE5_ROW
    n_studies = studies.max() + 1
    assert (raw["n_images"].iloc[:2] == len(df)).all() and (raw["n_studies"].iloc[:2] == n_studies).all()
    labels_raw = raw.iloc[2:].reset_index(drop=True)
    for c in range(len(names)):
        mask = binary_label_mask(y[:, c])
        r = labels_raw.iloc[c]
        assert r["n_pos"] + r["n_neg"] == r["n_images"] == mask.sum()
        assert r["n_excluded"] == (~mask).sum() and r["n_studies"] == np.unique(studies[mask]).size
    assert labels_raw["n_excluded"].gt(0).all()
    assert raw["p_boot"].between(0, 1).all() and np.allclose(raw["p"].iloc[:2], raw["p_boot"].iloc[:2])
    assert raw["p"].iloc[-1] < 0.001 and raw["delta"].iloc[-1] > 0, "strong gain not detected"
    assert raw["p_boot"].iloc[-1] < 0.01, "strong gain not detected by the cluster bootstrap"
    assert (raw["auc_single_lo"] <= raw["auc_single_hi"]).all() and (raw["delta_lo"] <= raw["delta_hi"]).all()
    by_patient = analyze(y, single, fused, names, patients, n_boot=300, seed=42, studies=studies)
    assert (by_patient["n_studies"].iloc[:2] == n_studies).all()
    assert (by_patient["n_clusters"].iloc[:2] == patients.max() + 1).all()
    null = analyze(y, single, single, names, studies, n_boot=200, seed=42)
    assert np.allclose(null["delta"], 0) and np.allclose(null["p"], 1.0) and np.allclose(null["p_boot"], 1.0)

    # 6. The replicates do not depend on the row order; a label without 0/1 targets is n/a.
    shuffled = frame.sample(frac=1.0, random_state=3)
    ys, ss, fs, _, df_s = load_predictions(io.StringIO(shuffled.to_csv(index=False)))
    again = analyze(ys, ss, fs, names, resampling_units(df_s, "study"), n_boot=300, seed=42)
    pd.testing.assert_frame_equal(raw, again)
    y_na = y.copy()
    y_na[:, -1] = -1
    assert np.isnan(analyze(y_na, single, fused, names, studies, n_boot=50)["auc_fused"].iloc[-1])

    # 7. Input checks: a repeated (study, image) row and a fused score that differs
    #    within a study are rejected; a per-study file (one row per study) makes
    #    every row its own cluster.
    changed = frame.copy()
    changed.loc[changed.index[changed["study_id"].duplicated()][0], f"fused_{names[0]}"] += 1.0
    for bad in (pd.concat([frame, frame.iloc[:1]]), changed):
        try:
            load_predictions(io.StringIO(bad.to_csv(index=False)))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid per-image file accepted")
    per_study = frame.drop_duplicates("study_id").drop(columns="image")
    *_, df_study = load_predictions(io.StringIO(per_study.to_csv(index=False)))
    assert np.unique(resampling_units(df_study, "study")).size == len(df_study)

    print(table.to_string(index=False))
    print(f"{len(df)} images, {n_studies} studies, {patients.max() + 1} patients")
    print("selftest passed")


def main():
    args = parse_args()
    if args.selftest:
        run_selftest()
        return
    labels, single, fused, names, df = load_predictions(args.predictions)
    studies = study_ids(df)
    units = resampling_units(df, args.resample)
    print(
        f"{len(df)} rows (images), {studies.max() + 1} studies, {units.max() + 1} {args.resample} "
        f"clusters, B = {args.n_boot}, seed = {args.seed}"
    )
    raw = analyze(labels, single, fused, names, units, n_boot=args.n_boot, seed=args.seed, studies=studies)
    short = raw.loc[raw["n_boot_valid"] < args.n_boot, "row"].tolist()
    if short:
        print(f"Replicates dropped (a label with one class) for: {short}")
    table = format_table(raw)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    if args.raw_output:
        raw.to_csv(args.raw_output, index=False)
    print(table.to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
