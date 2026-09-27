"""Label CSV reading and the persisted 90/10 patient split."""

import os
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.constants import LABEL_COLUMNS


def read_train_csv(path: str, label_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a Grand X-Ray SLAM train CSV; blank labels -> 0, labels must be 0/1."""
    label_columns = label_columns or LABEL_COLUMNS
    df = pd.read_csv(path)
    df[label_columns] = (
        df[label_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    )
    if not df[label_columns].isin([0.0, 1.0]).all().all():
        raise ValueError(f"{path}: labels must be 0/1 after filling blanks with 0")
    return df


def make_patient_split(
    patient_ids: Iterable, val_frac: float = 0.1, seed: int = 1337
) -> pd.DataFrame:
    """Assign patients to train/val (Patient_ID, split).

    Patients are taken in first-appearance order and shuffled with
    ``np.random.RandomState(seed)``; the last ``val_frac`` become validation.
    This matches the original training script only for the same CSV in the
    same row order, so the split is persisted once and then only read.
    """
    patients = pd.Series(list(patient_ids)).astype(str).unique()
    np.random.RandomState(seed).shuffle(patients)
    split_idx = int(len(patients) * (1 - val_frac))
    split = np.where(np.arange(len(patients)) < split_idx, "train", "val")
    return pd.DataFrame({"Patient_ID": patients, "split": split})


def load_patient_split(
    split_csv: str,
    df: Optional[pd.DataFrame] = None,
    val_frac: float = 0.1,
    seed: int = 1337,
) -> pd.DataFrame:
    """Read the persisted split; create it from ``df`` on first use.

    The file (columns Patient_ID, split in {train, val}) lives under
    ``configs/`` so that both phases use the same patients.
    """
    if os.path.exists(split_csv):
        return pd.read_csv(split_csv, dtype={"Patient_ID": str, "split": str})
    if df is None:
        raise FileNotFoundError(f"Split file {split_csv} not found and no data to create it")
    split = make_patient_split(df["Patient_ID"], val_frac=val_frac, seed=seed)
    Path(split_csv).parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_csv, index=False)
    print(f"Created patient split {split_csv} (seed {seed}, val_frac {val_frac})")
    return split


def split_by_patient(
    df: pd.DataFrame, split_csv: str, val_frac: float = 0.1, seed: int = 1337
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train_df, val_df) with the persisted patient split."""
    split = load_patient_split(split_csv, df, val_frac=val_frac, seed=seed)
    patient_ids = df["Patient_ID"].astype(str)
    unknown = set(patient_ids) - set(split["Patient_ID"])
    if unknown:
        raise ValueError(f"{len(unknown)} patients are missing from {split_csv}")
    val_ids = set(split.loc[split["split"] == "val", "Patient_ID"])
    is_val = patient_ids.isin(val_ids)
    train_df = df[~is_val].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)
    return train_df, val_df


def val_patients_from_splits(split_csvs: Iterable[str]) -> Set[str]:
    """Union of the validation patients of several divisions.

    In the full-data regime these patients are removed from the merged
    Phase-1 training set.
    """
    val_ids: Set[str] = set()
    for split_csv in split_csvs:
        split = load_patient_split(split_csv)
        val_ids |= set(split.loc[split["split"] == "val", "Patient_ID"])
    return val_ids
