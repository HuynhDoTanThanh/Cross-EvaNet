#!/usr/bin/env python3
"""
Build the CheXpert zero-shot evaluation cohort (paper Sec. 3.2).

Protocol:
  - images are grouped by patient and study, parsed from ``Path``
    (``.../patient00001/study1/view1_frontal.jpg``);
  - a study qualifies if it has at least one frontal and at least one lateral
    image (``Frontal/Lateral`` column; AP and PA both count as frontal);
  - one study per patient: the earliest qualifying one, i.e. the lowest study
    index (the public metadata has no dates);
  - within that study the first frontal and the first lateral image in metadata
    order (CSV order, files in the order given) form the pair; the two selected
    images keep their metadata order in the slots (``image_1``, ``image_2``),
    matching file order at inference: projection is used only to select the
    pair, never to assign slots;
  - labels: blank -> 0, 1 / 0 as-is, uncertain -1 kept as -1 (excluded per
    label by scripts/statistics.py and src.metrics).

The output study list (study_key, patient_id, study_index, image_1, image_2,
14 label columns) is the input of scripts/evaluate_chexpert.py. It contains
CheXpert labels, so it stays local (``outputs/`` is git-ignored);
``--ids-output`` writes the label-free list (ids and image paths) that can be
published, and anyone with CheXpert access regenerates the labelled list with
this script.

Usage:
  python -m scripts.build_chexpert_cohort \
    --csv /data/CheXpert-v1.0-small/train.csv \
    --output outputs/chexpert_cohort.csv --ids-output configs/chexpert_cohort_ids.csv
"""

import argparse
import re
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.constants import LABEL_COLUMNS

VIEW_COLUMN = "Frontal/Lateral"
PATH_PATTERN = re.compile(r"(patient\d+)/study(\d+)/")
STUDY_KEYS = ["patient_id", "study_index"]
ID_COLUMNS = ["study_key", "patient_id", "study_index", "image_1", "image_2"]
PAPER_N_STUDIES = 19539


def parse_args():
    p = argparse.ArgumentParser(description="Build the paired frontal/lateral CheXpert cohort")
    p.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Official CheXpert metadata CSV (Path, Frontal/Lateral, 14 labels); "
        "repeat to concatenate several files in metadata order",
    )
    p.add_argument("--output", default="outputs/chexpert_cohort.csv", help="Study list CSV to write (with labels)")
    p.add_argument(
        "--ids-output",
        default=None,
        help="Optional label-free study list (ids and image paths) for publishing, "
        "e.g. configs/chexpert_cohort_ids.csv",
    )
    return p.parse_args()


def read_metadata(csv_paths: List[str]) -> pd.DataFrame:
    """Concatenate CheXpert metadata CSVs in order; parse patient / study; map labels."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        missing = [c for c in ["Path", VIEW_COLUMN] + LABEL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    parts = df["Path"].str.extract(PATH_PATTERN)
    if parts.isna().any().any():
        bad = df.loc[parts.isna().any(axis=1), "Path"].iloc[0]
        raise ValueError(f"Cannot parse patient/study from Path, e.g. {bad!r}")
    df["patient_id"] = parts[0]
    df["study_index"] = parts[1].astype(int)
    df["view"] = df[VIEW_COLUMN].astype(str).str.strip().str.lower()

    labels = df[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    if not labels.isin([0, 1, -1]).all().all():
        raise ValueError("CheXpert labels must be 1, 0, -1 or blank")
    df[LABEL_COLUMNS] = labels.astype(int)
    return df


def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """One frontal + lateral pair per patient from its earliest qualifying study."""
    first_frontal = df[df["view"] == "frontal"].groupby(STUDY_KEYS, sort=False).head(1)
    first_lateral = df[df["view"] == "lateral"].groupby(STUDY_KEYS, sort=False).head(1)
    lateral = first_lateral.assign(row_lateral=first_lateral.index).set_index(STUDY_KEYS)
    # Inner join keeps the studies with at least one image of each orientation.
    pairs = first_frontal.join(
        lateral[["Path", "row_lateral"] + LABEL_COLUMNS], on=STUDY_KEYS, how="inner", rsuffix="_lateral"
    )
    earliest = pairs.groupby("patient_id", sort=False)["study_index"].transform("min")
    cohort = pairs[pairs["study_index"] == earliest]

    lateral_labels = cohort[[f"{c}_lateral" for c in LABEL_COLUMNS]].to_numpy()
    mismatch = (cohort[LABEL_COLUMNS].to_numpy() != lateral_labels).any(axis=1)
    if mismatch.any():
        warnings.warn(
            f"{int(mismatch.sum())} studies have different labels on the frontal and "
            "lateral rows; the frontal row's labels are used."
        )

    cohort = cohort.sort_index()  # metadata order of the frontal image
    # Slots follow metadata order (the index is the metadata row of the frontal image).
    lateral_first = cohort["row_lateral"].to_numpy() < cohort.index.to_numpy()
    out = pd.DataFrame(
        {
            "study_key": cohort["patient_id"] + "/study" + cohort["study_index"].astype(str),
            "patient_id": cohort["patient_id"],
            "study_index": cohort["study_index"],
            "image_1": np.where(lateral_first, cohort["Path_lateral"], cohort["Path"]),
            "image_2": np.where(lateral_first, cohort["Path"], cohort["Path_lateral"]),
        }
    )
    out[LABEL_COLUMNS] = cohort[LABEL_COLUMNS]
    return out.reset_index(drop=True)


def main():
    args = parse_args()
    df = read_metadata(args.csv)
    cohort = build_cohort(df)
    if cohort["patient_id"].duplicated().any():
        raise RuntimeError("More than one study selected for a patient")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    if args.ids_output:
        Path(args.ids_output).parent.mkdir(parents=True, exist_ok=True)
        cohort[ID_COLUMNS].to_csv(args.ids_output, index=False)

    n_studies = len(cohort)
    print(
        f"Metadata: {len(df)} images, {df['patient_id'].nunique()} patients, "
        f"{len(df.drop_duplicates(STUDY_KEYS))} studies"
    )
    print(
        f"Cohort: {n_studies} studies from {n_studies} patients, {2 * n_studies} images "
        f"(paper: {PAPER_N_STUDIES} studies) -> {args.output}"
    )
    if args.ids_output:
        print(f"Label-free study list -> {args.ids_output}")
    counts = pd.DataFrame(
        {v: (cohort[LABEL_COLUMNS] == v).sum() for v in (1, 0, -1)}
    ).rename(columns={1: "positive", 0: "negative", -1: "uncertain (excluded)"})
    print(counts.to_string())


if __name__ == "__main__":
    main()
