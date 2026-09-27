"""Evaluation / inference datasets for multi-view chest X-ray studies."""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import LABEL_COLUMNS

from .study import (
    STUDY_COLUMNS,
    group_test_images,
    is_test_image,
    load_rgb_image,
    study_labels,
    study_pairs,
)


class MultiViewEvalDataset(Dataset):
    """Deterministic study-level evaluation samples (no RNG).

    Each study (key, image paths in slot/file order) is expanded into the
    pairs of ``study_pairs``: a self-pair flagged ``single`` for one image,
    else the windows of consecutive images. Items are dicts with ``images``
    [2, C, H, W], ``image_names`` (the two names in slot order; a batch holds
    one sequence per slot), ``study_key`` (str), ``single`` (bool) and, when
    labels are given, ``labels`` [num_labels] (values in {0, 1}, or -1 for
    uncertain).
    """

    def __init__(
        self,
        studies: Sequence[Tuple[str, Sequence[str]]],
        image_dir: str,
        transform,
        labels: Optional[np.ndarray] = None,
        num_views: int = 2,
    ) -> None:
        if num_views != 2:
            raise ValueError(f"Only image pairs are supported (num_views=2), got {num_views}")
        keys = [str(key) for key, _ in studies]
        if len(set(keys)) != len(keys):
            raise ValueError("Study keys must be unique")
        if labels is not None and len(labels) != len(studies):
            raise ValueError("labels must have one row per study")
        self.image_dir = image_dir
        self.transform = transform
        self.num_views = num_views
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

        self.samples: List[Dict[str, Any]] = []
        for study_idx, (key, images) in enumerate(studies):
            images = list(images)
            for pair in study_pairs(images):
                self.samples.append(
                    {
                        "study_key": str(key),
                        "study_idx": study_idx,
                        "images": pair,
                        "single": len(images) == 1,
                    }
                )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        image_dir: str,
        transform,
        num_views: int = 2,
        label_columns: Optional[List[str]] = None,
    ) -> "MultiViewEvalDataset":
        """Validation studies from a train-format CSV (images in file-name order)."""
        label_columns = label_columns or LABEL_COLUMNS
        study_map = df.groupby(STUDY_COLUMNS)["Image_name"].apply(list).to_dict()
        labels_df = study_labels(df, label_columns)
        keys = list(study_map.keys())
        studies = [
            (f"{pid}_{study}", sorted(str(name) for name in study_map[(pid, study)]))
            for pid, study in keys
        ]
        labels = labels_df.loc[keys].to_numpy(dtype=np.float32)
        return cls(studies, image_dir, transform, labels=labels, num_views=num_views)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        tensor_stack = [
            self.transform(load_rgb_image(os.path.join(self.image_dir, name)))
            for name in sample["images"]
        ]
        item = {
            "images": torch.stack(tensor_stack, dim=0),
            "image_names": list(sample["images"]),
            "study_key": sample["study_key"],
            "single": sample["single"],
        }
        if self.labels is not None:
            item["labels"] = torch.from_numpy(self.labels[sample["study_idx"]].copy())
        return item


class TestSingleViewDataset(Dataset):
    """Every image of a test directory (jpg / png) on its own, in file-name order.

    Used for single-view submissions: items are dicts with ``image`` [C, H, W]
    and ``image_name`` (str); no pairing, no study grouping.
    """

    def __init__(self, root_dir: str, transform) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = sorted(f for f in os.listdir(root_dir) if is_test_image(f))

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        name = self.image_names[idx]
        image = self.transform(load_rgb_image(os.path.join(self.root_dir, name)))
        return {"image": image, "image_name": name}


class TestMultiViewDataset(MultiViewEvalDataset):
    """Test dataset that discovers studies from a directory of images (jpg / png)."""

    def __init__(self, root_dir: str, transform, num_views: int = 2) -> None:
        self.root_dir = root_dir
        all_files = [f for f in os.listdir(root_dir) if is_test_image(f)]
        self.study_map = group_test_images(all_files)
        super().__init__(
            list(self.study_map.items()), root_dir, transform, num_views=num_views
        )
