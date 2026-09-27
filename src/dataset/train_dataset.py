"""Training datasets for chest X-ray studies (Phase-2 pairs, Phase-1 single images)."""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import LABEL_COLUMNS

from .study import STUDY_COLUMNS, load_rgb_image, sample_training_pair, study_labels


class MultiViewXRayDataset(Dataset):
    """Phase-2 training dataset: one random image pair per (Patient_ID, Study).

    Studies with n >= 2 images yield two distinct images drawn uniformly at
    random, in random order; single-image studies yield a self-pair.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform,
        num_views: int = 2,
        label_columns: Optional[List[str]] = None,
    ) -> None:
        if num_views != 2:
            raise ValueError(f"Only image pairs are supported (num_views=2), got {num_views}")
        self.image_dir = image_dir
        self.transform = transform
        self.num_views = num_views
        self.label_columns = label_columns or LABEL_COLUMNS

        self.study_map = df.groupby(STUDY_COLUMNS)["Image_name"].apply(list).to_dict()
        self.study_keys = list(self.study_map.keys())
        self.df_labels = study_labels(df, self.label_columns)

    def __len__(self) -> int:
        return len(self.study_keys)

    def __getitem__(self, idx: int):
        key = self.study_keys[idx]
        image_names = self.study_map[key]
        labels = torch.from_numpy(
            self.df_labels.loc[key].to_numpy(dtype=np.float32).copy()
        )

        tensor_stack = []
        for name in sample_training_pair(image_names):
            img = load_rgb_image(os.path.join(self.image_dir, str(name)))
            tensor_stack.append(self.transform(img))

        images = torch.stack(tensor_stack, dim=0)
        return images, labels


class SingleViewXRayDataset(Dataset):
    """Phase-1 dataset: one radiograph per item with its own row labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform,
        label_columns: Optional[List[str]] = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.label_columns = label_columns or LABEL_COLUMNS
        self.image_names = df["Image_name"].astype(str).tolist()
        self.labels = df[self.label_columns].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img = load_rgb_image(os.path.join(self.image_dir, self.image_names[idx]))
        return self.transform(img), torch.from_numpy(self.labels[idx].copy())
