"""Study grouping, pairing and image-loading helpers shared by all datasets."""

import random
import warnings
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

STUDY_COLUMNS = ["Patient_ID", "Study"]
# File types discovered in a test directory (training images are listed by the CSV).
TEST_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_rgb_image(path: str) -> Image.Image:
    """Open an image as RGB; raise with the path if it cannot be read."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Unreadable image: {path}") from exc


def is_test_image(filename: str) -> bool:
    """True for image files of a test directory (``TEST_IMAGE_EXTENSIONS``)."""
    return filename.lower().endswith(TEST_IMAGE_EXTENSIONS)


def test_study_key(filename: str) -> str:
    """Study key of a test image named ``<patient>_<study>_*.jpg``."""
    parts = filename.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else filename


def group_test_images(filenames: Iterable[str]) -> Dict[str, List[str]]:
    """Group test image names by study key; images of a study in file order."""
    study_map: Dict[str, List[str]] = {}
    for name in sorted(filenames):
        study_map.setdefault(test_study_key(name), []).append(name)
    return study_map


def study_pairs(images: Sequence[str]) -> List[List[str]]:
    """Deterministic evaluation pairs of one study, images in file order.

    One image -> a self-pair ``[a, a]`` (routed to the single-view prediction
    at inference); n >= 2 images -> the n - 1 windows of consecutive images,
    whose sigmoid probabilities are averaged per study.
    """
    images = list(images)
    if len(images) == 1:
        return [[images[0], images[0]]]
    return [[images[i], images[i + 1]] for i in range(len(images) - 1)]


def sample_training_pair(images: Sequence[str]) -> List[str]:
    """Phase-2 training pair: two distinct images drawn uniformly at random, in
    random order; a single-image study is paired with itself."""
    images = list(images)
    if len(images) == 1:
        return [images[0], images[0]]
    return random.sample(images, 2)


def study_labels(df: pd.DataFrame, label_columns: List[str]) -> pd.DataFrame:
    """Study-level labels indexed by (Patient_ID, Study).

    Labels are expected to be constant within a study; if they are not, the
    per-label maximum is used and a warning is issued.
    """
    grouped = df.groupby(STUDY_COLUMNS)[label_columns]
    labels_max = grouped.max()
    inconsistent = (labels_max != grouped.min()).any(axis=1)
    if inconsistent.any():
        warnings.warn(
            f"{int(inconsistent.sum())} studies have labels that differ between "
            "images; using the per-label maximum."
        )
    return labels_max
