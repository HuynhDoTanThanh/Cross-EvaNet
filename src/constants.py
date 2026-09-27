"""Label definitions and constants for CheXray multi-label classification."""

import random

import numpy as np
import torch

LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
NUM_LABELS = len(LABEL_COLUMNS)


def seed_everything(seed: int = 1337, use_xla: bool = False) -> None:
    """Seed Python, NumPy and torch (CPU/CUDA), and the XLA RNG when requested."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_xla:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(seed)
