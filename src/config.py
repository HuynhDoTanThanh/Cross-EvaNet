"""Training and inference configuration."""

import json
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    """Configuration for Phase-2 multi-view CheXray training."""

    # Data
    train_csv: str = "train_mv.csv"
    train_dir: str = ""
    val_split: float = 0.1
    # Persisted patient split (Patient_ID,split); created from ``seed`` on
    # first use and read by both phases, e.g. configs/split_B.csv.
    split_csv: Optional[str] = None

    # Checkpoints
    pretrained: Optional[str] = None  # Phase-1 single-view checkpoint theta_s* (E_s)
    mv_init_ckpt: Optional[str] = None  # EVA-X weights initialising E_m (MIM or Phase-1)
    save_path: str = "outputs/crossevanet_phase2.pth"  # final-epoch checkpoint

    # Model
    img_size: int = 448
    num_views: int = 2
    drop_path_rate: float = 0.2
    fusion_head_type: str = "vaaf"  # "vaaf" or "linear" (without-VAAF ablation)
    single_view_route: bool = True  # single-image studies -> y1 at inference

    # Training
    batch_size: int = 10
    num_workers: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 8
    warmup_pct: float = 0.05
    grad_accum_steps: int = 4
    clip_grad: float = 1.0
    loss: str = "apl"  # bce, focal, twoway, zlpr, asl, apl
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.num_views != 2:
            raise ValueError(f"Only image pairs are supported (num_views=2), got {self.num_views}")


def load_json_config(path: str) -> Dict[str, Any]:
    """Read a per-experiment JSON config whose keys are TrainConfig fields."""
    with open(path) as f:
        values = json.load(f)
    unknown = sorted(set(values) - {f.name for f in fields(TrainConfig)})
    if unknown:
        raise ValueError(f"Unknown TrainConfig fields in {path}: {unknown}")
    return values
