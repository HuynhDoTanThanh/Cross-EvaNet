"""Mixed-precision helpers shared by training and inference.

bfloat16 autocast on XLA (TPU) with fp32 master weights; on CUDA bfloat16
when supported, else float16 with a GradScaler; no autocast on CPU. Losses
and sigmoids are computed in fp32 outside autocast by the callers.
"""

import contextlib
import os
from typing import Optional

import torch


def amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    """Autocast dtype for ``device`` (None = run in fp32)."""
    if device.type == "xla":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def check_xla_mixed_precision_env() -> None:
    """Raise if an environment flag would turn XLA mixed precision into pure bf16."""
    for flag in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16"):
        if os.environ.get(flag, "0") not in ("", "0"):
            raise RuntimeError(
                f"{flag}={os.environ[flag]} runs the whole model in pure bf16; bf16 mixed "
                f"precision (fp32 master weights) requires it to be unset."
            )


def autocast(device: torch.device):
    """Autocast context for ``device`` (a no-op on CPU)."""
    dtype = amp_dtype(device)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(device: torch.device):
    """GradScaler for CUDA float16 autocast, else None."""
    if amp_dtype(device) is not torch.float16:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()
