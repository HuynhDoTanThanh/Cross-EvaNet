"""Training loop, evaluation, and scheduler."""

import math
from typing import Any, Dict

import numpy as np
import torch
from scipy.special import expit

from src.inference import predict_images
from src.metrics import macro_auc, per_label_auc
from src.precision import autocast


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    cfg,
    use_xla: bool = False,
    scaler=None,
) -> float:
    """Run one training epoch. Supports TPU (XLA), GPU or CPU.

    The forward pass runs under bf16 autocast on XLA / CUDA (fp16 + ``scaler``
    on CUDA without bf16); the loss is computed in fp32 outside autocast.
    Micro-batches left over at the end of an epoch (len % grad_accum_steps)
    are discarded rather than carried into the next epoch. Works for any
    model mapping ``images`` to logits (Phase 1 and Phase 2).
    """
    model.train()
    # Accumulated on the device: a per-step .item() would force an extra graph
    # execution on XLA (and a host sync on CUDA).
    total_loss = torch.zeros((), device=device)
    step = 0
    xm = None
    if use_xla:
        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            use_xla = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device):
            logits = model(imgs)
        loss = criterion(logits.float(), labels.float()) / cfg.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        step += 1

        if step % cfg.grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
            if use_xla and xm is not None:
                xm.optimizer_step(optimizer)
            elif scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.detach() * cfg.grad_accum_steps

    return total_loss.item() / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, use_xla: bool = False) -> Dict[str, Any]:
    """Deterministic per-image validation of a TripleBranchEVA (monitoring only).

    The unit of evaluation is the image, as on the Grand X-Ray SLAM
    leaderboard; every image carries its study's labels, and labels of -1
    are excluded per label. One forward pass gives:

    - ``macro_auc`` / ``per_class_auc``: the deployed output over all
      validation images. Every image receives its study's prediction (Eq. (1),
      window-averaged for n > 2 images; y1 for single-image studies when the
      single-view route is on).
    - ``paired_fused_*`` vs ``paired_single_*``: on the images of paired
      studies (>= 2 images), the fused study prediction assigned to each image
      vs. the frozen single-view prediction y_k of each image on its own.

    Counts: ``n_images`` / ``n_studies`` (all), ``n_paired_images`` /
    ``n_paired`` (images and studies of paired studies).
    """
    preds = predict_images(model, loader, device, use_xla=use_xla)
    labels, paired = preds["labels"], preds["paired"]
    study_keys = np.asarray(preds["study_keys"], dtype=object)
    per_class_auc = per_label_auc(labels, preds["probs"])
    fused_auc = per_label_auc(labels[paired], preds["probs"][paired])
    single_auc = per_label_auc(labels[paired], preds["single_probs"][paired])
    return {
        "macro_auc": macro_auc(per_class_auc),
        "per_class_auc": per_class_auc,
        "n_images": int(len(labels)),
        "n_studies": len(set(study_keys)),
        "n_paired_images": int(paired.sum()),
        "n_paired": len(set(study_keys[paired])),
        "paired_fused_macro_auc": macro_auc(fused_auc),
        "paired_fused_per_class_auc": fused_auc,
        "paired_single_macro_auc": macro_auc(single_auc),
        "paired_single_per_class_auc": single_auc,
    }


@torch.no_grad()
def evaluate_single_view(model, loader, device) -> Dict[str, Any]:
    """Per-image AUC of a single-view model over a loader of (images, labels)."""
    model.eval()
    all_probs = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device):
            logits = model(imgs)
        all_probs.append(logits.float().cpu())
        all_labels.append(labels.float().cpu())
    # Sigmoid in float64 on the host, so that large logits keep their ranking.
    probs = expit(torch.cat(all_probs, dim=0).numpy().astype(np.float64))
    labels = torch.cat(all_labels, dim=0).numpy()
    per_class_auc = per_label_auc(labels, probs)
    return {"macro_auc": macro_auc(per_class_auc), "per_class_auc": per_class_auc}


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Cosine schedule to 10% of the peak LR with linear warmup (5% of steps).

    ``steps_per_epoch`` is the number of micro-batches per epoch; the schedule
    counts optimizer steps, i.e. ``epochs * (steps_per_epoch // accum)``.
    """
    total_steps = max(1, cfg.epochs * (steps_per_epoch // cfg.grad_accum_steps))
    warmup_steps = int(cfg.warmup_pct * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
