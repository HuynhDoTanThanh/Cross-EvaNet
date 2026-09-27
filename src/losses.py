"""Loss functions for multi-label chest X-ray classification."""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.register_buffer(
            "alpha", alpha if isinstance(alpha, torch.Tensor) else None
        )
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        if self.alpha is not None:
            bce = self.alpha * bce
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class AsymmetricLossOptimized(nn.Module):
    """Optimized ASL: minimizes memory allocation and favors inplace operations."""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.targets = (
            self.anti_targets
        ) = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(
            self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps))
        )
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            grad_ctx = (
                torch.no_grad() if self.disable_torch_grad_focal_loss else contextlib.nullcontext()
            )
            with grad_ctx:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(
                    1 - self.xs_pos - self.xs_neg,
                    self.gamma_pos * self.targets
                    + self.gamma_neg * self.anti_targets,
                )
            self.loss *= self.asymmetric_w
        return -self.loss.sum()


class APLLoss(nn.Module):
    """Asymmetric Polynomial Loss (Taylor-expanded ASL), summed over labels and batch.

    Positive term: log(p) + eps1_pos * (1 - p) + 0.5 * eps2_pos * (1 - p)^2 with
    eps1_pos = 1, eps2_pos = -2.5; negative term: log(p_m) + eps1_neg * p_m with
    eps1_neg = 0 and the shifted probability p_m = min(1 - p + clip, 1). The
    one-sided focusing weight (gamma_neg=5, gamma_pos=0) is detached.
    """

    def __init__(
        self,
        gamma_neg=5,
        gamma_pos=0,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(APLLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = -2.5

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * (
            torch.log(xs_pos.clamp(min=self.eps))
            + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps))
            + self.epsilon_pos_pow
            * 0.5
            * torch.pow(1 - xs_pos.clamp(min=self.eps), 2)
        )
        los_neg = (1 - y) * (
            torch.log(xs_neg.clamp(min=self.eps))
            + self.epsilon_neg * (xs_neg.clamp(min=self.eps))
        )
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            grad_ctx = (
                torch.no_grad() if self.disable_torch_grad_focal_loss else contextlib.nullcontext()
            )
            with grad_ctx:
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w
        return -loss.sum()


class TwoWayLoss(nn.Module):
    """Two-way multi-label loss (Kobayashi, CVPR 2023) with Tp=4, Tn=1.

    Sample-wise and class-wise softplus(LSE_neg + LSE_pos) terms, each averaged
    over the samples / classes that have at least one positive. Masking uses
    a large finite negative instead of -inf and weighted means instead of
    boolean indexing, so shapes stay static (XLA) and gradients stay finite.
    """

    def __init__(self, Tp: float = 4.0, Tn: float = 1.0):
        super().__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        pos = y > 0
        neg = y == 0
        class_mask = pos.any(dim=0).to(x.dtype)
        sample_mask = pos.any(dim=1).to(x.dtype)
        big_neg = -1e4
        pmask = torch.where(pos, torch.zeros_like(x), torch.full_like(x, big_neg))
        nmask = torch.where(neg, torch.zeros_like(x), torch.full_like(x, big_neg))

        plogit_class = torch.logsumexp(-x / self.Tp + pmask, dim=0) * self.Tp
        plogit_sample = torch.logsumexp(-x / self.Tp + pmask, dim=1) * self.Tp
        nlogit_class = torch.logsumexp(x / self.Tn + nmask, dim=0) * self.Tn
        nlogit_sample = torch.logsumexp(x / self.Tn + nmask, dim=1) * self.Tn

        loss_class = F.softplus(nlogit_class + plogit_class) * class_mask
        loss_sample = F.softplus(nlogit_sample + plogit_sample) * sample_mask
        return (
            loss_class.sum() / class_mask.sum().clamp(min=1)
            + loss_sample.sum() / sample_mask.sum().clamp(min=1)
        )


class ZLPRLoss(nn.Module):
    """ZLPR / multi-label categorical cross-entropy (Su, 2022), batch mean.

    log(1 + sum_{neg} e^{s}) + log(1 + sum_{pos} e^{-s}); no hyperparameters.
    """

    def forward(self, x, y):
        x = (1 - 2 * y) * x
        x_neg = x - y * 1e12
        x_pos = x - (1 - y) * 1e12
        zeros = torch.zeros_like(x[..., :1])
        neg_loss = torch.logsumexp(torch.cat([x_neg, zeros], dim=-1), dim=-1)
        pos_loss = torch.logsumexp(torch.cat([x_pos, zeros], dim=-1), dim=-1)
        return (neg_loss + pos_loss).mean()


LOSS_NAMES = ("bce", "focal", "twoway", "zlpr", "asl", "apl")


def build_loss(name: str) -> nn.Module:
    """Return the training objective selected by ``--loss``.

    bce: BCEWithLogits (mean); focal: Focal (gamma=2, mean); twoway: Two-Way
    (Tp=4, Tn=1); zlpr: ZLPR (batch mean); asl: ASL (gamma_neg=4, gamma_pos=1,
    clip=0.05, sum); apl: APL (gamma_neg=5, gamma_pos=0, clip=0.05, sum).
    """
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "focal":
        return FocalLoss(gamma=2.0)
    if name == "twoway":
        return TwoWayLoss(Tp=4.0, Tn=1.0)
    if name == "zlpr":
        return ZLPRLoss()
    if name == "asl":
        return AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05)
    if name == "apl":
        return APLLoss()
    raise ValueError(f"Unknown loss {name!r}; choose from {LOSS_NAMES}")
