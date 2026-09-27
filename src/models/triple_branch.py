"""Triple-branch model: single-view + multi-view + fusion head."""

import contextlib
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.config import TrainConfig

from .eva_backbone import eva_x_base_patch16, load_single_view_weights, unwrap_state_dict
from .vaaf import ViewAwareAttentionFusion
from .multiview import MultiImageHybridEVA, build_multiview_model

FUSION_HEAD_TYPES = ("vaaf", "linear")


def build_model_evax(num_classes: int, cfg, load_pretrained: bool = True):
    """Single-view EVA-X encoder E_s (for triple-branch).

    With ``load_pretrained`` the Phase-1 checkpoint ``cfg.pretrained``
    (theta_s*) is required and loaded strictly, classification head included.
    """
    model = eva_x_base_patch16(
        pretrained=False,
        drop_path_rate=cfg.drop_path_rate,
        img_size=cfg.img_size,
        num_classes=num_classes,
    )
    if load_pretrained:
        load_single_view_weights(model, getattr(cfg, "pretrained", None))
    return model


class TripleBranchEVA(nn.Module):
    """Single-view (frozen) + multi-view + fusion head.

    Final logits follow Eq. (1): y = 0.5 * (0.5 * (y1 + y2) + y'), where y' is
    the VAAF output (``fusion_head_type="vaaf"``) or, for the "without VAAF"
    ablation, the linear head of E_m applied to z' (``"linear"``).
    """

    def __init__(
        self,
        single_view_model: nn.Module,
        multi_view_model: MultiImageHybridEVA,
        num_classes: int,
        embed_dim: int,
        freeze_single: bool = True,
        fusion_head_type: str = "vaaf",
        single_view_route: bool = True,
    ):
        super().__init__()
        if fusion_head_type not in FUSION_HEAD_TYPES:
            raise ValueError(
                f"fusion_head_type must be one of {FUSION_HEAD_TYPES}, got {fusion_head_type!r}"
            )
        self.num_classes = num_classes
        self.fusion_head_type = fusion_head_type
        self.single_view_route = single_view_route
        self.single_model = single_view_model
        if freeze_single:
            print("🔒 Freezing Single View Branch (Weights & BN stats)...")
            for param in self.single_model.parameters():
                param.requires_grad = False
            self.single_model.eval()

        self.multi_model = multi_view_model
        if fusion_head_type == "vaaf":
            self.fusion_head = ViewAwareAttentionFusion(
                embed_dim=embed_dim,
                num_labels=num_classes,
                num_heads=4,
                dropout=0.2,
            )
        else:
            # y' = W z' + b with the (default-initialised) head of E_m.
            self.fusion_head = None

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "single_model"):
            is_frozen = not next(self.single_model.parameters()).requires_grad
            if is_frozen:
                self.single_model.eval()
        return self

    def forward_single_branch(self, x):
        features = self.single_model.forward_features(x)
        vec = self.single_model.forward_head(features, pre_logits=True)
        logits = self.single_model.head(vec)
        return vec, logits

    def forward_multi_branch(self, x):
        return self.multi_model(x, return_features=True)

    def forward_fusion(
        self, x, vec_1, vec_2
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return the residual logits y' and, for VAAF, the affinity alpha."""
        vec_fused = self.forward_multi_branch(x)
        if self.fusion_head is None:
            return self.multi_model.model.head(vec_fused), None
        logits = self.fusion_head(z_slot1=vec_1, z_slot2=vec_2, z_cross=vec_fused)
        return logits, self.fusion_head.get_disease_view_affinity()

    def forward(
        self,
        x,
        single_mask: Optional[torch.Tensor] = None,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Score an image pair.

        Args:
            x: [B, 2, C, H, W] images in slot order.
            single_mask: optional [B] bool, True for studies with exactly one
                image (self-paired). With ``single_view_route`` and in eval
                mode, those rows return the frozen single-view logits y1
                unchanged and E_m / the fusion head are skipped for them. In
                training mode it is ignored, so Phase-2 self-pairs always go
                through the fusion pathway.
            return_all: return a dict with ``logits`` (as returned by default),
                ``single_logits`` = 0.5 * (y1 + y2), ``logits_1``, ``logits_2``,
                ``fusion_logits`` (y'; 0 for routed rows) and ``alpha``
                ([B, C, 3], NaN for routed rows; None for the linear head).
        """
        img1 = x[:, 0]
        img2 = x[:, 1]
        frozen = not next(self.single_model.parameters()).requires_grad
        with torch.no_grad() if frozen else contextlib.nullcontext():
            vec_1, logits_1 = self.forward_single_branch(img1)
            vec_2, logits_2 = self.forward_single_branch(img2)
        avg_single_logits = 0.5 * (logits_1 + logits_2)

        route = None
        if self.single_view_route and single_mask is not None and not self.training:
            route = single_mask.to(device=logits_1.device, dtype=torch.bool)

        if route is not None and x.device.type != "xla":
            # Run E_m and the fusion head on paired rows only.
            paired_idx = (~route).nonzero(as_tuple=True)[0]
            logits_fusion = torch.zeros_like(avg_single_logits)
            alpha = None
            if self.fusion_head is not None:
                alpha = logits_1.new_full((x.shape[0], self.num_classes, 3), float("nan"))
            if paired_idx.numel() > 0:
                sub_logits, sub_alpha = self.forward_fusion(
                    x[paired_idx], vec_1[paired_idx], vec_2[paired_idx]
                )
                logits_fusion[paired_idx] = sub_logits.to(logits_fusion.dtype)
                if alpha is not None:
                    alpha[paired_idx] = sub_alpha.to(alpha.dtype)
        else:
            # XLA keeps static shapes: fusion runs on every row and routed
            # rows are replaced below.
            logits_fusion, alpha = self.forward_fusion(x, vec_1, vec_2)

        final_logits = 0.5 * (avg_single_logits + logits_fusion)
        if route is not None:
            final_logits = torch.where(route[:, None], logits_1, final_logits)
        if not return_all:
            return final_logits
        if route is not None:
            logits_fusion = logits_fusion.masked_fill(route[:, None], 0.0)
            if alpha is not None:
                alpha = alpha.masked_fill(route[:, None, None], float("nan"))
        return {
            "logits": final_logits,
            "single_logits": avg_single_logits,
            "logits_1": logits_1,
            "logits_2": logits_2,
            "fusion_logits": logits_fusion,
            "alpha": alpha,
        }


def build_triple_branch_model(
    num_classes: int, cfg, embed_dim: int = 768, load_pretrained: bool = True
):
    """Build full triple-branch model.

    With ``load_pretrained`` E_s is loaded from ``cfg.pretrained`` (Phase-1
    theta_s*, required) and E_m is initialised from ``cfg.mv_init_ckpt``
    (EVA-X weights, required). Use ``load_pretrained=False`` before loading a
    complete Phase-2 checkpoint.
    """
    single_backbone = build_model_evax(num_classes, cfg, load_pretrained=load_pretrained)
    multi_view_model = build_multiview_model(num_classes, cfg, load_pretrained=load_pretrained)
    model = TripleBranchEVA(
        single_view_model=single_backbone,
        multi_view_model=multi_view_model,
        num_classes=num_classes,
        embed_dim=embed_dim,
        freeze_single=True,
        fusion_head_type=getattr(cfg, "fusion_head_type", "vaaf"),
        single_view_route=getattr(cfg, "single_view_route", True),
    )
    return model


def load_triple_branch_model(
    checkpoint_path: str,
    num_classes: int,
    img_size: int = 448,
    single_view_route: bool = True,
) -> TripleBranchEVA:
    """Build a TripleBranchEVA and strictly load a Phase-2 checkpoint into it.

    Accepts the ``{"model": state_dict, "config": ..., "epoch": ...}`` files
    written by ``scripts/train.py`` as well as bare state_dicts. The fusion
    head type is inferred from the keys (``fusion_head.*`` -> VAAF).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)
    fusion_head_type = (
        "vaaf" if any(k.startswith("fusion_head.") for k in state_dict) else "linear"
    )
    cfg = TrainConfig(
        img_size=img_size,
        fusion_head_type=fusion_head_type,
        single_view_route=single_view_route,
    )
    model = build_triple_branch_model(num_classes, cfg, load_pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    return model
