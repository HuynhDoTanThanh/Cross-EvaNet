"""EVA backbone and checkpoint loading utilities."""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed, resample_patch_embed
from timm.models.eva import Eva


def checkpoint_filter_fn(
    state_dict,
    model,
    interpolation="bicubic",
    antialias=True,
):
    """timm's EVA checkpoint filter.

    Remaps EVA-02 / MIM key names to timm, drops ``mask_token`` / ``lm_head`` /
    rope buffers, moves the MIM ``norm`` to ``fc_norm`` and resamples
    ``pos_embed`` / ``patch_embed`` to the model's grid (bicubic).
    ``state_dict`` must already be unwrapped (``unwrap_state_dict``, which
    prefers non-EMA weights).
    """
    out_dict = {}
    if "visual.trunk.pos_embed" in state_dict:
        prefix = "visual.trunk."
    elif "visual.pos_embed" in state_dict:
        prefix = "visual."
    else:
        prefix = ""
    mim_weights = prefix + "mask_token" in state_dict
    no_qkv = prefix + "blocks.0.attn.q_proj.weight" in state_dict
    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue
        if "rope" in k:
            continue
        if "patch_embed.proj.weight" in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v, (H, W), interpolation=interpolation, antialias=antialias, verbose=True
                )
        elif k == "pos_embed" and v.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = (
                0
                if getattr(model, "no_embed_class", False)
                else getattr(model, "num_prefix_tokens", 1)
            )
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        k = k.replace("mlp.ffn_ln", "mlp.norm")
        k = k.replace("attn.inner_attn_ln", "attn.norm")
        k = k.replace("mlp.w12", "mlp.fc1")
        k = k.replace("mlp.w1", "mlp.fc1_g")
        k = k.replace("mlp.w2", "mlp.fc1_x")
        k = k.replace("mlp.w3", "mlp.fc2")
        if no_qkv:
            k = k.replace("q_bias", "q_proj.bias")
            k = k.replace("v_bias", "v_proj.bias")
        if mim_weights and k in (
            "mask_token",
            "lm_head.weight",
            "lm_head.bias",
            "norm.weight",
            "norm.bias",
        ):
            if k == "norm.weight" or k == "norm.bias":
                k = k.replace("norm", "fc_norm")
            else:
                continue
        out_dict[k] = v
    return out_dict


# Wrapper prefixes stripped when every key of a checkpoint carries them
# (DataParallel / training-script containers / a TripleBranchEVA sub-module).
_STRIP_PREFIXES = ("module.", "model.", "single_model.")
# Containers unwrapped in order; non-EMA weights are preferred and EMA weights
# are used only when no other container is present.
_CONTAINER_KEYS = ("model", "module", "state_dict", "model_ema", "state_dict_ema")


def unwrap_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """Return the bare state_dict of a checkpoint.

    Unwraps ``model`` / ``module`` / ``state_dict`` containers (EMA only when
    nothing else is present) and strips the ``module.``, ``model.`` and
    ``single_model.`` prefixes when all keys share them.
    """
    state_dict = checkpoint.state_dict() if isinstance(checkpoint, nn.Module) else checkpoint
    unwrapped = True
    while unwrapped:
        unwrapped = False
        for key in _CONTAINER_KEYS:
            value = state_dict.get(key)
            if isinstance(value, dict):
                state_dict = value
                unwrapped = True
                break
    stripped = True
    while stripped and state_dict:
        stripped = False
        for prefix in _STRIP_PREFIXES:
            if all(k.startswith(prefix) for k in state_dict):
                state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
                stripped = True
    return dict(state_dict)


def load_evax_init_weights(model: "EVA_X", checkpoint_path: Optional[str]) -> str:
    """Initialise a bare EVA-X backbone from public EVA-X or Phase-1 weights.

    Used for the multi-view encoder E_m (before it is wrapped by
    ``MultiImageHybridEVA``) and for Phase-1 single-view fine-tuning. The
    source is auto-detected: a masked-image-modelling (MIM) checkpoint
    (``mask_token`` / ``lm_head`` present) or a single-view classification
    checkpoint such as the Phase-1 ``theta_s*``. Keys go through
    ``checkpoint_filter_fn`` (EVA-02 -> timm key remap, drop ``mask_token`` /
    ``lm_head`` / rope buffers, bicubic resampling of ``pos_embed`` e.g.
    14x14 -> 28x28, and of ``patch_embed`` if needed).

    Raises if no path is given, if any key other than ``head.weight`` /
    ``head.bias`` is missing, or if any checkpoint key is unexpected.

    Returns:
        ``"mim"`` or ``"single_view"`` (the detected source).
    """
    if not checkpoint_path:
        raise ValueError("An EVA-X checkpoint path is required to initialise the backbone.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)
    source = (
        "mim"
        if "mask_token" in state_dict or any(k.startswith("lm_head.") for k in state_dict)
        else "single_view"
    )
    state_dict = checkpoint_filter_fn(state_dict, model)
    # The classifier is (re)initialised when the checkpoint has none or a
    # different label space; it is the only tolerated missing key.
    for key in ("head.weight", "head.bias"):
        if key in state_dict and state_dict[key].shape != model.state_dict()[key].shape:
            print(f"Dropping {key} with shape {tuple(state_dict[key].shape)} (label space differs)")
            del state_dict[key]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = sorted(set(missing) - {"head.weight", "head.bias"})
    if bad_missing or unexpected:
        raise RuntimeError(
            f"EVA-X checkpoint {checkpoint_path} does not match the backbone: "
            f"missing={bad_missing}, unexpected={sorted(unexpected)}"
        )
    print(
        f"Loaded {source} EVA-X weights from {checkpoint_path} "
        f"({len(state_dict)} tensors; missing: {sorted(missing) or 'none'})"
    )
    return source


def load_single_view_weights(model: "EVA_X", checkpoint_path: Optional[str]) -> None:
    """Strictly load the Phase-1 single-view checkpoint ``theta_s*`` (encoder E_s).

    The checkpoint must be a fine-tuned EVA-X classifier at the model's
    resolution, including ``head.weight`` [C, D] and ``head.bias`` [C].
    Containers and wrapper prefixes are removed; rope buffers are ignored.
    """
    if not checkpoint_path:
        raise ValueError(
            "Phase 2 requires the Phase-1 single-view checkpoint (theta_s*) for E_s."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = unwrap_state_dict(checkpoint)
    if "mask_token" in state_dict or any(k.startswith("lm_head.") for k in state_dict):
        raise ValueError(
            f"{checkpoint_path} is a MIM pre-training checkpoint; E_s needs the "
            "Phase-1 fine-tuned single-view checkpoint."
        )
    state_dict = {k: v for k, v in state_dict.items() if "rope" not in k}
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded Phase-1 single-view weights from {checkpoint_path}")


class EVA_X(Eva):
    """EVA with explicit forward_features / forward_head for integration."""

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def eva_x_base_patch16(
    pretrained: Union[bool, str] = False,
    drop_path_rate: float = 0.0,
    img_size: int = 448,
    num_classes: int = 14,
) -> EVA_X:
    """Build EVA-X base 448 patch16.

    ``pretrained`` may be a path to public EVA-X (MIM) weights or to a
    single-view EVA-X checkpoint; it is loaded with ``load_evax_init_weights``.
    RoPE uses raw patch coordinates: its reference grid is the input grid
    (28x28 at 448, 14x14 at 224). The rope buffers are not saved in
    checkpoints, so every builder of a given ``img_size`` gets the same RoPE.
    """
    grid = img_size // 16
    model = EVA_X(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(grid, grid),
        drop_path_rate=drop_path_rate,
    )
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    if isinstance(pretrained, str):
        load_evax_init_weights(model, pretrained)
    return model
