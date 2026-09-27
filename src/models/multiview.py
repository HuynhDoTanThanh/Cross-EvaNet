"""Multi-view fusion model (MultiImageHybridEVA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .eva_backbone import EVA_X, eva_x_base_patch16, load_evax_init_weights


class MultiImageHybridEVA(nn.Module):
    """EVA backbone with learnable view embeddings and multi-image token fusion."""

    def __init__(self, backbone: EVA_X, num_classes: int, n: int):
        super().__init__()
        self.n = n
        self.num_classes = num_classes
        # ``backbone.head`` is kept although the VAAF model only uses the pooled
        # feature z' (the parameter counts of Table 4 include it); it is the
        # linear head of the ``fusion_head_type="linear"`` ablation.
        self.model = backbone
        self.embed_dim = self.model.embed_dim
        self.img_embed_matrix = nn.Parameter(
            torch.zeros(1, n, self.embed_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self.img_embed_matrix)

    def format_multi_image_tokens(self, x, batch_size: int):
        x = x.view(batch_size, self.n, -1, self.embed_dim)
        cls_token = x[:, 0, 0:1, :]
        patch_tokens = x[:, :, 1:, :]
        patch_tokens = patch_tokens.reshape(batch_size, -1, self.embed_dim)
        view_embeds = F.normalize(self.img_embed_matrix, dim=-1)
        patches_per_view = patch_tokens.shape[1] // self.n
        view_embeds_expanded = view_embeds.unsqueeze(2).repeat(
            1, 1, patches_per_view, 1
        )
        view_embeds_expanded = view_embeds_expanded.reshape(1, -1, self.embed_dim)
        patch_tokens = patch_tokens + view_embeds_expanded
        x_out = torch.cat([cls_token, patch_tokens], dim=1)
        return x_out

    def adapt_rope_for_multiview(self, rot_pos_embed):
        if rot_pos_embed is None:
            return None
        if rot_pos_embed.dim() == 2:
            new_rope = rot_pos_embed.repeat(self.n, 1)
        elif rot_pos_embed.dim() == 3 and rot_pos_embed.shape[0] == 1:
            new_rope = rot_pos_embed.repeat(1, self.n, 1)
        elif rot_pos_embed.dim() == 3:
            new_rope = rot_pos_embed.repeat(self.n, 1, 1)
        elif rot_pos_embed.dim() == 4:
            new_rope = rot_pos_embed.repeat(1, self.n, 1, 1)
        else:
            shape = list(rot_pos_embed.shape)
            repeats = [1] * len(shape)
            repeats[-2] = self.n
            new_rope = rot_pos_embed.repeat(*repeats)
        return new_rope

    def forward(self, x, return_features: bool = False):
        batch_size = x.shape[0]
        x = einops.rearrange(x, "b n c h w -> (b n) c h w")
        x = self.model.patch_embed(x)
        x, rot_pos_embed = self.model._pos_embed(x)
        mv_tokens = self.format_multi_image_tokens(x, batch_size)
        mv_rope = self.adapt_rope_for_multiview(rot_pos_embed)
        for blk in self.model.blocks:
            mv_tokens = blk(mv_tokens, rope=mv_rope)
        mv_tokens = self.model.norm(mv_tokens)
        features = self.model.forward_head(mv_tokens, pre_logits=True)
        if return_features:
            return features
        return self.model.head(features)


def build_multiview_model(
    num_classes: int, cfg, load_pretrained: bool = True
) -> MultiImageHybridEVA:
    """Build the multi-view encoder E_m.

    With ``load_pretrained`` the bare backbone is initialised from
    ``cfg.mv_init_ckpt`` (public EVA-X MIM weights, or a Phase-1 single-view
    checkpoint) *before* it is wrapped; the view embeddings v_k keep their
    Xavier-uniform init. Pass ``load_pretrained=False`` when the weights come
    from a full Phase-2 checkpoint.
    """
    backbone = eva_x_base_patch16(
        pretrained=False,
        drop_path_rate=cfg.drop_path_rate,
        img_size=cfg.img_size,
        num_classes=num_classes,
    )
    if load_pretrained:
        load_evax_init_weights(backbone, cfg.mv_init_ckpt)
    return MultiImageHybridEVA(backbone, num_classes=num_classes, n=cfg.num_views)
