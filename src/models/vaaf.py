"""View-Aware Attention Fusion (VAAF) for multi-view CXR late fusion.

VAAF maps three view-level features -- the pooled features of the two image
slots of a study, z_1 and z_2 (frozen single-view encoder E_s), and the
cross-view feature z' (multi-view encoder E_m) -- to a per-label residual
logit y'. Each feature is a layer-normalised mean of patch tokens (not the
CLS token). A slot is the position an image occupies in the pair, not its
projection: frontal/lateral identity is not supplied to the network.

    T  = LN([z_1; z_2; z'] + l2norm(E_v))                    [3, D]
    Q  = LN(Q_d)                                             [C, D]
    alpha^(h) = softmax(Q W_q^(h) (T W_k^(h))^T / sqrt(d_k))  [C, 3]
    A  = Concat_h(alpha^(h) T W_v^(h)) W_o                   [C, D]
    g  = sigmoid([Q, A] W_g + b_g),  b_g initialised to -2.0
    F  = g * A + (1 - g) * Q
    y' = Dropout(LN(F)) W_c                                  [C]

alpha = mean_h alpha^(h) is the Disease-View Affinity Matrix (rows sum to 1
at inference); it is stored before attention dropout.

References
----------
- Q2L (Liu et al., 2021)  -- label queries + cross-attention for multi-label
- Flamingo (Alayrac et al., NeurIPS 2022) -- gated cross-attention
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Token indices (for readability and attention map access)
VIEW_SLOT1 = 0
VIEW_SLOT2 = 1
VIEW_CROSS = 2
VIEW_NAMES = ["slot-1", "slot-2", "cross-view"]


class ViewAwareAttentionFusion(nn.Module):
    """CXR-specific View-Aware Attention Fusion (VAAF).

    Each of the C pathological labels has a learnable disease-query prototype
    that attends over 3 view-aware tokens (slot 1, slot 2, cross-view) via
    multi-head cross-attention with gating.

    Parameters
    ----------
    embed_dim : int
        Feature vector dimensionality (768 for ViT-B / EVA-X).
    num_labels : int
        Number of classification labels (14 for CXR).
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate for attention weights and for the fused features
        before the classifier.
    gate_bias : float
        Initial gate bias b_g (-2.0): the model starts by relying on the
        query priors and gradually learns to trust attention.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_labels: int = 14,
        num_heads: int = 4,
        dropout: float = 0.2,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_views = 3  # slot 1, slot 2, cross-view

        # ── View-aware embeddings E_v ──────────────────────────────────
        # One learnable embedding per token (slot 1, slot 2, cross-view).
        self.view_embeddings = nn.Parameter(
            torch.randn(1, self.num_views, embed_dim) * 0.02
        )

        # ── Disease query prototypes Q_d ───────────────────────────────
        # One learnable query per label.
        self.disease_queries = nn.Parameter(
            torch.randn(1, num_labels, embed_dim) * 0.02
        )

        # ── Multi-head cross-attention projections ─────────────────────
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # ── Gating mechanism ──────────────────────────────────────────
        self.gate_proj = nn.Linear(2 * embed_dim, embed_dim)
        nn.init.constant_(self.gate_proj.bias, gate_bias)

        # ── Layer normalization ────────────────────────────────────────
        self.ln_views = nn.LayerNorm(embed_dim)
        self.ln_queries = nn.LayerNorm(embed_dim)
        self.ln_out = nn.LayerNorm(embed_dim)

        # ── Per-label classifier ───────────────────────────────────────
        self.classifier = nn.Linear(embed_dim, 1)

        # ── Dropout ────────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # ── Stored attention map for visualization ─────────────────────
        self._disease_view_affinity: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z_slot1: torch.Tensor,
        z_slot2: torch.Tensor,
        z_cross: torch.Tensor,
    ) -> torch.Tensor:
        """Compute disease prediction by attending over view-aware tokens.

        Args:
            z_slot1: [B, D] - pooled (LN of mean patch token) feature of the
                first image slot, from the frozen single-view encoder
            z_slot2: [B, D] - same for the second image slot
            z_cross: [B, D] - pooled cross-view feature z' (multi-view encoder)

        Returns:
            logits: [B, C] - per-label residual logits y'
        """
        B = z_slot1.shape[0]

        # 1. Stack view features and add view-aware embeddings  [B, 3, D]
        #    Order: [slot 1, slot 2, cross-view]
        views = torch.stack([z_slot1, z_slot2, z_cross], dim=1)
        view_emb = F.normalize(self.view_embeddings, dim=-1)
        views = views + view_emb  # view-aware tokens
        views = self.ln_views(views)

        # 2. Disease query prototypes  [B, C, D]
        queries = self.disease_queries.expand(B, -1, -1)
        queries = self.ln_queries(queries)

        # 3. Multi-head cross-attention: queries attend over views
        #    Q: [B, H, C, d_k]   K,V: [B, H, 3, d_k]
        q = (
            self.W_q(queries)
            .view(B, self.num_labels, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.W_k(views)
            .view(B, self.num_views, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.W_v(views)
            .view(B, self.num_views, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention weights: [B, H, C, 3]  (Disease-View Affinity Matrix)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # Store Disease-View Affinity Matrix (averaged over heads) before
        # dropout, so each row sums to 1; dropped weights only aggregate V.
        self._disease_view_affinity = attn_weights.mean(dim=1).detach()
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted aggregation: [B, H, C, d_k] -> [B, C, D]
        attn_out = torch.matmul(attn_weights, v)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B, self.num_labels, self.embed_dim)
        )
        attn_out = self.W_o(attn_out)

        # 4. Gated residual fusion
        gate_input = torch.cat([queries, attn_out], dim=-1)  # [B, C, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))     # [B, C, D]

        fused = gate * attn_out + (1 - gate) * queries       # [B, C, D]
        fused = self.ln_out(fused)
        fused = self.dropout(fused)

        # 5. Per-label prediction: [B, C] 
        logits = self.classifier(fused).squeeze(-1)

        return logits

    # ------------------------------------------------------------------
    # Visualization & Interpretability
    # ------------------------------------------------------------------

    def get_disease_view_affinity(self) -> Optional[torch.Tensor]:
        """Return the Disease-View Affinity Matrix of the last forward pass.

        Returns
        -------
        affinity : Tensor [B, 14, 3] or None
            Column order: [slot 1, slot 2, cross-view].
            Each row sums to 1 and gives each label's reliance on each token.
        """
        return self._disease_view_affinity

    @staticmethod
    def format_affinity_table(
        affinity: torch.Tensor,
        label_names: Optional[List[str]] = None,
    ) -> str:
        """Pretty-print a single sample's Disease-View Affinity Matrix.

        Args:
            affinity: [C, 3] tensor (one sample, no batch dim).
            label_names: optional list of C label names.

        Returns:
            Formatted string table.
        """
        default_labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomed.", "Fracture", "Lung Lesion", "Lung Opacity",
            "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
            "Pneumothorax", "Support Devices",
        ]
        names = label_names or default_labels
        header = f"{'Pathology':<22s} {'Slot 1':>8s} {'Slot 2':>8s} {'Cross-V':>8s}"
        sep = "─" * len(header)
        rows = [sep, header, sep]
        for i, name in enumerate(names):
            s1_val = affinity[i, VIEW_SLOT1].item()
            s2_val = affinity[i, VIEW_SLOT2].item()
            x_val = affinity[i, VIEW_CROSS].item()
            rows.append(f"{name:<22s} {s1_val:>8.3f} {s2_val:>8.3f} {x_val:>8.3f}")
        rows.append(sep)
        return "\n".join(rows)
