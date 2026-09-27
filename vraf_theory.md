# View-Aware Attention Fusion (VAAF)

This note restates the VAAF equations of the paper (Sec. 4.2.3, Eqs. 10-17) and maps each one to
`src/models/vaaf.py` and `src/models/triple_branch.py`. If this note and the paper disagree, the
paper is authoritative.

## Inputs: three view-level features

VAAF takes three pooled features of dimension D = 768:

| Token | Source | Code |
|---|---|---|
| `z_1` | Frozen single-view encoder `E_s`, image in **slot 1** | `z_slot1` |
| `z_2` | Frozen single-view encoder `E_s`, image in **slot 2** | `z_slot2` |
| `z'` | Multi-view encoder `E_m`, merged `1 + 2N` token sequence | `z_cross` |

Each feature is the layer-normalised mean of the final-layer patch tokens. The CLS token is not
pooled.

A slot is the position an image occupies in the pair, not its projection:

- Phase-2 training loads the two images in random order.
- Inference takes them in file order.
- No frontal/lateral metadata is used.

The paper writes `z_F = z_1` and `z_L = z_2`. These are frontal and lateral only when a study
happens to be ordered that way.

## Equations

With `N_h = 4` heads, `d_k = D / N_h = 192` and C = 14 labels (batch dimension omitted):

```
T        = LN([z_1; z_2; z'] + E_v_hat)                         in R^{3 x D}   (view-aware tokens)
Q        = LN(Q_d)                                              in R^{C x D}   (disease queries)

Q^(h)    = Q W_q^(h),   K^(h) = T W_k^(h),   V^(h) = T W_v^(h)
alpha^(h)= softmax(Q^(h) K^(h)^T / sqrt(d_k))                   in R^{C x 3}
Attn_out = Concat_h(alpha^(h) V^(h)) W_o                        in R^{C x D}

g        = sigmoid(Concat(Q, Attn_out) W_g + b_g)               in R^{C x D}   (b_g initialised to -2.0)
F        = g * Attn_out + (1 - g) * Q                           in R^{C x D}
y'       = Dropout(LN(F)) W_c                                   in R^{C}

y        = 1/2 * ( 1/2 * (y_1 + y_2) + y' )                     (Eq. 1 = Eq. 17)
```

- **`E_v_hat`:** the row-wise l2-normalised view-embedding matrix `E_v` in R^{3 x D}. It is added
  to the features before the LayerNorm, so the view signal cannot dominate the feature magnitude.
- **Query normalisation:** the paper writes `Q_d` in the gate and residual equations to denote the
  layer-normalised prototypes, i.e. `Q` above.
- **`y_k = W_h z_k + b_h`:** the frozen single-view logits.
- **Logit level:** every operation is on logits. No temperature is applied. The sigmoid is used
  only inside the loss and to turn the final logits into probabilities.
- **Single-view route:** at inference, a study with exactly one image returns `y_1` unchanged,
  and VAAF is not used for it (see the README).
- **Evaluation unit:** `y` is one prediction per study and is assigned to every image of the
  study (for n > 2 images, the average of the window probabilities). AUCs are computed per image:
  the single-view comparator scores each image with its own `y_k` (`y_1` for the slot-1 image,
  `y_2` for the slot-2 image, no averaging), and the bootstrap resamples studies as clusters (see
  the README, "Evaluation protocol").

## Disease-View Affinity Matrix

`alpha = (1 / N_h) * sum_h alpha^(h)` in `[0,1]^{C x 3}`, with columns [slot 1, slot 2, cross-view].

- It is taken from the softmax **before** attention dropout, so every row sums to 1. The
  dropped-out weights are used only to aggregate `V`.
- `alpha` for the last forward pass is available from
  `ViewAwareAttentionFusion.get_disease_view_affinity()` ([B, C, 3]).
- `format_affinity_table` prints it for one sample.
- `TripleBranchEVA.forward(..., return_all=True)` returns it as `alpha`. Routed single-image rows
  are NaN there, and the key is `None` for the linear head.

The slot columns read as frontal/lateral reliance only when the inputs are ordered by projection
at inference. The cross-view column does not depend on order. The paper does not report learned
`alpha` values.

## Hyperparameters and initialisation

| Item | Value | Code |
|---|---|---|
| Heads `N_h` | 4 | `num_heads` |
| Dropout | 0.2 on the attention weights and on `LN(F)` before `W_c` | `attn_dropout`, `dropout` |
| Gate bias `b_g` | constant -2.0 (`sigmoid(-2) ~ 0.12`) | `gate_proj.bias` |
| `E_v`, `Q_d` | `N(0, 0.02^2)` | `view_embeddings`, `disease_queries` |
| `W_q`, `W_k`, `W_v`, `W_o` (with biases), `W_g` weight | PyTorch default `nn.Linear` init | `W_q`, `W_k`, `W_v`, `W_o`, `gate_proj` |
| LayerNorms on `T`, `Q_d`, `F` | PyTorch default | `ln_views`, `ln_queries`, `ln_out` |
| `W_c` | `nn.Linear(D, 1)`, shared by all labels, with a scalar bias | `classifier` |
| Parameters | 3.56 M | - |

## Ablation: without VAAF (Table 9)

`fusion_head_type="linear"` (`--fusion-head linear`) keeps `E_m` and maps `z'` to logits with
`E_m`'s linear classification head. The result enters the same form as Eq. (1), with no queries
and no gating:

```
y' = W z' + b,      y = 1/2 * ( 1/2 * (y_1 + y_2) + y' )
```

## References

- Liu et al., *Query2Label*, 2021: label queries with cross-attention.
- Alayrac et al., *Flamingo*, NeurIPS 2022: gated cross-attention with a suppressed initial
  gate.
