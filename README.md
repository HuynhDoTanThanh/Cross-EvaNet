# Cross-EvaNet: Multi-view Vision Transformer Fusion for Automated Thoracic Disease Classification

Reference implementation of **Cross-EvaNet**, a multi-view chest X-ray classifier built on the
[EVA-X](https://github.com/hustvl/EVA-X) foundation model. Cross-EvaNet fuses the two radiographs
of a study (typically frontal and lateral) for 14-label thoracic disease classification. Training
follows a two-phase curriculum:

- **Phase 1** fine-tunes a single-view EVA-X encoder.
- **Phase 2** freezes that encoder and trains a token-merged multi-view encoder and a View-Aware
  Attention Fusion (VAAF) head. The fused output corrects the frozen single-view prediction as
  a residual.

All reported runs used a Kaggle TPU v3-8 (`torch_xla`). The same code runs on CUDA GPUs and, for smoke
tests, on CPU.

![Cross-EvaNet architecture](images/architecture.png)

## Contents

- [Method](#method)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Weights and checkpoints](#weights-and-checkpoints)
- [Data](#data)
- [Patient splits](#patient-splits)
- [Phase 1: single-view fine-tuning](#phase-1-single-view-fine-tuning)
- [Phase 2: multi-view fusion training](#phase-2-multi-view-fusion-training)
- [Inference and submission](#inference-and-submission)
- [Evaluation protocol (per image)](#evaluation-protocol-per-image)
- [In-domain evaluation (paired validation studies)](#in-domain-evaluation-paired-validation-studies)
- [CheXpert zero-shot evaluation](#chexpert-zero-shot-evaluation)
- [Reproducing the paper](#reproducing-the-paper)
- [Hardware and precision](#hardware-and-precision)
- [Citation](#citation)
- [License](#license)

## Method

Let `X^1, X^2` be two radiographs of one study. The model has two pathways and a fusion head, and
every operation below is at the logit level (before the sigmoid).

| Component | What it does | Trained in Phase 2 |
|---|---|---|
| Single-view encoder `E_s` + head | EVA-X ViT-B (D = 768, 12 layers, 12 heads, SwiGLU, 2-D RoPE) fine-tuned in Phase 1. Each image is encoded on its own with shared weights. The pooled feature `z_k` is the layer-normalised mean of the patch tokens (the CLS token is not pooled). The shared head gives `y_k = W_h z_k + b_h`. | frozen (`eval()`, `no_grad`) |
| Multi-view encoder `E_m` | A second EVA-X ViT-B that reads both images as one sequence of `1 + 2N` tokens. Before merging, each image's patch tokens get an l2-normalised, Xavier-initialised slot embedding `v_k / ||v_k||`. The merged sequence keeps the CLS token of slot 1, and the RoPE grid is replicated for slot 2. `z'` is the layer-normalised mean of the `2N` patch tokens. `E_m` starts from the public EVA-X MIM weights; `pos_embed` is resampled bicubically from 14x14 to 28x28. | yes |
| VAAF | Takes the three tokens `[z_1; z_2; z']` plus l2-normalised view embeddings and applies LayerNorm. C = 14 layer-normalised disease queries attend over these tokens (`N_h = 4` heads). A gated residual follows (gate bias `b_g` initialised to -2.0), then `LN`, dropout 0.2 and a shared per-label classifier `W_c`, which give the residual logits `y'`. The head-averaged attention `alpha` in `[0,1]^{14x3}` is the Disease-View Affinity Matrix. See [`vraf_theory.md`](vraf_theory.md) for the equations. | yes |
| Output (Eq. 1 / Eq. 17) | `y = 1/2 * (1/2 * (y_1 + y_2) + y')` | - |

**Slots, not projections.** The index `k` is the slot an image occupies in the pair.
Frontal/lateral identity is never given to the network. Training, validation and inference read
no view metadata (AP/PA/lateral); only the CheXpert cohort builder uses it, to select each
frontal-lateral pair.

- **Phase-2 training:** a study with `n >= 2` images contributes two distinct images drawn
  uniformly at random, in random order. A study with one image is paired with itself and still
  goes through the fusion path.
- **Validation and inference:** images are taken in file order. A study with `n > 2` images is
  scored on its `n - 1` windows of consecutive images, and the sigmoid probabilities of the
  windows are averaged.

**Single-view route (inference).** A study with exactly one image returns the frozen single-view
logits `y_1` unchanged; `E_m` and VAAF are not used for it. Routing depends only on the image count.
It is on by default. Disable it with `--no-single-view-route`, and single-image studies are then
self-paired through the fusion path, as in training.

Paired studies are therefore scored on the Eq. (1) scale, `1/2 * (...)`, while routed single-image
studies are scored on the scale of `y_1`. The study prediction is assigned to every image of the
study; evaluation is per image (see [Evaluation protocol](#evaluation-protocol-per-image)).

**Objective.** The objective is the Asymmetric Polynomial Loss (APL): `gamma- = 5`, `gamma+ = 0`,
clip 0.05, `eps1+ = 1`, `eps2+ = -2.5`, `eps1- = 0`, summed over labels and batch. It is used in
both phases. Other objectives are selectable with `--loss` (see [Losses](#losses)).

**Parameter budget (Table 4).**

| Component | Parameters |
|---|---|
| `E_s` + head | 86.31 M |
| `E_m`, including the 1,536 view-embedding parameters | 86.31 M |
| VAAF | 3.56 M |
| Trainable in Phase 2 | 89.88 M |
| Total | 176.19 M |

`E_m` keeps an unused classification head, which Table 4 counts. The same head is the linear
head of the "without VAAF" ablation.

## Repository layout

```
Cross-EvaNet/
├── README.md
├── vraf_theory.md            # VAAF equations (paper Sec. 4.2.3) and their mapping to the code
├── requirements.txt
├── configs/
│   ├── phase1_{A_matched,B_matched,full}_448[_asl].json   # per-experiment Phase-1 configs (APL / ASL)
│   ├── phase2_{A,B}_{matched,full}_{apl,asl}.json   # per-experiment Phase-2 configs
│   ├── split_{A,B}.csv       # persisted 90/10 patient splits (created on first use)
│   └── chexpert_cohort_ids.csv   # label-free CheXpert study list (build_chexpert_cohort --ids-output)
├── images/architecture.png
├── scripts/
│   ├── train_phase1.py       # Phase 1: single-view fine-tuning
│   ├── train.py              # Phase 2: multi-view fusion training
│   ├── inference.py          # Grand X-Ray SLAM test inference and submission CSV
│   ├── inference_single_view.py   # single-view submission from a Phase-1 checkpoint (each image alone)
│   ├── build_chexpert_cohort.py   # CheXpert paired-study cohort
│   ├── evaluate_chexpert.py       # per-image fused (study-level) + single-view logits (CheXpert or validation split)
│   └── statistics.py              # study-cluster bootstrap CIs, paired DeLong, paired bootstrap
└── src/
    ├── config.py             # TrainConfig (Phase 2) and JSON config loading
    ├── constants.py          # LABEL_COLUMNS, NUM_LABELS, seed_everything
    ├── losses.py             # BCE, Focal, Two-Way, ZLPR, ASL, APL; build_loss
    ├── metrics.py            # per-label AUC with -1 labels masked
    ├── precision.py          # autocast / GradScaler per device
    ├── inference.py          # run_inference, predict_images (per-image rows), submission building
    ├── dataset/
    │   ├── train_dataset.py  # SingleViewXRayDataset (Phase 1), MultiViewXRayDataset (Phase 2)
    │   ├── test_dataset.py   # MultiViewEvalDataset (deterministic), TestMultiViewDataset, TestSingleViewDataset
    │   ├── study.py          # study grouping, pairing rules, study labels
    │   ├── splits.py         # label CSV reading, persisted patient split
    │   ├── transforms.py     # Strong Augment and validation transforms
    │   └── loaders.py        # Phase-2 DataLoaders
    ├── models/
    │   ├── eva_backbone.py   # EVA-X ViT-B, EVA-X / Phase-1 checkpoint loaders
    │   ├── multiview.py      # E_m: token merging with view embeddings
    │   ├── vaaf.py           # View-Aware Attention Fusion
    │   └── triple_branch.py  # TripleBranchEVA (E_s + E_m + fusion head), builders, loader
    └── train/train_loop.py   # train_one_epoch, evaluate, build_scheduler
```

Run every command from the repository root with `python -m scripts.<name>` so that `src` is
importable. Alternatively, set `PYTHONPATH=.` and run `python scripts/<name>.py`.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **CUDA:** install the `torch` / `torchvision` build that matches your CUDA version first
  (https://pytorch.org), then `pip install -r requirements.txt`.
- **TPU (Kaggle TPU v3-8 for the reported runs; Colab or Cloud TPU VM also work):** use a runtime that ships
  `torch_xla` matching its `torch`, or install the matching `torch_xla` wheel. `torch_xla` is
  imported only when `--use-tpu` is passed.

`timm >= 0.9.6` is required, because the multi-view encoder uses `Eva._pos_embed` to obtain the
RoPE embedding. EVA-X's own repository pins `timm==0.9.0`, which is too old for this code.

## Weights and checkpoints

The EVA-X ViT-B masked-image-modelling (MIM) checkpoint initialises both Phase 1 and `E_m`. It is
released by the EVA-X authors ([hustvl/EVA-X](https://github.com/hustvl/EVA-X), Apache-2.0):

```bash
mkdir -p weights
wget -O weights/eva_x_base_patch16_merged520k_mim.pt \
  https://huggingface.co/MapleF/eva_x/resolve/main/eva_x_base_patch16_merged520k_mim.pt
```

The loader in `src/models/eva_backbone.py` (`load_evax_init_weights`) does the following:

- unwraps `model` / `module` / `state_dict` containers and strips the `module.`, `model.` and
  `single_model.` prefixes;
- remaps EVA-02 key names, drops `mask_token`, `lm_head` and the RoPE buffers, and resamples
  `pos_embed` bicubically to the 448 grid;
- fails unless the only missing keys are `head.weight` / `head.bias` and no key is unexpected.

The loader also accepts a Phase-1 checkpoint and detects which kind it was given.

The file names below are the ones the configs use; all are placeholders you may change.

| File | Produced by | Used as |
|---|---|---|
| `weights/eva_x_base_patch16_merged520k_mim.pt` | EVA-X release | Phase 1 `--init-ckpt`; Phase 2 `--mv-init-ckpt` (`E_m`) |
| `outputs/phase1_{A,B}_matched_448.pth` | Phase 1, matched-data, APL | `--pretrained` (`E_s`) for `phase2_*_matched_apl` |
| `outputs/phase1_full_448.pth` | Phase 1, full data, APL | `--pretrained` for `phase2_*_full_apl` |
| `outputs/phase1_{A,B}_matched_448_asl.pth`, `outputs/phase1_full_448_asl.pth` | Phase 1 with `--loss asl` | `--pretrained` for the `*_asl` configs |
| `outputs/phase2_{A,B}_{matched,full}_{apl,asl}.pth` | Phase 2 (final epoch) | `--checkpoint` for inference and CheXpert |

- **Phase-1 checkpoint:** an `eva_x_base_patch16(img_size=448, num_classes=14)` state dict,
  bare or under `"model"`. It must include `head.weight` [14, 768] and `head.bias` [14]; `E_s`
  is loaded strictly.
- **Phase-2 checkpoint:** `{"model": TripleBranchEVA.state_dict(), "config": ..., "epoch": ...}`.
  Load it with `src.models.load_triple_branch_model(path, 14)`, which infers the fusion head type
  from the keys.

Weights are not stored in git (`*.pt`, `*.pth` and `outputs/` are ignored).

## Data

**Grand X-Ray SLAM** ([Division A](https://www.kaggle.com/competitions/grand-xray-slam-division-a),
[Division B](https://www.kaggle.com/competitions/grand-xray-slam-division-b)). The configs expect
this layout:

```
data/
├── division_a/
│   ├── train_mv.csv     # training CSV of the division
│   ├── train/           # training images (file names as in Image_name)
│   └── test/            # test images, named <Patient>_<Study>_*.jpg
└── division_b/  (same)
```

- **Training CSV:** columns `Patient_ID`, `Study`, `Image_name` and the 14 label columns:
  Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture,
  Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia,
  Pneumothorax, Support Devices.
- **Labels:** blank labels become 0, and every label must then be 0 or 1. The study label is the
  per-image label, which must be constant within a study. If it is not, the per-label maximum is
  used and a warning is printed.
- **Test studies:** grouped by the first two `_`-separated fields of the file name, and the
  images of a study are sorted by file name.

**CheXpert** (zero-shot only): download it from the
[Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) under its Research
Use Agreement.

## Patient splits

Each division is split 90/10 by `Patient_ID` with seed 1337. The split is saved as
`configs/split_<DIV>.csv` (columns `Patient_ID`, `split` in {`train`, `val`}), and both phases
read the same file.

- It is created on first use by `src.dataset.split_by_patient`: patients in first-appearance
  order are shuffled with `RandomState(1337)`, and the last 10% become validation. This matches
  the original training script only when it is built from the same CSV with the same row order,
  so create both files once, before any run, and commit them.
- Both training scripts create a missing split file from the division's CSV. A full-data
  Phase-1 run creates both files, then excludes the validation patients of both divisions.
- To create the files explicitly:

```bash
python - <<'EOF'
from src.dataset import load_patient_split, read_train_csv
for div in ("A", "B"):
    df = read_train_csv(f"data/division_{div.lower()}/train_mv.csv")
    load_patient_split(f"configs/split_{div}.csv", df, val_frac=0.1, seed=1337)
EOF
```

`.gitignore` allows `configs/*.csv` and `configs/*.txt`, so commit the split files (and the
label-free CheXpert study list) to keep them fixed.

## Phase 1: single-view fine-tuning

Phase 1 fine-tunes EVA-X ViT-B on individual radiographs, initialised from the EVA-X MIM
checkpoint. Paper settings:

| Setting | Value |
|---|---|
| Resolution | 448 |
| Epochs | 10 |
| Learning rate | 1e-4, AdamW |
| Weight decay | 0.05 |
| Batch size | 16 |
| Schedule | linear warmup over the first 5% of steps, then cosine decay to 10% of the peak |
| Loss | APL |
| Augmentation | Strong Augment |
| Seed | 1337 |

The paper does not state the remaining settings. This implementation uses drop-path 0.2,
gradient clipping 1.0, no gradient accumulation, no layer-wise LR decay and no EMA.

`--train-csv`, `--train-dir` and `--split-csv` take one entry per division, and the number of
entries selects the regime:

- **Matched-data regime (one division):** train on the target division's training patients
  only, i.e. the `train` rows of `configs/split_<DIV>.csv`.
- **Full-data regime (both divisions):** train on the merged training partitions of both
  divisions, **excluding the validation patients of both divisions**.

Validation AUC (per image) is logged each epoch for monitoring only, and the final epoch is
saved.

```bash
# Matched-data (add --use-tpu on TPU)
python -m scripts.train_phase1 --config configs/phase1_A_matched_448.json
python -m scripts.train_phase1 --config configs/phase1_B_matched_448.json
# Full data (Divisions A + B)
python -m scripts.train_phase1 --config configs/phase1_full_448.json

# Without a config
python -m scripts.train_phase1 \
  --train-csv data/division_b/train_mv.csv --train-dir data/division_b/train \
  --split-csv configs/split_B.csv \
  --init-ckpt weights/eva_x_base_patch16_merged520k_mim.pt \
  --save-path outputs/phase1_B_matched_448.pth
```

Flags:

```
--config --train-csv CSV [CSV ...] --train-dir DIR [DIR ...] --split-csv CSV [CSV ...]
--val-split --init-ckpt --save-path --img-size --drop-path-rate --batch-size --num-workers --lr
--weight-decay --epochs --warmup-pct --grad-accum-steps --clip-grad
--loss {bce,focal,twoway,zlpr,asl,apl} --seed --use-tpu
```

`--train-csv`, `--train-dir`, `--split-csv` and `--init-ckpt` are required.

- **ASL rows of Table 7:** `configs/phase1_{A_matched,B_matched,full}_448_asl.json` (`--loss asl`,
  saved as `outputs/phase1_<...>_448_asl.pth`), which the `phase2_*_asl` configs load.
- **224 single-view baselines:** `--img-size 224` with a new `--save-path`. RoPE uses raw patch
  coordinates at every resolution (reference grid = input grid: 28x28 at 448, 14x14 at 224), so
  every script builds the same RoPE for a given size. A 224 checkpoint cannot serve as `E_s` for
  the 448 Phase 2.
- **Loss ablation (Table 11):** 224, Division B, `--loss {bce,focal,twoway,zlpr,asl,apl}`.

## Phase 2: multi-view fusion training

`E_s` is loaded strictly from the Phase-1 checkpoint and frozen. `E_m` is initialised from the
EVA-X MIM checkpoint, and `E_m`, the view embeddings and VAAF are trained. Paper settings:

| Setting | Value |
|---|---|
| Resolution | 448 |
| Epochs | 8 |
| Learning rate | 1e-4, AdamW |
| Weight decay | 0.05, over all trainable parameters |
| Batch size | 10 per step, gradient accumulation 4 (effective 40) |
| Drop-path | 0.2 |
| Gradient clipping | 1.0 |
| Schedule | warmup 5%, cosine to 10% |
| VAAF | `N_h = 4`, gate bias -2.0, dropout 0.2 |
| Loss | APL |
| Seed | 1337 |

**No checkpoint selection.** Training runs the full epoch budget and saves the final epoch.
Validation AUC is computed per image each epoch, for monitoring only. Three numbers are logged:

- all images, deployed output (each image receives its study's prediction);
- images of paired studies, fused output (the study's prediction on each image);
- images of paired studies, single-view `y_k` of each image on its own.

Run one of the eight configs (`configs/phase2_{A,B}_{matched,full}_{apl,asl}.json`):

```bash
python -m scripts.train --config configs/phase2_B_full_apl.json --use-tpu    # TPU
python -m scripts.train --config configs/phase2_B_full_apl.json              # CUDA / CPU
```

Settings are applied in this order: `TrainConfig` defaults, then the JSON config, then CLI flags.
Without a config:

```bash
python -m scripts.train \
  --train-csv data/division_b/train_mv.csv --train-dir data/division_b/train \
  --split-csv configs/split_B.csv \
  --pretrained outputs/phase1_B_matched_448.pth \
  --mv-init-ckpt weights/eva_x_base_patch16_merged520k_mim.pt \
  --save-path outputs/phase2_B_matched_apl.pth
```

Flags (every `TrainConfig` field):

```
--config --train-csv --train-dir --val-split --split-csv --pretrained --mv-init-ckpt --save-path
--img-size --num-views {2} --drop-path-rate --fusion-head {vaaf,linear} --[no-]single-view-route
--batch-size --num-workers --lr --weight-decay --epochs --warmup-pct --grad-accum-steps
--clip-grad --loss {bce,focal,twoway,zlpr,asl,apl} --seed --use-tpu
```

`--pretrained`, `--mv-init-ckpt`, `--split-csv` and `--train-dir` are required. `--pretrained-2`
is a deprecated alias of `--mv-init-ckpt`.

**Without-VAAF ablation (Table 9).** `--fusion-head linear` replaces VAAF with `E_m`'s linear
head, `y' = W z' + b`, combined in the same Eq. (1) form: `y = 1/2 * (1/2 * (y_1 + y_2) + y')`.

### Losses

| `--loss` | Objective | Settings | Reduction |
|---|---|---|---|
| `apl` (default) | Asymmetric Polynomial Loss | `gamma- = 5`, `gamma+ = 0`, clip 0.05, `eps1+ = 1`, `eps2+ = -2.5` (applied to `1/2 * (1 - p)^2`), `eps1- = 0`; focusing weight detached | sum |
| `asl` | Asymmetric Loss | `gamma- = 4`, `gamma+ = 1`, clip 0.05 | sum |
| `focal` | Focal Loss | `gamma = 2` | mean |
| `twoway` | Two-Way Loss | `Tp = 4`, `Tn = 1` | see `src/losses.py` |
| `zlpr` | ZLPR | none | batch mean |
| `bce` | BCEWithLogits | none | mean |

## Inference and submission

```bash
python -m scripts.inference \
  --checkpoint outputs/phase2_B_full_apl.pth \
  --test-dir data/division_b/test \
  --output submissions/phase2_B_full_apl.csv \
  [--use-tpu] [--no-single-view-route]
```

Other flags: `--img-size 448 --num-views 2 --batch-size 10 --num-workers 4`.

- Images of a study are taken in file order.
- Single-image studies use the single-view route.
- Studies with more than two images average the probabilities of their consecutive pairs.
- Every image of a study receives the study's probabilities in the per-image submission CSV
  (one prediction per study; the one image of a single-image study receives `sigmoid(y_1)`).
- No test-time augmentation is used.

**Single-view submission** (the single-view rows of Tables 5 and 7).
`scripts/inference_single_view.py` loads a Phase-1 checkpoint strictly and scores every test image
on its own with the validation transform: each image gets `sigmoid(y)` of that image, with no
pairing and no averaging over the images of a study. The CSV has the same layout as above.

```bash
python -m scripts.inference_single_view \
  --checkpoint outputs/phase1_B_matched_448.pth \
  --test-dir data/division_b/test \
  --output submissions/phase1_B_matched_448.csv \
  [--img-size 224] [--use-tpu]
```

Other flags: `--batch-size 16 --num-workers 4`. Pass `--img-size 224` for a Phase-1 checkpoint
trained at 224.

## Evaluation protocol (per image)

All AUCs in this repository are computed **per image** (radiograph), the unit of the Grand X-Ray
SLAM leaderboard. Every image carries its study's labels.

- **Single-view (EVA-X) comparator:** the frozen single-view encoder scores every image on its
  own: the slot-1 image gets `y_1`, the slot-2 image gets `y_2`. There is no averaging over the
  images of a study.
- **Cross-EvaNet (multi-view):** one study-level prediction (Eq. 1; for a study with more than two
  images, the average of the window probabilities) is assigned to every image of the study. A
  single-image study uses the single-view route, so its image gets `y_1`.
- **AUC:** per label over the image rows. Uncertain (-1) labels are excluded per label; the images
  of a study share them. Macro average = mean over the 14 labels; CheXpert Core 5 = mean over
  Atelectasis, Cardiomegaly, Consolidation, Edema and Pleural Effusion.
- **Uncertainty:** the bootstrap resamples **studies** as clusters, so all images of a drawn study
  enter the replicate together (`--resample patient` resamples patients instead; an extra, not
  used in the paper). See step 3 of the CheXpert section.

Both outputs come from one forward pass of the Phase-2 model, so the comparison isolates the
multi-view pathway: the study-level combination of both images (`1/2 * (y_1 + y_2)`) plus the
fusion term `y'`, against each image's own `y_k`. No test-time augmentation is used.

## In-domain evaluation (paired validation studies)

Table 6 compares, on the images of the paired validation studies (`>= 2` images) of the Division B
split:

- the fused output: the study's Eq. (1) prediction, assigned to each of its images;
- the single-view comparator: the frozen Phase-1 encoder inside the model applied to each image on
  its own (`y_k`), in the same forward pass.

Evaluation is deterministic: file order, no augmentation. The paper does not state which Phase-2
regime Table 6 used, so pass the checkpoint you want to report.

`scripts/evaluate_chexpert.py` has a validation-split mode for this. It keeps only the studies
with `>= 2` images. A study with `n > 2` images is scored on its `n - 1` consecutive pairs, whose
fused sigmoid probabilities are averaged; each image keeps its own single-view logits. The split
file must already exist.

```bash
python -m scripts.evaluate_chexpert --checkpoint outputs/phase2_B_full_apl.pth \
  --train-csv data/division_b/train_mv.csv --split-csv configs/split_B.csv \
  --image-root data/division_b/train --output outputs/divB_val_logits.csv
```

The script prints the per-image fused and single-view macro AUC. It writes one row per image
(`study_id`, `patient_id`, `image`, `slot`, `n_images`, `label_<L>`, `single_<L>`, `fused_<L>`);
the per-label AUCs of Table 6 can be read from them. The paper attaches no CIs or tests to Table 6.
If you want them anyway, `scripts/statistics.py` accepts this file; its `--resample patient`
option (not used in the paper) keeps all studies of a validation patient together:

```bash
python -m scripts.statistics --predictions outputs/divB_val_logits.csv \
  --output outputs/divB_val_statistics.csv [--resample patient]
```

`--per-study` writes one row per study with the earlier comparator `1/2 * (y_1 + y_2)` instead. It
is not the paper's protocol. `--include-single` also keeps the single-image validation studies
(their one image gets `y_1` in both columns), i.e. all validation images as on the leaderboard;
Table 6 uses the paired studies only.

`scripts/train.py` also logs, after every epoch (via `src.train.evaluate`), the per-image macro
AUC of the deployed output over all validation images and, on the images of paired studies, the
fused and single-view macro AUC. The last epoch's numbers belong to the saved checkpoint.

## CheXpert zero-shot evaluation

This section reproduces Table 8. The model is the **full-data Division B** Phase-2 checkpoint at
448, applied without any fine-tuning, weight adaptation or threshold recalibration.

**1. Build the cohort** (`scripts/build_chexpert_cohort.py`). The protocol:

- Parse patient and study from the `Path` column. Keep studies with at least one frontal and at
  least one lateral image; AP and PA both count as frontal.
- Keep one study per patient, the chronologically earliest, which is the lowest study index. The
  public metadata has no dates.
- In that study, take the first frontal and the first lateral image in metadata order. The two
  images keep their metadata order in the slots (`image_1`, `image_2`), as file order does at
  inference; projection is used only to select the pair.
- Labels: blank becomes 0, 1 and 0 are kept as they are, and uncertain stays -1.

The paper's cohort has 19,539 studies and 39,078 images. `--csv` may be repeated to concatenate
several metadata files in order. The paper does not state which CheXpert files were used.

- `--output` (default `outputs/chexpert_cohort.csv`, git-ignored) is the study list with CheXpert
  labels, the input of step 2. Keep it local: CheXpert labels fall under the Research Use
  Agreement.
- `--ids-output` writes the same list without labels (`study_key`, `patient_id`, `study_index`,
  `image_1`, `image_2`). Commit and publish this one. Anyone with CheXpert access regenerates the
  labelled list with the same command and can compare it with the published ids.

```bash
python -m scripts.build_chexpert_cohort \
  --csv data/CheXpert-v1.0-small/train.csv \
  --output outputs/chexpert_cohort.csv --ids-output configs/chexpert_cohort_ids.csv
```

**2. Predict** (`scripts/evaluate_chexpert.py`).

- It uses the in-domain validation transform: bicubic resize of the shorter side to 448, a 448
  center crop and the EVA-X normalisation.
- For every study, one forward pass gives the fused study logits (Eq. 1), written for both of its
  images, and the single-view logits of each image on its own (`y_1` for `image_1`, `y_2` for
  `image_2`). The output has one row per image (2 x 19,539 = 39,078 rows for the paper's cohort).
- `--image-root` is the directory that the `Path` entries are relative to.
- Mixed precision follows training; use `--no-amp` for fp32.

```bash
python -m scripts.evaluate_chexpert --checkpoint outputs/phase2_B_full_apl.pth \
  --study-list outputs/chexpert_cohort.csv --image-root data \
  --output outputs/chexpert_zeroshot_logits.csv [--use-tpu]
```

**3. Statistics** (`scripts/statistics.py`). The AUC of each label is computed over the image
rows. Uncertain (-1) labels are excluded per label from the AUC, the bootstrap and DeLong. The
script reports:

- Two-sided 95% percentile bootstrap CIs with `B = 2000` resamples and seed 42, resampling
  studies as clusters: all images of a drawn study enter the replicate together. The same
  replicates are used for both arms, every label and the aggregates.
- The two-sided paired DeLong test per label, on the image rows. `--raw-output` also holds, for
  comparison, the two-sided cluster-bootstrap p-value of each label (`p_boot`, with
  `z_boot = Delta / SE_boot`) and the Holm-adjusted DeLong p-values; the paper reports neither.
- For the macro average and the CheXpert Core 5 labels (Atelectasis, Cardiomegaly,
  Consolidation, Edema, Pleural Effusion), `z = Delta / SE_boot` and the empirical paired
  cluster-bootstrap p-value `2 * min(P(Delta* <= 0), P(Delta* >= 0))`.
- The counts: images and studies overall, and per label the evaluable images and studies and
  the positive and negative images.

The fused scores and labels of a per-image file must be constant within a study, and each
(study, image) row must appear once. Clusters are numbered in sorted-id order, so the replicates
do not depend on the row order of the file.

The output follows the column layout of Table 8, followed by the columns `N images`, `N studies`,
`N positive` and `N negative`. The script needs numpy, scipy, pandas and scikit-learn, not torch.

```bash
python -m scripts.statistics --predictions outputs/chexpert_zeroshot_logits.csv \
  --output outputs/chexpert_statistics.csv [--raw-output outputs/chexpert_statistics_raw.csv]
python -m scripts.statistics --selftest    # self-check on synthetic data
```

## Reproducing the paper

| Paper result | How |
|---|---|
| Tables 5 and 7: multi-view rows | Phase 1 at 448 (matched or full) -> `scripts.train --config configs/phase2_<DIV>_<regime>_<loss>.json` -> `scripts.inference` -> Kaggle submission |
| Tables 5 and 7: single-view rows (224 / 448) | Phase 1 at `--img-size 224` or 448 -> `scripts.inference_single_view` (each image scored alone) -> Kaggle submission |
| Table 4: parameters | See the snippet below (89.88 M trainable, 176.19 M total) |
| Table 6: per-label, validation | `scripts.evaluate_chexpert` in validation-split mode ([details](#in-domain-evaluation-paired-validation-studies)) |
| Table 8: CheXpert zero-shot | [CheXpert zero-shot evaluation](#chexpert-zero-shot-evaluation) |
| Table 9: without VAAF | Phase 2 with `--fusion-head linear` |
| Table 11: losses | Phase 1 at 224, Division B, `--loss ...` |

```bash
python - <<'EOF'
from src.config import TrainConfig
from src.constants import NUM_LABELS
from src.models import build_triple_branch_model
m = build_triple_branch_model(NUM_LABELS, TrainConfig(), load_pretrained=False)
count = lambda ps: sum(p.numel() for p in ps) / 1e6
print(f"trainable {count(p for p in m.parameters() if p.requires_grad):.2f} M, total {count(m.parameters()):.2f} M")
EOF
```

Not included in this repository:

- the ResNet-152 / ViT-L-16 baselines and the Weak / Trivial Augment policies of Table 10;
- the Grad-CAM figure (Fig. 2);
- the organiser-scored leaderboard. Kaggle computes it from the submission CSV, and its labels
  are not released.

## Hardware and precision

- **TPU (`--use-tpu`):** `torch_xla` with `bfloat16` autocast. Master weights, gradients and
  AdamW state stay fp32. Do **not** set `XLA_USE_BF16=1` (or `XLA_DOWNCAST_BF16=1`): it would
  run the whole model in pure bf16, so the training, inference and evaluation scripts stop with an
  error when it is set.
  - On XLA, the fusion path is computed for all rows and replaced by `y_1` for routed rows. This
    keeps shapes static, and the outputs are identical to skipping those rows.
  - The XLA RNG is seeded too.
- **CUDA:** `bfloat16` autocast when the GPU supports it; otherwise `float16` with a GradScaler,
  which unscales before clipping.
- **CPU:** fp32 without autocast. Suitable for smoke tests only.
- **All devices:** losses are computed in fp32 outside autocast; evaluation sigmoids in float64
  on the host.
- **Seeds:** training uses 1337 for the split, augmentation and initialisation; the statistics
  use 42.
- **Memory:** a Phase-2 step processes 10 studies with 1,569 tokens each. If memory is short,
  lower `--batch-size` and raise `--grad-accum-steps` so that their product stays 40.

## Citation

If you use this code, please cite the paper and EVA-X:

```bibtex
@misc{le2026crossevanet,
  title  = {Cross-EvaNet: Multi-view Vision Transformer Fusion for Automated Thoracic Disease Classification},
  author = {Le, Minh Hung and Huynh, Do Tan Thanh and Nguyen, Minh Son},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/HuynhDoTanThanh/Cross-EvaNet-Multi-View-Chest-X-RayClassificationvia-Hybrid-Fusion-of-Vision-Transformer-Features}
}

@article{yao2025evax,
  title   = {{EVA-X}: a foundation model for general chest x-ray analysis with self-supervised learning},
  author  = {Yao, Jingfeng and Wang, Xinggang and Song, Yuehao and Zhao, Huangxuan and Ma, Jun and Chen, Yajie and Liu, Wenyu and Wang, Bo},
  journal = {npj Digital Medicine},
  volume  = {8},
  pages   = {678},
  year    = {2025},
  doi     = {10.1038/s41746-025-02032-z}
}
```

## License

A license for this code has not been added yet.

- **EVA-X code and weights:** Apache-2.0 ([hustvl/EVA-X](https://github.com/hustvl/EVA-X)).
- **Grand X-Ray SLAM data:** subject to the Kaggle competition rules.
- **CheXpert:** subject to the Stanford CheXpert Research Use Agreement.
