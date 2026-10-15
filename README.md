# SSL-Rebalance: Melanoma Detection under Severe Class Imbalance

Research code and experiment notebooks for the manuscript:

> **Shifting vs. Lifting Melanoma Recall under Severe Class Imbalance: An Empirical Study of Loss-Level, Representation-Level, and Generative Interventions**
> I. S. Tunib, Md. Sazzadur Ahamed, M. F. Bin

> **Status:** manuscript in preparation. This repository holds the implementation, experiment notebooks, selected figures, and configuration used in the study. Raw datasets and trained checkpoints are not included.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c) ![Task](https://img.shields.io/badge/task-melanoma%20classification-lightgrey) ![Use](https://img.shields.io/badge/use-research%20only-red)

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/fig_two_stage_pipeline.png?raw=true" width="760" alt="Two-stage training pipeline: SimCLR pre-training followed by adaptive fine-tuning">
</p>
<p align="center"><em>The two-stage pipeline: SimCLR pre-training on 24,000 unlabeled ISIC images, then fine-tuning on a small labeled split with melanoma-targeted rebalancing.</em></p>

## Contents

- [Why this project](#why-this-project)
- [Summary of findings](#summary-of-findings)
- [Method](#method)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Experiment-to-notebook map](#experiment-to-notebook-map)
- [Setup, data, and configuration](#setup-data-and-configuration)
- [Running the experiments](#running-the-experiments)
- [Limitations](#limitations)
- [Citation and license](#citation-and-license)

## Why this project

Melanoma is rare in dermoscopic datasets. A classifier that predicts the majority class almost every time can post a high accuracy while missing nearly every melanoma, and in this setting accuracy is the wrong number to trust. This project studies a compact, reproducible three-class pipeline (BKL, MEL, NV) on ISIC 2019 and asks a narrow question: **when labeled data are scarce and severely imbalanced, what actually improves melanoma recall, and what only appears to?**

The study is deliberately framed as a resource-constrained engineering problem rather than a state-of-the-art attempt. The labeled set is tiny (691 images, 35 of them melanoma, about 10:1 NV to MEL), the encoder has about 0.1M parameters, and inference takes about 1.28 ms per image on a single GPU. These limits are treated as design inputs, and every component is evaluated under the same three-seed protocol.

## Summary of findings

| Question | Finding |
|---|---|
| Can a loss-level change improve melanoma recall? | It can **shift** recall, not **lift** it. Adaptive class weighting raises melanoma recall from 41.7% to 45.8% but lowers macro-F1, and its operating points sit on or below the fixed-weight precision–recall curve. |
| Does a representation-level change help? | Yes, modestly. Supervised contrastive loss with squeeze-and-excitation attention raises melanoma Average Precision from 0.2085 to 0.261 ± 0.025 with melanoma precision essentially unchanged. |
| Does synthetic GAN augmentation help? | No. It reached only 16.7% ± 11.8% melanoma recall, and feature-space diagnostics show mode collapse. |
| Does internal recall transfer to a new dataset? | No. On HAM10000, melanoma recall of the SSL + rebalancing model fell to 6.6%. A small real supplement (41 HAM10000 melanoma images) restored it to 29.9% ± 9.6%. |
| Are the adaptive-weighting gains statistically solid? | No. Bootstrap intervals overlap substantially, and a 15-run cross-validation estimate stays within noise of the three-seed result. |

## Method

### Data and splits

ISIC 2019 provides 25,331 dermoscopic images. 24,000 are used unlabeled for SimCLR pre-training, and a 1,331-image labeled candidate set yields 691 images (BKL, MEL, NV) for supervised fine-tuning, split 483 / 104 / 104 into train / validation / test. HAM10000 is used only for external validation.

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/fig_data_split_updated.png?raw=true" width="760" alt="ISIC 2019 data usage and the train/validation/test split">
</p>
<p align="center"><em>ISIC 2019 data usage. The unlabeled pool feeds Stage 1; the labeled set feeds Stage 2.</em></p>

| Split | BKL | MEL | NV |
|---|---:|---:|---:|
| ISIC train (n = 483) | 99 | 35 | 349 |
| ISIC validation (n = 104) | 21 | 7 | 76 |
| ISIC test (n = 104) | 21 | 8 | 75 |
| HAM10000 external (n = 8,917) | 1,099 | 1,113 | 6,705 |

The internal test set contains only 8 melanoma images, so one changed prediction moves melanoma recall by 12.5 percentage points. This motivates the multi-seed protocol and the external validation.

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/class_samples.png?raw=true" width="640" alt="Representative ISIC 2019 images: BKL, MEL, NV">
</p>
<p align="center"><em>One representative image per class (BKL, MEL, NV). ISIC archive images are subject to their original licences; see the <a href="https://www.isic-archive.com/">ISIC archive</a>.</em></p>

### Encoder

All experiments share a compact three-block CNN encoder (Conv, BatchNorm, ReLU, MaxPool, with 32, 64, and 128 channels), followed by adaptive average pooling to a 128-dimensional feature vector. The classification head is a two-layer MLP (128 → 64 → 3) with dropout 0.3. A ResNet-18 baseline (ImageNet-pretrained, 11.2M parameters) is used for comparison.

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/3-Layer%20CNN%20architecture.png?raw=true" width="860" alt="Three-block CNN encoder architecture">
</p>
<p align="center"><em>The shared three-block encoder. In the SupCon + SE variant, a squeeze-and-excitation block follows each convolutional stage.</em></p>

### Stage 1: self-supervised pre-training

SimCLR with an NT-Xent loss (temperature 0.5) on the 24,000 unlabeled images: 20 epochs, batch size 64, Adam with learning rate 3e-4. The projection head (128 → 128 → 64) is discarded after pre-training.

### Stage 2: rebalanced fine-tuning

- **Melanoma 5x oversampling** and **heavy melanoma-specific augmentation** (flips, rotation up to 45°, colour jitter, affine translation and scaling)
- **Focal loss** with γ = 1.5 and class weights MEL = 4, BKL = 2, NV = 1
- Adam, learning rate 1e-4, batch size 16, gradient clipping at 1.0, early stopping with patience 7 (maximum 30 epochs)
- **Checkpoints selected by validation macro-F1**, not accuracy, because accuracy is dominated by the majority class

### The three interventions

| Level | Intervention | Idea |
|---|---|---|
| Loss | **Adaptive melanoma weight** | A bounded rule raises the melanoma loss weight when validation melanoma recall falls below a target (target 0.45, cap 8, step 0.5, three-epoch warm-up). |
| Representation | **SupCon + SE attention** | A supervised contrastive term (λ = 0.3, τ = 0.1) plus squeeze-and-excitation blocks after each convolutional stage, to improve the melanoma ranking itself. |
| Generative | **DCGAN augmentation** | A lightweight DCGAN with DiffAugment is trained on the 35 real melanoma images; 140 synthetic images match the 5x oversampling count. |

Every result is reported over three seeds (42, 123, 2024), with bootstrap confidence intervals and a 5-fold cross-validation re-estimate for robustness.

## Results

### Internal comparison (ISIC 2019 test set)

Mean ± standard deviation over three seeds unless noted.

| Model | Accuracy (%) | Macro-F1 | MEL recall (%) |
|---|---:|---:|---:|
| Simple CNN, real images (no SSL, no rebalancing) | 73.1 ± 0.8 | n/a | 0.0 ± 0.0 |
| ResNet-18 + heavy augmentation | 78.5 ± 2.5 | 0.591 ± 0.013 | 33.3 ± 5.9 |
| SSL only, no rebalancing (single seed) | 74.0 | 0.392 | 0.0 |
| Rebalancing only, no SSL (single seed) | 67.3 | 0.536 | 37.5 |
| **SSL + rebalancing (fixed weights)** | 64.7 ± 3.2 | 0.510 ± 0.021 | 41.7 ± 5.9 |
| SSL + adaptive weighting | 60.6 ± 1.4 | 0.483 ± 0.014 | 45.8 ± 5.9 |
| SSL + GAN-synthetic MEL | 62.2 ± 4.8 | 0.452 ± 0.023 | 16.7 ± 11.8 |
| SupCon + SE attention | 67.3 ± 2.5 | 0.512 ± 0.009 | 45.8 ± 14.4 |

Accuracy is misleading here: the simple CNN reaches 73.1% accuracy while detecting no melanoma at all.

### Shifting, not lifting

Adaptive weighting moves the decision boundary along a fixed underlying ranking. The precision–recall analysis shows its operating points (red crosses) lie on or slightly below the fixed-weight curve (blue), so the recall gain is bought with a comparable precision loss. Mean melanoma recall rises from 0.417 to 0.458 while mean precision falls from 0.201 to 0.172.

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/pr_curve_with_adaptive_ops.png?raw=true" width="620" alt="Fixed-weight precision-recall curve with adaptive-weight operating points overlaid">
</p>
<p align="center"><em>Fixed-weight melanoma PR curve (mean over three seeds, shaded ±1 SD; Average Precision 0.2085 ± 0.0132) with per-seed operating points for the fixed-weight model (blue circles) and the adaptive-weight model (red crosses).</em></p>

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/roc_pr_mel_3seed_updated.png?raw=true" width="860" alt="Per-seed ROC and precision-recall curves for the fixed-weight model">
</p>
<p align="center"><em>Per-seed ROC (left) and precision–recall (right) curves. The curves are step-shaped because the test set has only 8 melanoma images.</em></p>

### Lifting the ranking with a representation-level change

SupCon + SE attention raises melanoma Average Precision from 0.2085 to 0.261 ± 0.025, and at the default 0.5 threshold recall rises from 41.7% to 45.8% with melanoma precision essentially unchanged (20.1% vs. 19.9% ± 2.8%). The per-seed recall is wide (±14.4%) because a single prediction moves recall by 12.5 points.

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/ap_comparison.png?raw=true" width="560" alt="Per-class Average Precision for the fixed, adaptive, and SupCon + SE models">
</p>
<p align="center"><em>Per-class Average Precision across the three model variants. Only the melanoma bar rises; BKL and NV are roughly unchanged.</em></p>

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/roc_pr_curves.png?raw=true" width="860" alt="Pooled ROC and precision-recall curves for the three models">
</p>
<p align="center"><em>Pooled ROC (left) and precision–recall (right) curves for the fixed-weight, adaptive-weight, and SupCon + SE models. The SupCon + SE curve sits above the other two in the PR panel.</em></p>

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/confusion_matrices.png?raw=true" width="860" alt="Confusion matrices for the fixed, adaptive, and SupCon + SE models">
</p>
<p align="center"><em>Confusion matrices pooled over three seeds (n = 312 predictions per model). The fixed-weight model concentrates melanoma predictions in the NV and BKL columns, which is the majority-class behaviour this work targets.</em></p>

### Statistical robustness

Bootstrap 95% intervals (2,000 resamples, pooled over three seeds) overlap substantially for fixed and adaptive weighting, on both recall and precision. An independent 15-run estimate (3 seeds x 5-fold cross-validation over the 587 train + validation images, evaluated on the unchanged test set) gives 40.0% ± 5.2% melanoma recall, within noise of the three-seed result. The adaptive-weighting shift is therefore reported descriptively and is not claimed to be statistically significant.

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/bootstrap_ci_comparison.png?raw=true" width="620" alt="Bootstrap 95% confidence intervals for melanoma recall and precision">
</p>
<p align="center"><em>95% bootstrap confidence intervals for melanoma recall and precision (n = 312 pooled predictions per model).</em></p>

### Synthetic augmentation did not help

A DCGAN trained on only 35 melanoma images collapsed to near-duplicates. In the SSL encoder's feature space, synthetic images were closer to individual real melanoma images than real images are to each other (mean nearest-neighbour cosine 0.9565 vs. 0.8003), yet the Fréchet Encoder Distance was about four times larger (8.63 vs. 2.13). That pattern means a narrow, shifted distribution rather than diverse new samples, and the augmented model performed worse than oversampling alone.

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/synthetic_mel_grid.png?raw=true" width="620" alt="Synthetic melanoma samples generated by the DCGAN">
</p>
<p align="center"><em>Synthetic melanoma samples from the DCGAN. The repeated textures and colour patterns are a visual sign of mode collapse.</em></p>

### External validation on HAM10000

| Model | Accuracy | Macro-F1 | MEL recall | MEL precision |
|---|---:|---:|---:|---:|
| SSL + rebalancing | 0.718 | 0.459 | 6.6% | 33.9% |
| ResNet-18 | 0.774 | 0.568 | 29.3% | 34.2% |

The ranking inverts externally: the model that looks best on the internal split degrades most under domain shift. Pooling a **small** real supplement (41 HAM10000 melanoma images, 24 lesions) into training raises external melanoma recall from 6.6% to 29.9% ± 9.6% and internal recall to 58.3% ± 7.2%, without an accuracy collapse (68.6% ± 3.7%). Pooling the full spare pool (555 images) over-corrected the imbalance, collapsing internal accuracy to 46–60% as the model called almost everything melanoma.

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/ham_pooling_comparison.png?raw=true" width="560" alt="Melanoma recall with and without a modest HAM10000 supplement">
</p>
<p align="center"><em>Melanoma recall with and without a modest real HAM10000 supplement, on the internal test set and the external holdout (±1 SD over three seeds).</em></p>

### Training dynamics

<p align="center">
  <img src="https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready/blob/main/figures/finetune_curves_3seed_updated.png?raw=true" width="860" alt="Fine-tuning loss and accuracy curves across three seeds">
</p>
<p align="center"><em>Fine-tuning loss (left) and train/validation accuracy (right) for the SSL + rebalancing pipeline across seeds 42, 123, and 2024. Loss converges within 5–10 epochs, and validation accuracy plateaus around epoch 10–15.</em></p>

## Repository structure

```text
.
├── README.md
├── config.py                 # environment-variable based paths
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── shared/                   # models, losses, datasets, evaluation, resumable runs (used by v2 notebooks)
├── v1_notebooks/             # original experiments: baselines, ablations, main framework, external evaluation
├── v2_notebooks/             # revision experiments: adaptive weighting, PR analysis, SupCon + SE,
│                             #   bootstrap CI, k-fold CV, GAN augmentation, HAM10000 pooling
├── figures/                  # publication figures used in this README (see figures/README.md)
├── data/                     # data instructions (no raw images)
├── results/                  # generated metrics (large artifacts are git-ignored)
└── paper/                    # reserved for publication materials
```

## Experiment-to-notebook map

**`v1_notebooks/` (original experiments)**

| Notebook | What it does |
|---|---|
| `SSL_pre-train_24k_real.ipynb` | Stage 1 SimCLR pre-training on 24,000 unlabeled ISIC images |
| `Baseline_CNN_with_real_image.ipynb` | Simple CNN baseline; independent of the SSL stage |
| `ssl-only-no-rebalancing-macro.ipynb` | Ablation: SSL pre-training without rebalancing |
| `rebalancing-only-no-ssl-macro.ipynb` | Ablation: rebalancing without SSL pre-training |
| `rebalancing_only-3seeds-various_technique.ipynb` | Three-seed ablation of the rebalancing components |
| `ssl-finetune-3seed-macrof1.ipynb` | Full SSL + rebalancing framework, three seeds, macro-F1 checkpoint selection |
| `resnet18-baseline-3seeds.ipynb` | ResNet-18 comparison |
| `ham10000-external-evaluation.ipynb` | External validation on HAM10000 |

**`v2_notebooks/` (revision experiments)**

| Notebook | What it does |
|---|---|
| `ssl-finetune-adaptive-weights.ipynb` | Bounded adaptive melanoma class-weight rule |
| `adaptive-test-probs.ipynb` | Exports test-set probabilities from the adaptive checkpoints |
| `val-probs-ssl-rebalance-checkpoints.ipynb` | Exports validation-set probabilities from the SSL + rebalancing checkpoints |
| `precision-recall-analysis-3seed.ipynb` | Precision–recall analysis and melanoma Average Precision |
| `supcon-attention.ipynb` | SupCon loss and SE attention (`USE_SUPCON` / `USE_SE_BLOCK` toggles) |
| `bootstrap-ci.ipynb`, `bootstrap-ci-rerun.ipynb` | Bootstrap confidence intervals for melanoma recall and precision |
| `kfold-cv.ipynb` | 5-fold stratified cross-validation re-estimate (3 repeats) |
| `gan-based-synthetic-mel-generation.ipynb` | Trains the DCGAN on the 35 real melanoma images and evaluates the synthetic output |
| `addendum-real-vs-real-baseline-for-the-gan-simila.ipynb` | Real-vs-real reference for the similarity metrics |
| `ssl-rebalance-fine-tuning-with-gan-synthetic-mel.ipynb` | Fine-tuning with GAN-synthetic melanoma augmentation |
| `ham-pooling.ipynb`, `ham-pooling-rerun.ipynb` | Pools real HAM10000 melanoma images into training |

Where a `-rerun` notebook exists it contains corrections noted in its header (dated 2026-09-26); use it in preference to the original, which is kept for history.

Some figures (for example the confusion matrices and the convergence curves) were produced from saved experiment outputs, and their plotting code is not included in the notebooks above.

## Setup, data, and configuration

```bash
git clone https://github.com/Rumas0/ssl-rebalance_melanoma_detection-ready.git
cd ssl-rebalance_melanoma_detection
pip install -r requirements.txt
```

The experiments were developed for a GPU environment such as Kaggle or Colab with a single NVIDIA T4 (16 GB).

**Data.** Raw images are not included. Experiments use the [ISIC 2019 training set](https://www.isic-archive.com/) (filtered to BKL, MEL, NV) and [HAM10000](https://doi.org/10.1038/sdata.2018.161) (Tschandl et al., 2018) for external validation only. The v2 notebooks read the split files `expA_train.csv`, `expA_val.csv`, and `expA_test.csv`. See [`data/README.md`](data/README.md) for where to place everything.

**Configuration.** Notebooks use `config.py` instead of hard-coded machine paths. Set these environment variables for your machine:

| Variable | Purpose |
|---|---|
| `SKIN_CSV_DIR` | Directory with the CSV metadata and split files |
| `SKIN_IMAGE_DIR` | Extracted ISIC image directory |
| `SKIN_ISIC_ARCHIVE_ZIP` | Optional; used only by notebooks that extract from an ISIC ZIP |
| `SKIN_OUTPUT_ROOT` | Writable output and checkpoint directory (on Kaggle, normally under `/kaggle/working/`) |
| `HAM_IMAGE_DIR_PART1`, `HAM_IMAGE_DIR_PART2`, `HAM_META_PATH` | HAM10000 images and metadata (external evaluation) |
| `RESNET18_CHECKPOINT_DIR` | ResNet-18 checkpoints for external evaluation |

```python
import os
os.environ["SKIN_CSV_DIR"] = "/kaggle/input/<your-dataset>/CSVs"
os.environ["SKIN_IMAGE_DIR"] = "/kaggle/input/<your-dataset>/ISIC_2019_Training_Input"
os.environ["SKIN_OUTPUT_ROOT"] = "/kaggle/working/Thesis_Outputs"
```

Do not commit private paths, credentials, tokens, or raw datasets.

## Running the experiments

1. Run `SSL_pre-train_24k_real.ipynb` to produce the Stage-1 encoder checkpoint.
2. Run `ssl-finetune-3seed-macrof1.ipynb` for the main SSL + rebalancing result. The baseline, ResNet-18, and ablation notebooks in `v1_notebooks/` can run independently of it.
3. Run the `v2_notebooks/` experiments. They reuse the Stage-1 checkpoint and the fixed 483 / 104 / 104 split, and each states its prerequisites in its first cell. For example, generate synthetic images with `gan-based-synthetic-mel-generation.ipynb` before the GAN fine-tuning notebook, and export probabilities before the bootstrap and precision–recall notebooks.

Because datasets and checkpoints are not shipped, end-to-end reproduction requires attaching the data and setting the paths above.

## Limitations

- The internal test set has only **eight melanoma images**. Single-seed melanoma recall is unstable, confidence intervals are wide, and neither the bootstrap nor the 15-run cross-validation can raise the test set's resolution.
- External validation is limited to HAM10000, and melanoma recall dropped sharply there.
- The encoder is a compact CNN, not a state-of-the-art architecture, by design.
- Only one generative family (DCGAN with DiffAugment) was tested. Diffusion models were not, and are a natural next step starting from a pre-trained prior.
- The adaptive rule uses validation melanoma recall, which can itself be noisy on a small validation set.
- This is a research artifact and **not a clinical diagnostic tool**.

## Citation and license

The manuscript is not yet published. Citation metadata are in [`CITATION.cff`](CITATION.cff); please cite the manuscript once its bibliographic record is finalized. See `LICENSE` for the code license.
