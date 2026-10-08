# SSL-Rebalance: Melanoma Detection under Severe Class Imbalance

Research code and experiment notebooks accompanying the manuscript:

> **Shifting vs. Lifting Melanoma Recall under Severe Class Imbalance: An Empirical Study of Loss-Level, Representation-Level, and Generative Interventions**
> I. S. Tunib, Md. Sazzadur Ahamed, M. F. Bin

> **Status:** manuscript in preparation. This repository contains the implementation, experiment notebooks, selected figures, and configuration used for the study. Raw datasets and trained checkpoints are not included.

## Overview

Melanoma is rare in dermoscopic datasets, so a classifier can reach high accuracy while missing almost every melanoma. This project studies a compact, reproducible three-class pipeline (BKL, MEL, NV) on ISIC 2019 and asks what actually improves melanoma recall when labeled data are scarce and severely imbalanced (691 labeled images, 35 of them melanoma).

The central claim of the study: melanoma recall can be **shifted** by loss-level changes (adaptive class weighting) but not **lifted** without a representation-level intervention (supervised contrastive loss with squeeze-and-excitation attention). Neither transfers externally without adding real external data, and synthetic GAN augmentation is not a substitute for real data at this scale.

**Pipeline**

1. **Stage 1: SimCLR pre-training** of a compact three-block CNN encoder (about 0.1M parameters) on 24,000 unlabeled ISIC images.
2. **Stage 2: supervised fine-tuning** on the small labeled split with melanoma 5x oversampling, heavy melanoma-specific augmentation, and focal loss with class weights.
3. **Interventions studied on top of Stage 2:**
   - a bounded adaptive melanoma class-weight rule (loss level)
   - a supervised contrastive loss with squeeze-and-excitation attention (representation level)
   - DCGAN-based synthetic melanoma augmentation (generative level)
4. **Validation:** three random seeds (42, 123, 2024), bootstrap confidence intervals, 5-fold cross-validation, and external evaluation on HAM10000.

## Main results

Internal ISIC 2019 test set (104 images, 8 melanoma). Mean ± standard deviation over three seeds.

| Model | Accuracy (%) | Macro-F1 | MEL recall (%) |
|---|---:|---:|---:|
| Simple CNN, real images (no SSL, no rebalancing) | 73.1 ± 0.8 | n/a | 0.0 ± 0.0 |
| ResNet-18 + heavy augmentation | 78.5 ± 2.5 | 0.591 ± 0.013 | 33.3 ± 5.9 |
| **SSL + rebalancing (fixed weights)** | 64.7 ± 3.2 | 0.510 ± 0.021 | 41.7 ± 5.9 |
| SSL + adaptive weighting | 60.6 ± 1.4 | 0.483 ± 0.014 | 45.8 ± 5.9 |
| SSL + GAN-synthetic MEL | 62.2 ± 4.8 | 0.452 ± 0.023 | 16.7 ± 11.8 |
| SupCon + SE attention | 67.3 ± 2.5 | 0.512 ± 0.009 | 45.8 ± 14.4 |

**Key findings**

- **Adaptive weighting shifts, it does not lift.** It raises melanoma recall from 41.7% to 45.8% but lowers macro-F1, and the precision–recall analysis places its operating points on or below the fixed-weight curve (melanoma Average Precision 0.2085 ± 0.0132). Bootstrap confidence intervals overlap substantially, so the gain is not statistically distinguishable at this sample size.
- **A representation-level change does lift the ranking.** SupCon + SE attention raises melanoma Average Precision from 0.2085 to 0.261 ± 0.025 with melanoma precision essentially unchanged (20.1% vs. 19.9% ± 2.8%).
- **GAN augmentation hurt.** Synthetic images were closer to individual real melanoma images than real images are to each other (cosine 0.9565 vs. 0.8003) while the Fréchet Encoder Distance was about four times larger (8.63 vs. 2.13), which indicates mode collapse.
- **Internal recall did not transfer.** On HAM10000, melanoma recall fell to 6.6% for the SSL + rebalancing model, versus 29.3% for ResNet-18. Adding a small real supplement (41 HAM10000 melanoma images) raised external recall to 29.9% ± 9.6%; pooling many more over-corrected the imbalance.

These numbers come from a test set with only eight melanoma images, where a single prediction changes recall by 12.5 points. See [Limitations](#limitations).

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
├── figures/                  # selected publication figures
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
| `gan-based-synthetic-mel-generation.ipynb` | Trains a DCGAN on the 35 real melanoma images and evaluates the synthetic output |
| `addendum-real-vs-real-baseline-for-the-gan-simila.ipynb` | Real-vs-real reference for the similarity metrics |
| `ssl-rebalance-fine-tuning-with-gan-synthetic-mel.ipynb` | Fine-tuning with GAN-synthetic melanoma augmentation |
| `ham-pooling.ipynb`, `ham-pooling-rerun.ipynb` | Pools real HAM10000 melanoma images into training |

Where a `-rerun` notebook exists, it contains corrections noted in its header (dated 2026-09-26); use it in preference to the original, which is kept for history.

## Setup

```bash
git clone https://github.com/Rumas0/ssl-rebalance_melanoma_detection.git
cd ssl-rebalance_melanoma_detection
pip install -r requirements.txt
```

The experiments were developed for a GPU environment such as Kaggle or Colab with a single NVIDIA T4 (16 GB). Runtime depends on the attached dataset and environment.

## Data

Raw images are **not** included. Experiments use:

- **ISIC 2019 training set** (25,331 dermoscopic images, filtered to BKL, MEL, NV for classification) from the [ISIC archive](https://www.isic-archive.com/). 24,000 images form the unlabeled SimCLR pool; 691 labeled images are used for fine-tuning, split 483 / 104 / 104 into train / validation / test.
- **HAM10000** (Tschandl et al., 2018), filtered to BKL, MEL, NV, for external validation only.

| Split | BKL | MEL | NV |
|---|---:|---:|---:|
| ISIC train (n = 483) | 99 | 35 | 349 |
| ISIC validation (n = 104) | 21 | 7 | 76 |
| ISIC test (n = 104) | 21 | 8 | 75 |
| HAM10000 external (n = 8,917) | 1,099 | 1,113 | 6,705 |

The v2 notebooks read the split files `expA_train.csv`, `expA_val.csv`, and `expA_test.csv`. See `data/README.md` for where to place the data.

## Configuration

Notebooks use `config.py` instead of hard-coded machine paths. Set environment variables for your environment.

Required for the ISIC workflow:

- `SKIN_CSV_DIR`: directory containing the CSV metadata and split files
- `SKIN_IMAGE_DIR`: extracted ISIC image directory
- `SKIN_ISIC_ARCHIVE_ZIP`: optional; used only by notebooks that extract from an ISIC ZIP
- `SKIN_OUTPUT_ROOT`: writable output and checkpoint directory (on Kaggle, normally under `/kaggle/working/`)

Additionally for HAM10000 evaluation:

- `HAM_IMAGE_DIR_PART1`, `HAM_IMAGE_DIR_PART2`, `HAM_META_PATH`, `RESNET18_CHECKPOINT_DIR`

```python
import os
os.environ["SKIN_CSV_DIR"] = "/kaggle/input/<your-dataset>/CSVs"
os.environ["SKIN_IMAGE_DIR"] = "/kaggle/input/<your-dataset>/ISIC_2019_Training_Input"
os.environ["SKIN_OUTPUT_ROOT"] = "/kaggle/working/Thesis_Outputs"
```

Do not commit private paths, credentials, tokens, or raw datasets.

## Running the experiments

1. Run `SSL_pre-train_24k_real.ipynb` to produce the Stage-1 encoder checkpoint.
2. Run `ssl-finetune-3seed-macrof1.ipynb` for the main SSL + rebalancing result. The baseline, ResNet-18, and ablation notebooks in `v1_notebooks/` can be run independently of it.
3. Run the `v2_notebooks/` experiments. They reuse the Stage-1 checkpoint and the fixed 483 / 104 / 104 split. Each notebook states its prerequisites in its first cell; for example, generate synthetic images with `gan-based-synthetic-mel-generation.ipynb` before running the GAN fine-tuning notebook, and export probabilities before the bootstrap and precision–recall notebooks.

Because datasets and checkpoints are not shipped, end-to-end reproduction requires attaching the data and setting the paths above.

## Limitations

- The internal test set contains only **eight melanoma images**, so single-seed melanoma recall is unstable and confidence intervals are wide. Bootstrap and 15-run cross-validation estimates corroborate the point estimates but cannot increase the test set's resolution.
- External validation is limited to HAM10000, and melanoma recall dropped sharply there.
- The encoder is a compact CNN, not a state-of-the-art architecture, by design.
- Only one generative family (DCGAN with DiffAugment) was tested; diffusion models were not.
- This is a research artifact and **not a clinical diagnostic tool**.

## Citation

The manuscript has not yet been published. Citation metadata are in `CITATION.cff`; please cite the manuscript once its bibliographic record is finalized.

## License

See `LICENSE`.
