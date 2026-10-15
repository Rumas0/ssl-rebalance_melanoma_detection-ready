# Results

Summary of the results reported in the manuscript *Shifting vs. Lifting Melanoma Recall under Severe Class Imbalance*, with the notebook that produces each one. Numbers are copied from the manuscript's tables (table numbers are given so each value can be checked).

**How to read these numbers.** The internal ISIC 2019 test set has only **8 melanoma (MEL) images**, so one changed prediction moves MEL recall by 12.5 percentage points and recall can take only nine distinct values. Treat differences of a few points as descriptive, not as proof of improvement. Unless stated otherwise, values are mean ± standard deviation over seeds 42, 123, and 2024.

A machine-readable copy of the main table is in [`main_results.csv`](main_results.csv).

## Contents

- [Main internal results](#main-internal-results)
- [Component ablation](#component-ablation)
- [Adaptive weighting: shift, not lift](#adaptive-weighting-shift-not-lift)
- [SupCon + SE attention](#supcon--se-attention)
- [Statistical robustness](#statistical-robustness)
- [GAN augmentation diagnostics](#gan-augmentation-diagnostics)
- [External validation on HAM10000](#external-validation-on-ham10000)
- [Where each result is produced](#where-each-result-is-produced)
- [What belongs in this folder](#what-belongs-in-this-folder)

## Main internal results

ISIC 2019 internal test set (104 images: 21 BKL, 8 MEL, 75 NV). Manuscript Table II.

| Model | Accuracy | Macro-F1 | MEL recall | Params | Notes |
|---|---:|---:|---:|---:|---|
| Simple CNN, real images | 0.731 ± 0.008 | n/a | 0.000 ± 0.000 | 0.1M | 3 seeds |
| ResNet-18 + heavy augmentation | 0.785 ± 0.025 | 0.591 ± 0.013 | 0.333 ± 0.059 | 11.2M | 3 seeds |
| SSL only, no rebalancing | 0.740 | 0.392 | 0.000 | 0.1M | single seed |
| Rebalancing only, no SSL | 0.673 | 0.536 | 0.375 | 0.1M | single seed |
| **SSL + rebalancing (fixed weights)** | 0.647 ± 0.032 | 0.510 ± 0.021 | 0.417 ± 0.059 | 0.1M | 3 seeds |
| SSL + adaptive weighting | 0.606 ± 0.014 | 0.483 ± 0.014 | 0.458 ± 0.059 | 0.1M | 3 seeds |
| SSL + GAN-synthetic MEL | 0.622 ± 0.048 | 0.452 ± 0.023 | 0.167 ± 0.118 | 0.1M | 3 seeds |
| SupCon + SE attention | 0.673 ± 0.025 | 0.512 ± 0.009 | 0.458 ± 0.144 | 0.1M | 3 seeds |

The ResNet-18 baseline has the best accuracy and macro-F1 but 110 times more parameters. The compact SSL-family models are compared with each other, not presented as beating ResNet-18 overall.

## Component ablation

Rebalancing components over three seeds (manuscript Table III). Note that the "Baseline" row here comes from this ablation run and differs from the simple-CNN baseline in the main table above; see the manuscript for the setup.

| Condition | Accuracy | Macro-F1 | MEL recall |
|---|---:|---:|---:|
| Baseline | 0.628 ± 0.125 | 0.314 ± 0.033 | 20.8% ± 29.5% |
| Focal loss only | 0.718 ± 0.016 | 0.375 ± 0.026 | 0.0% ± 0.0% |
| Class weights only | 0.683 ± 0.016 | 0.421 ± 0.018 | 0.0% ± 0.0% |
| Augmentation only | 0.724 ± 0.012 | 0.331 ± 0.035 | 0.0% ± 0.0% |
| Oversampling only | 0.670 ± 0.016 | 0.358 ± 0.025 | 25.0% ± 10.2% |
| **All rebalancing** | 0.660 ± 0.028 | 0.524 ± 0.016 | 33.3% ± 5.9% |

No single component recovers melanoma on its own; the combination does.

## Adaptive weighting: shift, not lift

Fixed vs. adaptive melanoma operating points per seed (manuscript Table V) and the isolated effect of adaptive weighting (Table IV).

| Seed | Fixed MEL recall | Fixed MEL precision | Adaptive MEL recall | Adaptive MEL precision |
|---|---:|---:|---:|---:|
| 42 | 0.375 | 0.200 | 0.500 | 0.200 |
| 123 | 0.500 | 0.190 | 0.500 | 0.174 |
| 2024 | 0.375 | 0.214 | 0.375 | 0.143 |
| **Mean** | **0.417** | **0.201** | **0.458** | **0.172** |

Fixed-weight melanoma Average Precision: **0.2085 ± 0.0132** (per seed 0.2216 / 0.1904 / 0.2135).

| Seed | Accuracy | Macro-F1 | MEL recall |
|---|---:|---:|---:|
| 42 | 62.5% | 0.501 | 50.0% |
| 123 | 59.6% | 0.482 | 50.0% |
| 2024 | 59.6% | 0.466 | 37.5% |
| Mean ± SD | 60.6% ± 1.4% | 0.483 ± 0.014 | 45.8% ± 5.9% |

Recall rises 0.417 → 0.458 while precision falls 0.201 → 0.172, and all adaptive operating points lie on or slightly below the fixed-weight precision–recall curve. The rule moves the decision boundary along a fixed ranking; it does not improve the ranking.

<p align="center">
  <img src="../figures/fig05_pr_curve_adaptive.png" width="560" alt="Fixed-weight PR curve with adaptive-weight operating points">
</p>

## SupCon + SE attention

The representation-level intervention (manuscript Table VI) raises melanoma Average Precision from 0.2085 to **0.261 ± 0.025**. At the default 0.5 threshold, MEL recall goes from 41.7% to 45.8% while MEL precision is essentially unchanged (20.1% vs. 19.9% ± 2.8%). The per-seed recall spread is wide (± 14.4%) because of the 8-image test set.

<p align="center">
  <img src="../figures/fig08_roc_pr_pooled.png" width="760" alt="Pooled ROC and PR curves for the fixed, adaptive, and SupCon + SE models">
</p>

## Statistical robustness

95% bootstrap intervals (2,000 resamples), pooled over three seeds, n = 312 test predictions per model (manuscript Table VII).

| Model | Metric | Point estimate | 95% CI |
|---|---|---:|---|
| Fixed weight | MEL recall | 0.417 | [0.217, 0.611] |
| Fixed weight | MEL precision | 0.200 | [0.093, 0.319] |
| Adaptive weight | MEL recall | 0.458 | [0.259, 0.652] |
| Adaptive weight | MEL precision | 0.172 | [0.081, 0.271] |

The intervals overlap substantially, so the adaptive-weighting change is reported descriptively and is not claimed to be significant. An independent 5-fold stratified cross-validation estimate (3 seeds x 5 folds = 15 training runs, over the 587 train + validation images, evaluated on the unchanged test set) gives MEL recall of **40.0% ± 5.2%**, within noise of the three-seed result.

## GAN augmentation diagnostics

Feature-space similarity measured with the SSL encoder (manuscript Table IX). Lower Fréchet Encoder Distance (FED) is better; high cosine similarity is only desirable when it is not accompanied by a narrow distribution.

| Comparison | Mean nearest-neighbour cosine | FED |
|---|---:|---:|
| Synthetic vs. real melanoma | 0.9565 | 8.63 |
| Real vs. real melanoma (reference) | 0.8003 | 2.13 |

Synthetic images are closer to individual real images than real images are to each other, yet the FED is about four times larger. That combination indicates a narrow, shifted distribution (mode collapse) and matches the weaker classifier result (MEL recall 16.7% ± 11.8%).

## External validation on HAM10000

Manuscript Tables X and XI. HAM10000 filtered to BKL, MEL, NV (8,917 images).

| Model | Accuracy | Macro-F1 | MEL recall | MEL precision | MEL F1 |
|---|---:|---:|---:|---:|---:|
| SSL + rebalancing | 0.718 | 0.459 | 6.6% | 33.9% | 0.111 |
| ResNet-18 | 0.774 | 0.568 | 29.3% | 34.2% | 0.316 |

Effect of a **modest real supplement** (41 HAM10000 melanoma images from 24 lesions added to training; 3 seeds):

| Split | Accuracy | Macro-F1 | MEL recall |
|---|---:|---:|---:|
| ISIC test | 0.686 ± 0.037 | 0.532 ± 0.057 | 0.583 ± 0.072 |
| HAM10000 holdout | 0.719 ± 0.007 | 0.516 ± 0.010 | 0.299 ± 0.096 |

Pooling the full spare pool (555 images) over-corrected the imbalance and collapsed internal accuracy to 46–60%.

<p align="center">
  <img src="../figures/fig12_ham_supplement.png" width="520" alt="Melanoma recall with and without a modest HAM10000 supplement">
</p>

## Where each result is produced

| Result | Notebook(s) |
|---|---|
| Simple CNN baseline | `v1_notebooks/Baseline_CNN_with_real_image.ipynb` |
| ResNet-18 baseline | `v1_notebooks/resnet18-baseline-3seeds.ipynb` |
| SSL-only ablation | `v1_notebooks/ssl-only-no-rebalancing-macro.ipynb` |
| Rebalancing-only ablation, component ablation | `v1_notebooks/rebalancing-only-no-ssl-macro.ipynb`, `v1_notebooks/rebalancing_only-3seeds-various_technique.ipynb` |
| SSL + rebalancing (fixed weights) | `v1_notebooks/ssl-finetune-3seed-macrof1.ipynb` (after `SSL_pre-train_24k_real.ipynb`) |
| Adaptive weighting | `v2_notebooks/ssl-finetune-adaptive-weights.ipynb`, `adaptive-test-probs.ipynb` |
| Precision–recall analysis, Average Precision | `v2_notebooks/precision-recall-analysis-3seed.ipynb`, `val-probs-ssl-rebalance-checkpoints.ipynb` |
| SupCon + SE attention | `v2_notebooks/supcon-attention.ipynb` |
| Bootstrap intervals | `v2_notebooks/bootstrap-ci.ipynb` (use `bootstrap-ci-rerun.ipynb`, which carries a dated correction) |
| 5-fold cross-validation | `v2_notebooks/kfold-cv.ipynb` |
| GAN generation and diagnostics | `v2_notebooks/gan-based-synthetic-mel-generation.ipynb`, `addendum-real-vs-real-baseline-for-the-gan-simila.ipynb` |
| GAN-augmented fine-tuning | `v2_notebooks/ssl-rebalance-fine-tuning-with-gan-synthetic-mel.ipynb` |
| HAM10000 external evaluation | `v1_notebooks/ham10000-external-evaluation.ipynb` |
| HAM10000 supplement and pooling | `v2_notebooks/ham-pooling.ipynb` (use `ham-pooling-rerun.ipynb`, which carries a dated correction) |

## What belongs in this folder

This folder holds **small, reviewable summaries**: result tables, CSV or JSON metric summaries, and short notes. It is a different location from the notebooks' working output directory (`SKIN_OUTPUT_ROOT/results`, set in `config.py`), which is git-ignored.

Keep out of version control: model checkpoints (`*.pth`, `*.pt`), per-image prediction dumps, raw images, and any large generated artifact (see `.gitignore`). If you add a summary file here, name the notebook and seed that produced it.

This is a research artifact and **not a clinical diagnostic tool**.
