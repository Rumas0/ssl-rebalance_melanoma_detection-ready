# Self-Supervised Pretraining and Imbalance-Aware Rebalancing for Melanoma Detection

Code and experiment scripts for a two-stage framework that combines SimCLR self-supervised pretraining with class-rebalancing (focal loss, targeted augmentation, oversampling) to improve melanoma sensitivity on small, severely imbalanced dermoscopic datasets. This repository accompanies the paper *"Self-Supervised Pretraining and Imbalance-Aware Rebalancing for Melanoma Detection in Small, Imbalanced Dermoscopic Datasets"* (I.S. Tunib, Md. Sazzadur Ahamed, M.F. Bin), itself derived from an FYDP thesis at Daffodil International University.

## Why this exists

Standard supervised training on imbalanced dermoscopic data can reach ~73% accuracy while detecting **zero** melanomas — a majority-class collapse that aggregate accuracy hides. This project isolates two candidate fixes:

- **Self-supervised pretraining** (SimCLR) — learns dermoscopy-specific features from 24,000 unlabeled ISIC images, without needing labels.
- **Imbalance-aware rebalancing** — class-weighted focal loss, melanoma-specific augmentation, and 5× oversampling during fine-tuning on 483 labeled images.

A controlled 2×2 ablation plus component-level ablations show the two techniques address different failure modes and are most effective combined.

## Key results (3-seed mean ± SD)

| Condition | Accuracy (%) | Macro-F1 | MEL Recall (%) |
|---|---|---|---|
| Baseline (no SSL, no rebalancing) | 73.1 ± 0.8 | — | 0.0 ± 0.0 |
| All rebalancing (no SSL) | 66.0 ± 2.8 | 0.524 ± 0.016 | 33.3 ± 5.9 |
| **SSL + rebalancing (full framework)** | **64.7 ± 3.2** | **0.510 ± 0.021** | **41.7 ± 5.9** |

Full breakdowns (component ablations, ResNet-18 comparison, HAM10000 external validation) are in the paper.

## Repository structure

```
.
├── notebooks/
│   ├── 01_SSL_pre-train_24k_real.ipynb              # Stage 1 — SimCLR on 24,000 real unlabeled ISIC images
│   ├── 02_Baseline_CNN_with_real_image.ipynb        # Table 1 — conventional baseline, 3 seeds
│   ├── 03_ssl-only-no-rebalancing-macro.ipynb        # Table 2 — SSL-only ablation (single seed)
│   ├── 04_rebalancing_only-3seeds-various_technique.ipynb  # Table 2 — all no-SSL rebalancing components, 3 seeds
│   ├── 05_ssl-finetune-3seed-macrof1.ipynb           # Table 3 + most of Table 2 — full SSL+rebalance framework, 3 seeds
│   ├── 06_resnet18-baseline-3seeds.ipynb             # Table 2 / §4.4 — ResNet-18 comparison, 3 seeds
│   └── 07_ham10000-external-evaluation.ipynb         # Table 4 — external validation on HAM10000
├── data/                 # Dataset split manifests (expA_train/val/test.csv), unlabeled_pool_24k.csv
├── results/              # Per-seed metrics (JSON) and test-set prediction probabilities (CSV)
├── figures/              # Scripts that regenerate every figure in the paper from results/*.json,*.csv
├── paper/                # LaTeX source for the manuscript
├── requirements.txt
└── README.md
```

The numeric prefixes reflect pipeline order — Stage 1 pretraining, then baseline, then the ablations, then the full framework, then the ResNet-18 comparison and external validation. Rename to match your own convention if you prefer.

## Setup

```bash
git clone https://github.com/Rumas0/ssl-rebalance_melanoma_detection.git
cd ssl-rebalance_melanoma_detection
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch, torchvision, on a single NVIDIA T4 GPU (16 GB, Colab/Kaggle). SSL pretraining takes ~5 hours; each 3-seed fine-tuning notebook takes well under an hour.

## Usage

Run the notebooks in order. Each writes its checkpoint(s) and results file to a shared output directory that later notebooks read from:

1. **`01_SSL_pre-train_24k_real.ipynb`** — pretrains the 3-block CNN encoder with SimCLR on the unlabeled pool. Produces `ssl_encoder_real_24000.pth`.
2. **`02_Baseline_CNN_with_real_image.ipynb`** — trains the same encoder from random initialization with plain cross-entropy, 3 seeds. No dependency on step 1.
3. **`03_ssl-only-no-rebalancing-macro.ipynb`** — loads the Stage-1 encoder, fine-tunes with plain cross-entropy (no rebalancing), single seed.
4. **`04_rebalancing_only-3seeds-various_technique.ipynb`** — random-initialized encoder, sweeps focal loss / class weights / augmentation / oversampling individually and combined, 3 seeds each.
5. **`05_ssl-finetune-3seed-macrof1.ipynb`** — loads the Stage-1 encoder and applies the full rebalancing package, 3 seeds. Produces `ssl_rebalance_3seed_results.json` and `test_probs_seed{42,123,2024}.csv`.
6. **`06_resnet18-baseline-3seeds.ipynb`** — same rebalancing recipe on an ImageNet-pretrained ResNet-18, 3 seeds. Produces `resnet18_seed{seed}_finetuned.pth`.
7. **`07_ham10000-external-evaluation.ipynb`** — loads a checkpoint from steps 5 and 6, evaluates both on HAM10000.

```bash
# Regenerate all paper figures from results/*.json and results/*.csv
python figures/make_all_figures.py
```

## Data

Experiments use dermoscopic images from the [ISIC archive](https://www.isic-archive.com/), used under its terms of use. Raw images are **not** included in this repository; `data/` contains only split manifests and derived metrics. See the paper's Data Availability statement for exact split identifiers.

## Limitations

The evaluation uses a single archive (ISIC 2019), an 8-melanoma test set, and no prospective validation — see the paper's Limitations section before drawing clinical conclusions. This is a research artifact, not a diagnostic tool.

## Citation

If you use this code, please cite the accompanying paper (citation details to be added once published/preprinted).

## License

License: see `LICENSE`.

## Acknowledgments

International Skin Imaging Collaboration (ISIC) for the dermoscopic image archive; Daffodil International University for academic and computational support.


## Configuration

The notebooks use `config.py` for dataset and output paths. By default they expect a local layout under `data/` and write generated files under `outputs/`.

For Kaggle, Colab, or another machine, set the corresponding environment variables rather than editing notebook cells. In particular:

- `SKIN_DATA_ROOT`
- `SKIN_CSV_DIR`
- `SKIN_IMAGE_DIR`
- `SKIN_ISIC_ARCHIVE_ZIP`
- `SKIN_OUTPUT_ROOT`
- `HAM_IMAGE_DIR_PART1`
- `HAM_IMAGE_DIR_PART2`
- `HAM_META_PATH`
- `RESNET18_CHECKPOINT_DIR`

Raw datasets and model checkpoints are intentionally excluded from version control.
