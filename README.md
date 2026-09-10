# Self-Supervised Pretraining and Imbalance-Aware Rebalancing for Melanoma Detection

Research code and experiment notebooks accompanying the manuscript *Self-Supervised Pretraining and Imbalance-Aware Rebalancing for Melanoma Detection in Small, Imbalanced Dermoscopic Datasets*. The project investigates whether domain-specific self-supervised representation learning and imbalance-aware optimization provide complementary benefits when labeled dermoscopic data are scarce and severely imbalanced.

> **Status:** Research manuscript submitted for peer review. This repository contains the implementation, experiment notebooks, selected figures, and configuration scaffolding used for the study.

## Research question

The study separates two failure mechanisms: representation learning under limited labels and minority-class collapse during supervised optimization. Stage 1 uses SimCLR pretraining on an unlabeled ISIC image pool; Stage 2 fine-tunes on a small three-class labeled split using weighted focal loss, melanoma-specific augmentation, and oversampling. Controlled ablations compare supervised baseline, SSL-only, rebalancing-only, and the combined framework.

## Key reported results

| Condition | Accuracy (%) | Macro-F1 | MEL Recall (%) |
|---|---:|---:|---:|
| Baseline (no SSL, no rebalancing) | 73.1 ± 0.8 | — | 0.0 ± 0.0 |
| All rebalancing (no SSL) | 66.0 ± 2.8 | 0.524 ± 0.016 | 33.3 ± 5.9 |
| **SSL + rebalancing (full framework)** | **64.7 ± 3.2** | **0.510 ± 0.021** | **41.7 ± 5.9** |

The manuscript also reports a matched ResNet-18 comparison and external validation on HAM10000. These results should be interpreted in light of the small eight-melanoma internal test set and the observed external domain shift.

## Repository structure

```text
.
├── README.md
├── config.py
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── notebooks/
│   ├── Baseline_CNN_with_real_image.ipynb
│   ├── SSL_pre-train_24k_real.ipynb
│   ├── ham10000-external-evaluation.ipynb
│   ├── rebalancing-only-no-ssl-macro.ipynb
│   ├── rebalancing_only-3seeds-various_technique.ipynb
│   ├── resnet18-baseline-3seeds.ipynb
│   ├── ssl-finetune-3seed-macrof1.ipynb
│   └── ssl-only-no-rebalancing-macro.ipynb
├── data/
│   └── README.md
├── results/
│   └── README.md
├── figures/
│   ├── README.md
│   └── selected publication figures
└── paper/
    └── README.md
```

The raw ISIC and HAM10000 images are **not** included. The current public repository also does not claim to contain the full manuscript source or complete generated-results archive; those can be added in a future release if appropriate.

## Setup

```bash
git clone https://github.com/Rumas0/ssl-rebalance_melanoma_detection.git
cd ssl-rebalance_melanoma_detection
pip install -r requirements.txt
```

The experiments were developed for a GPU environment such as Kaggle/Colab using a single NVIDIA T4 (16 GB). Exact runtime depends on the attached dataset and environment.

## Configuration

The notebooks use `config.py` instead of embedding personal machine paths. Set environment variables for the machine on which you run the experiments. For Kaggle, configure them near the start of a notebook or through the notebook environment.

Required variables for the ISIC workflow:

- `SKIN_CSV_DIR` — directory containing the CSV metadata/split files.
- `SKIN_IMAGE_DIR` — extracted ISIC image directory.
- `SKIN_ISIC_ARCHIVE_ZIP` — optional archive path used only by the two notebooks that extract from an ISIC ZIP.
- `SKIN_OUTPUT_ROOT` — writable output/checkpoint directory; on Kaggle this should normally be under `/kaggle/working/`.

HAM10000 external evaluation additionally uses:

- `HAM_IMAGE_DIR_PART1`
- `HAM_IMAGE_DIR_PART2`
- `HAM_META_PATH`
- `RESNET18_CHECKPOINT_DIR`

Example pattern (replace with the paths created by your own Kaggle dataset attachment):

```python
import os
os.environ["SKIN_CSV_DIR"] = "/kaggle/input/<your-dataset>/CSVs"
os.environ["SKIN_IMAGE_DIR"] = "/kaggle/input/<your-dataset>/ISIC_2019_Training_Input"
os.environ["SKIN_OUTPUT_ROOT"] = "/kaggle/working/Thesis_Outputs"
```

Do not commit private Kaggle paths, credentials, tokens, or raw datasets.

## Running the notebooks

The notebooks represent the experimental workflow used in the manuscript. Start with the SSL pretraining notebook when reproducing the complete pipeline; the baseline notebook is independent of the SSL stage. The remaining notebooks perform the ablations, full framework evaluation, ResNet-18 comparison, and HAM10000 external evaluation.

Some notebooks expect checkpoints/results produced by earlier experiments. Because the public repository currently does not ship the raw datasets or generated checkpoints, full end-to-end reproduction requires attaching the corresponding datasets and configuring the paths above.

## Data

Experiments use dermoscopic images from the [ISIC archive](https://www.isic-archive.com/) and HAM10000 for external validation. Raw images are excluded from this repository. See the manuscript for the documented split design, limitations, and data-availability information.

## Limitations

The study uses a small labeled training set and only eight melanoma cases in the internal test set. It also observes a substantial drop in melanoma recall on HAM10000, highlighting domain-shift limitations. The model is a research artifact and **not a clinical diagnostic tool**.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the manuscript once its bibliographic record is finalized.

## License

See `LICENSE`.
