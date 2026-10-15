# Data

Raw datasets are **not** included in this repository. Obtain them from the original sources and respect their licences.

## ISIC 2019 (training and internal evaluation)

- Source: [ISIC Archive](https://www.isic-archive.com/) (ISIC 2019 challenge training set, 25,331 dermoscopic images)
- Used as: a 24,000-image unlabeled pool for SimCLR pre-training, and a labeled BKL / MEL / NV set (691 images) for fine-tuning
- Split files read by the v2 notebooks: `expA_train.csv` (483), `expA_val.csv` (104), `expA_test.csv` (104)
- Point `SKIN_CSV_DIR` at the folder with the CSV files and `SKIN_IMAGE_DIR` at the extracted images

## HAM10000 (external validation only)

- Source: Tschandl et al., "The HAM10000 dataset", *Scientific Data*, 2018 (https://doi.org/10.1038/sdata.2018.161); also mirrored on Kaggle
- Used as: external validation, filtered to BKL, MEL, NV (8,917 images); for the supplement experiment it is split by `lesion_id` into a pooled-in-training set and a disjoint holdout
- Set `HAM_IMAGE_DIR_PART1`, `HAM_IMAGE_DIR_PART2`, and `HAM_META_PATH` (the `HAM10000_metadata.csv` file)

Never commit raw images or checkpoints (see `.gitignore`).
