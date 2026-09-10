# config.py
import os
from pathlib import Path

# ============================================================
# Dataset configuration
# ============================================================

DATA_ROOT = Path(
    os.environ.get("SKIN_DATA_ROOT", "./data")
)

CSV_DIR = Path(
    os.environ.get("SKIN_CSV_DIR", DATA_ROOT / "CSVs")
)

IMAGE_DIR = Path(
    os.environ.get(
        "SKIN_IMAGE_DIR",
        DATA_ROOT / "ISIC_2019_Training_Input"
    )
)

# HAM10000 paths are configured separately because it is
# used only for external validation.
HAM_IMAGE_DIR_PART1 = os.environ.get(
    "HAM_IMAGE_DIR_PART1",
    ""
)

HAM_IMAGE_DIR_PART2 = os.environ.get(
    "HAM_IMAGE_DIR_PART2",
    ""
)

HAM_META_PATH = os.environ.get(
    "HAM_META_PATH",
    ""
)

# ============================================================
# Output configuration
# ============================================================

OUTPUT_ROOT = Path(
    os.environ.get("SKIN_OUTPUT_ROOT", "./outputs")
)

LABELED_DIR = OUTPUT_ROOT / "labeled_real"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RESULTS_DIR = OUTPUT_ROOT / "results"

# Create output directories
for directory in [
    OUTPUT_ROOT,
    LABELED_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Configuration")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"CSV_DIR: {CSV_DIR}")
    print(f"IMAGE_DIR: {IMAGE_DIR}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"LABELED_DIR: {LABELED_DIR}")
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")