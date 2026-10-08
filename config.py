# config.py
import os
from pathlib import Path

# ============================================================
# Dataset configuration
# ============================================================

DATA_ROOT = Path(os.environ.get("SKIN_DATA_ROOT", "./data"))
CSV_DIR = Path(os.environ.get("SKIN_CSV_DIR", DATA_ROOT / "CSVs-20260711T054424Z-2-001/CSVs"))
IMAGE_DIR = Path(os.environ.get("SKIN_IMAGE_DIR", DATA_ROOT / "ISIC_2019_Training_Input"))
ISIC_ARCHIVE_ZIP = os.environ.get("SKIN_ISIC_ARCHIVE_ZIP", "")

# HAM10000 is used only for external validation.
HAM_IMAGE_DIR_PART1 = os.environ.get("HAM_IMAGE_DIR_PART1", "")
HAM_IMAGE_DIR_PART2 = os.environ.get("HAM_IMAGE_DIR_PART2", "")
HAM_META_PATH = os.environ.get("HAM_META_PATH", "")

# ============================================================
# Output / checkpoint configuration
# ============================================================

OUTPUT_ROOT = Path(os.environ.get("SKIN_OUTPUT_ROOT", "./outputs"))
LABELED_DIR = OUTPUT_ROOT / "labeled_real"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
RESULTS_DIR = OUTPUT_ROOT / "results"
RESNET18_CHECKPOINT_DIR = Path(os.environ.get("RESNET18_CHECKPOINT_DIR", CHECKPOINT_DIR))

for directory in [OUTPUT_ROOT, LABELED_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def require_existing_path(path_value, label):
    """Return a Path and fail with a clear message if it is unset/missing."""
    if not path_value:
        raise ValueError(f"{label} is not configured. Set the appropriate environment variable.")
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


if __name__ == "__main__":
    print("Configuration")
    for name in [
        "DATA_ROOT", "CSV_DIR", "IMAGE_DIR", "ISIC_ARCHIVE_ZIP",
        "OUTPUT_ROOT", "LABELED_DIR", "CHECKPOINT_DIR",
        "RESULTS_DIR", "RESNET18_CHECKPOINT_DIR",
        "HAM_IMAGE_DIR_PART1", "HAM_IMAGE_DIR_PART2", "HAM_META_PATH"
    ]:
        print(f"{name}: {globals()[name]}")
