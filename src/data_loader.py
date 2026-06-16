"""Download the Telco Customer Churn dataset via kagglehub and stage it into data/raw/."""

import shutil
from pathlib import Path

import kagglehub

DATASET_SLUG = "blastchar/telco-customer-churn"
EXPECTED_CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset() -> Path:
    """Download (or reuse cached) dataset via kagglehub, return path to its local cache dir."""
    cache_path = kagglehub.dataset_download(DATASET_SLUG)
    return Path(cache_path)


def stage_into_raw(cache_dir: Path, dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Copy the expected CSV from kagglehub's cache into data/raw/, idempotently."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / EXPECTED_CSV_NAME

    if dest_file.exists():
        print(f"[data_loader] {dest_file} already exists, skipping copy.")
        return dest_file

    candidates = list(cache_dir.rglob(EXPECTED_CSV_NAME))
    if not candidates:
        # Dataset publishers sometimes rename files; fall back to the only CSV present.
        csvs = list(cache_dir.rglob("*.csv"))
        if len(csvs) == 1:
            candidates = csvs
        else:
            raise FileNotFoundError(
                f"Could not locate '{EXPECTED_CSV_NAME}' under {cache_dir}. "
                f"Found CSVs: {csvs}"
            )

    shutil.copy2(candidates[0], dest_file)
    print(f"[data_loader] Copied {candidates[0]} -> {dest_file}")
    return dest_file


def ensure_raw_data() -> Path:
    """Idempotent entry point: ensure data/raw/<csv> exists, downloading if necessary."""
    dest_file = RAW_DATA_DIR / EXPECTED_CSV_NAME
    if dest_file.exists():
        print(f"[data_loader] Found existing {dest_file}, skipping download.")
        return dest_file

    print(f"[data_loader] Downloading '{DATASET_SLUG}' via kagglehub...")
    cache_dir = download_dataset()
    return stage_into_raw(cache_dir)


if __name__ == "__main__":
    path = ensure_raw_data()
    print(f"[data_loader] Raw data ready at: {path}")
