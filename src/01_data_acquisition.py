"""
ClimateScope — Step 1: Data Acquisition
Downloads the Global Weather Repository dataset from Kaggle.
Run: python src/01_data_acquisition.py
"""

import os
import zipfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY")

DATASET_SLUG = "nelgiriyewithana/global-weather-repository"
RAW_DIR      = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def check_credentials() -> bool:
    """Verify Kaggle credentials are available."""
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        print("❌  KAGGLE_USERNAME or KAGGLE_KEY not set.")
        print("    Copy .env.example → .env and fill in your credentials.")
        print("    Get them at: https://www.kaggle.com/settings → API → Create New Token")
        return False
    # Kaggle library reads from env vars automatically
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"]      = KAGGLE_KEY
    return True


def download_dataset() -> Path:
    """Download and extract the dataset, return the CSV path."""
    import kaggle  # imported here so the env vars are already set

    zip_path = RAW_DIR / "global-weather-repository.zip"

    print(f"⬇️   Downloading: {DATASET_SLUG}")
    kaggle.api.dataset_download_files(
        DATASET_SLUG,
        path=str(RAW_DIR),
        unzip=False,
        quiet=False,
    )

    # Locate the downloaded zip (kaggle names it after the dataset)
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("No zip file found after download.")
    zip_path = zips[0]

    print(f"📦  Extracting: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

    zip_path.unlink()  # delete the zip to save space

    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV found after extraction.")

    csv_path = csv_files[0]
    print(f"✅  Dataset ready: {csv_path}")
    return csv_path


def verify_dataset(csv_path: Path) -> None:
    """Quick sanity check on the downloaded file."""
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=5)
    print(f"\n📋  Shape preview  : {pd.read_csv(csv_path).shape}")
    print(f"📋  Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    if check_credentials():
        csv_path = download_dataset()
        verify_dataset(csv_path)
