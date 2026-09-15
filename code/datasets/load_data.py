"""Download raw data from Kaggle and place into data/raw/.

Run from the repository root:  python code/datasets/load_data.py
"""

import shutil
from pathlib import Path
import kagglehub

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download latest version
    path = Path(kagglehub.dataset_download("yasserh/walmart-dataset"))
    print("Path to dataset files:", path)

    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    for src in csv_files:
        dst = RAW_DIR / src.name
        shutil.copy2(src, dst)
        print(f"Saved {src.name} to {dst.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
