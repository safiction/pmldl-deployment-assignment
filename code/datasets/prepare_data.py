"""Stage 1: data engineering

Input:  data/raw/Walmart.csv
Output: data/processed/train.csv, data/processed/test.csv

This stage only produces clean, split datasets.

Run from the repository root:  python code/datasets/prepare_data.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "Walmart.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TARGET = "Weekly_Sales"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.relative_to(REPO_ROOT)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Typecast, handle missing values, and drop extreme outliers."""
    df = df.copy()

    # Typecasting date column
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    # Drop missing values
    before = len(df)
    df = df.dropna()
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with missing values")

    # Outliers: conservative global IQR filter on the target. EDA showed only
    # 0.5% of rows are outliers and most are legitimate large-store / holiday
    # peaks, so used the standard 1.5*IQR fence.
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    before = len(df)
    df = df[(df[TARGET] >= low) & (df[TARGET] <= high)]
    print(f"Removed {before - len(df)} outlier rows on {TARGET} "
          f"({100 * (before - len(df)) / before:.1f}%)")

    return df.reset_index(drop=True)


def split_and_save(df: pd.DataFrame) -> None:
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # For stricter evaluation better to split chronologically (because of time-series data),
    # here did just random split

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train: {len(train_df)} rows in {train_path.relative_to(REPO_ROOT)}")
    print(f"Saved test:  {len(test_df)} rows in {test_path.relative_to(REPO_ROOT)}")


def main() -> None:
    df = load_data(RAW_PATH)
    df = clean_data(df)
    split_and_save(df)


if __name__ == "__main__":
    main()
