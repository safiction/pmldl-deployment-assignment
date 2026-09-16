"""Stage 2: model engineering

Input:  data/processed/train.csv, data/processed/test.csv
Output: models/model.joblib, models/metrics.json

Feature engineering, training, evaluation, and packaging.

Run from the repository root:  python code/models/train.py
"""
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from features import TARGET, FEATURE_COLUMNS, add_date_features, build_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models"

def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    print(f"Loaded train: {len(train_df)} rows, test: {len(test_df)} rows")
    return train_df, test_df

def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }
    print(f"Test MAE:  {metrics['mae']:.2f}")
    print(f"Test RMSE: {metrics['rmse']:.2f}")
    print(f"Test R2:   {metrics['r2']:.4f}")
    return metrics


def main() -> None:
    train_df, test_df = load_data()
    train_df = add_date_features(train_df)
    test_df = add_date_features(test_df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET]

    model = build_pipeline()
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    metrics_path = MODEL_DIR / "metrics.json"

    dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {model_path.relative_to(REPO_ROOT)}")
    print(f"Saved metrics to {metrics_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
