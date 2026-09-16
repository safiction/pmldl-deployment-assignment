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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models"

TARGET = "Weekly_Sales"
CATEGORICAL_FEATURES = ["Store", "Holiday_Flag"]
NUMERIC_FEATURES = ["Temperature", "Fuel_Price", "CPI", "Unemployment",
                     "Year", "Month", "Week"]
ALPHA = 1.0
RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    print(f"Loaded train: {len(train_df)} rows, test: {len(test_df)} rows")
    return train_df, test_df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the Date column into plain numeric parts a linear model can use."""
    df = df.copy()
    date = pd.to_datetime(df["Date"])
    df["Year"] = date.dt.year
    df["Month"] = date.dt.month
    df["Week"] = date.dt.isocalendar().week.astype(int)
    return df.drop(columns=["Date"])


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("features", preprocessor),
        ("model", Ridge(alpha=ALPHA, random_state=RANDOM_STATE)),
    ])


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

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X_train, y_train = train_df[feature_columns], train_df[TARGET]
    X_test, y_test = test_df[feature_columns], test_df[TARGET]

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
