"""Feature contract shared between training (train.py) and serving (the API).

Keeping the column lists and the pipeline shape here guarantees the
API builds inputs the same as for training.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Weekly_Sales"
CATEGORICAL_FEATURES = ["Store", "Holiday_Flag"]
NUMERIC_FEATURES = ["Temperature", "Fuel_Price", "CPI", "Unemployment",
                     "Year", "Month", "Week"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

ALPHA = 1.0
RANDOM_STATE = 42


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the Date column into plain numeric parts a linear model can use.

    Only needed when processing the raw dataset (training); at inference
    time the API collects Year/Month/Week directly from the user.
    """
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
