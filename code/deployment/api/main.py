"""Stage 3: model API.

Loads the trained pipeline from models/model.joblib and exposes it as a
FastAPI /predict endpoint.

Run from the repository root:  uvicorn code.deployment.api.main:app --reload
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code" / "models"))

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field

from features import FEATURE_COLUMNS

MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
model = None

# to load the model file once when container starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run code/models/train.py first to produce it."
        )
    model = load(MODEL_PATH)
    yield


app = FastAPI(title="Walmart Weekly Sales Predictor", lifespan=lifespan)


class PredictionRequest(BaseModel):
    store: int = Field(..., ge=1, description="Store number")
    holiday_flag: int = Field(..., ge=0, le=1, description="1 if a holiday week, else 0")
    temperature: float = Field(..., description="Average temperature in the region (F)")
    fuel_price: float = Field(..., ge=0, description="Fuel price in the region")
    cpi: float = Field(..., ge=0, description="Consumer price index")
    unemployment: float = Field(..., ge=0, description="Unemployment rate")
    year: int = Field(..., ge=2000, le=2100, description="Year")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    week: int = Field(..., ge=1, le=53, description="ISO week number (1-53)")


class PredictionResponse(BaseModel):
    predicted_weekly_sales: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = {
        "Store": request.store,
        "Holiday_Flag": request.holiday_flag,
        "Temperature": request.temperature,
        "Fuel_Price": request.fuel_price,
        "CPI": request.cpi,
        "Unemployment": request.unemployment,
        "Year": request.year,
        "Month": request.month,
        "Week": request.week,
    }
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    prediction = model.predict(X)[0]
    return PredictionResponse(predicted_weekly_sales=float(prediction))
