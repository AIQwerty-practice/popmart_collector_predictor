from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"

FavoriteSeries = Literal["Labubu", "Molly", "Skullpanda", "Dimoo", "Hirono", "Crybaby"]
CollectorType = Literal["casual", "display-focused", "completionist", "reseller", "trend-chaser"]
Region = Literal["North America", "Europe", "East Asia", "Southeast Asia", "Oceania"]


class CollectorInput(BaseModel):
    age: int = Field(..., ge=16, le=55)
    monthly_budget_usd: float = Field(..., ge=20, le=350)
    collection_size: int = Field(..., ge=0, le=180)
    monthly_purchases: int = Field(..., ge=0, le=20)
    resale_interest: int = Field(..., ge=0, le=10)
    social_media_engagement: int = Field(..., ge=0, le=10)
    blind_box_risk_tolerance: int = Field(..., ge=0, le=10)
    favorite_series: FavoriteSeries
    collector_type: CollectorType
    region: Region


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    interpretation: str


app = FastAPI(
    title="POP MART Collector Predictor",
    description="Predicts whether a collector is likely to buy the next POP MART release.",
    version="1.0.0",
)

model = None


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model file is missing. Run python scripts/train_model.py from the project root.",
            )
        model = joblib.load(MODEL_PATH)
    return model


def interpret_probability(probability: float) -> str:
    if probability >= 0.75:
        return "Very likely to buy the next release."
    if probability >= 0.55:
        return "Likely to buy, especially if the release matches their favorite series."
    if probability >= 0.35:
        return "Mixed intent; interest may depend on price, rarity, and design."
    return "Unlikely to buy the next release."


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "POP MART Collector Predictor",
        "message": "Use POST /predict to estimate next-release purchase intent.",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(collector: CollectorInput) -> PredictionResponse:
    pipeline = get_model()
    input_frame = pd.DataFrame([collector.model_dump()])
    probability = float(pipeline.predict_proba(input_frame)[0][1])
    prediction = int(probability >= 0.5)

    return PredictionResponse(
        prediction=prediction,
        label="Will buy next release" if prediction else "May skip next release",
        probability=round(probability, 4),
        interpretation=interpret_probability(probability),
    )
