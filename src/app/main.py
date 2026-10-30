"""FastAPI service for the leave-or-stay churn model."""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "leave_or_stay.joblib"
FEATURES = ["tenure_months", "monthly_charges", "contract",
            "internet_service", "support_calls"]


class Customer(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_charges: float = Field(gt=0)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["fiber", "dsl", "none"]
    support_calls: int = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float


app = FastAPI(title="leave-or-stay churn API")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])[FEATURES]
    p = float(model.predict_proba(X)[0, 1])
    return Prediction(churn_probability=round(p, 3))