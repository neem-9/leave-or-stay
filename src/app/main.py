"""Churn prediction service.

    uvicorn src.app.main:app --port 8000

/health   which model is running (Lab 1 Twist 3, Lab 3 lineage)
/predict  a churn probability for one customer (Lab 2)
/metrics  Prometheus metrics: requests, latency, errors, predictions (Lab 5)
"""

import json
import os
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src import schema

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
model = joblib.load(MODEL_DIR / "model.joblib")
meta = json.loads((MODEL_DIR / "metadata.json").read_text())

app = FastAPI(title="AI381 churn API (starter)", version=meta["model_version"])

REQUESTS = Counter("http_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Histogram(
    "churn_prediction_probability",
    "Predicted churn probability",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=1000)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    support_calls: int = Field(ge=0, le=50)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    if path != "/metrics":
        LATENCY.labels(path).observe(time.perf_counter() - start)
        REQUESTS.labels(path, str(response.status_code)).inc()
    return response


@app.get("/health")
def health():
    """Which model is running, trained on which data, from which commit."""
    return {"status": "ok", **meta}


@app.post("/predict")
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])[schema.FEATURES]
    p = float(model.predict_proba(X)[0, 1])  # column 1 = probability of churn
    PREDICTIONS.observe(p)
    return {"churn_probability": round(p, 3), "model_version": meta["model_version"]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
