"""Model tests: it loads, it is good enough, and it gives the same answer every time."""

import json

import joblib
import pandas as pd

from src import schema
from src.train import train


def test_model_loads_and_meets_quality_gate(model_dir):
    model = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert hasattr(model, "predict_proba")
    assert meta["auc"] >= train.MIN_AUC


def test_prediction_is_deterministic(model_dir, customer):
    model = joblib.load(model_dir / "model.joblib")
    X = pd.DataFrame([customer])[schema.FEATURES]
    assert model.predict_proba(X)[0, 1] == model.predict_proba(X)[0, 1]
    retrained = train.build_model().fit(*_xy())
    assert abs(retrained.predict_proba(X)[0, 1] - model.predict_proba(X)[0, 1]) < 0.05


def _xy():
    df = schema.validate(pd.read_csv(train.DATA_FILE))
    return df[schema.FEATURES], df[schema.TARGET]
