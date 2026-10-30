"""Tests for the leave-or-stay churn service."""
from fastapi.testclient import TestClient

from src.app.main import MODEL_PATH, app, model

client = TestClient(app)

VALID = {
    "tenure_months": 5,
    "monthly_charges": 89.9,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


def test_api_contract():
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert list(body) == ["churn_probability"]
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_model_loads():
    assert MODEL_PATH.exists()
    assert hasattr(model, "predict_proba")


def test_input_schema_rejects_bad_requests():
    bad_contract = {**VALID, "contract": "weekly"}
    missing_field = {k: v for k, v in VALID.items() if k != "support_calls"}
    negative_tenure = {**VALID, "tenure_months": -1}
    for payload in (bad_contract, missing_field, negative_tenure):
        assert client.post("/predict", json=payload).status_code == 422


def test_edge_case_brand_new_customer():
    payload = {**VALID, "tenure_months": 0, "support_calls": 0,
               "internet_service": "none"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0


def test_prediction_is_deterministic():
    first = client.post("/predict", json=VALID).json()
    second = client.post("/predict", json=VALID).json()
    assert first == second
