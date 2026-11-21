"""API contract tests: what callers of the service can rely on."""


def test_health_reports_model_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert {"model_version", "data_file", "auc"} <= body.keys()


def test_predict_returns_a_probability(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0


def test_invalid_input_is_rejected(client, customer):
    assert client.post("/predict", json={**customer, "contract": "weekly"}).status_code == 422
    assert client.post("/predict", json={**customer, "support_calls": -1}).status_code == 422
    assert client.post("/predict", json={"tenure_months": 5}).status_code == 422


def test_risky_customer_scores_higher_than_loyal_customer(client):
    """A behaviour test: a new, unhappy month-to-month customer must look riskier than a loyal one."""
    risky = {
        "tenure_months": 2,
        "monthly_charges": 110,
        "contract": "month-to-month",
        "internet_service": "fiber",
        "support_calls": 6,
    }
    loyal = {
        "tenure_months": 60,
        "monthly_charges": 40,
        "contract": "two-year",
        "internet_service": "dsl",
        "support_calls": 0,
    }
    p_risky = client.post("/predict", json=risky).json()["churn_probability"]
    p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
    assert p_risky > p_loyal


def test_metrics_endpoint_exposes_prometheus_metrics(client, customer):
    client.post("/predict", json=customer)
    text = client.get("/metrics").text
    assert "http_requests_total" in text
    assert "churn_prediction_probability_bucket" in text
