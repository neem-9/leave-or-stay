# leave-or-stay

[![build-scan-ship](https://github.com/neem-9/leave-or-stay/actions/workflows/build-scan-ship.yml/badge.svg)](https://github.com/neem-9/leave-or-stay/actions/workflows/build-scan-ship.yml)

A small churn-prediction service. It trains a logistic regression model on
customer data and serves it through a FastAPI API, packaged in a Docker image
that is tested, scanned and published by GitHub Actions.

## Run it (one command)

```bash
docker build -t leave-or-stay . && docker run --rm -p 8000:8000 leave-or-stay
```

The service listens on http://127.0.0.1:8000.

## Try it

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 5, "monthly_charges": 89.9, "contract": "month-to-month", "internet_service": "fiber", "support_calls": 3}'
```

Invalid requests (for example an unknown `contract`) are rejected with HTTP 422.

## Run the tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest -v
```

## How the pipeline works

- Every pull request: lint (`ruff`) and tests (`pytest`).
- Every push to `main`: build the image, scan it with Trivy, and push it to
  GHCR tagged with the commit SHA.
- `main` is protected: changes go through a pull request and the `checks` job
  must pass.
- Every third-party action is pinned to a full commit SHA.

## Layout

- `src/train/train.py` trains the model and saves it to `models/`.
- `src/app/main.py` is the API (`/health`, `/predict`).
- `tests/` holds the pytest tests.
- `Dockerfile` is a multi-stage, non-root build with a pinned base image and a
  health check.
