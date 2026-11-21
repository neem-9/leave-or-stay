import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Train a fresh model into a temporary folder, exactly as the pipeline does."""
    d = tmp_path_factory.mktemp("models")
    os.environ["MODEL_DIR"] = str(d)
    from src.train import train

    train.train(out_dir=d)
    return d


@pytest.fixture(scope="session")
def client(model_dir):
    import src.app.main as main  # imported only now, so it loads the model trained above

    return TestClient(main.app)


@pytest.fixture
def customer():
    return {
        "tenure_months": 5,
        "monthly_charges": 89.9,
        "contract": "month-to-month",
        "internet_service": "fiber",
        "support_calls": 3,
    }
