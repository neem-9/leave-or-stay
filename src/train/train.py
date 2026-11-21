"""Train the churn model, check its quality, and save it with metadata that says exactly what it is.

python -m src.train.train
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import schema

DATA_FILE = "data/churn_v1.csv"  # the data file the model is trained on
MIN_AUC = 0.70  # quality gate: a worse model never gets packaged
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))


def build_model() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), schema.NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), schema.CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])


def train(data_file: str = DATA_FILE, out_dir: Path = MODEL_DIR) -> dict:
    df = schema.validate(pd.read_csv(data_file))
    X, y = df[schema.FEATURES], df[schema.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model().fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    auc, f1 = roc_auc_score(y_te, proba), f1_score(y_te, proba > 0.5)
    if auc < MIN_AUC:
        raise SystemExit(f"Quality gate failed: AUC {auc:.3f} is below {MIN_AUC}")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    meta = {
        "model_version": os.getenv("GIT_SHA", "local")[:7],
        "data_file": data_file,
        "data_sha256": hashlib.sha256(Path(data_file).read_bytes()).hexdigest()[:12],
        "auc": round(auc, 3),
        "f1": round(f1, 3),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
