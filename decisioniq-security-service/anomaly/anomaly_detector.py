from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model" / "isolation_forest.joblib"

FEATURES = [
    "login_hour",
    "failed_attempts",
    "new_device",
    "login_frequency",
    "location_difference",
]

K = 15


def to_anomaly_score(decision):
    """Convert Isolation Forest decision_function output into a 0-1 score.
    Above 0.5 = suspicious, below 0.5 = normal."""
    return 1 / (1 + np.exp(K * np.asarray(decision)))


_model = None


def get_model():
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            _model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Isolation Forest model not found at {MODEL_PATH}")
    return _model


def predict_login(login: dict) -> dict:
    """Score a login event using Anisa's Isolation Forest model."""
    model = get_model()
    X = pd.DataFrame([login])[FEATURES]

    is_anomaly = bool(model.predict(X)[0] == -1)
    raw_decision = float(model.decision_function(X)[0])
    score = float(to_anomaly_score(raw_decision))

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(score, 4),
        "risk_score": round(score * 100, 1),
        "raw_decision": round(raw_decision, 4),
        "features": {k: login.get(k, 0) for k in FEATURES}
    }
