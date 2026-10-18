import os
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# Setup import paths
SERVICE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_DIR))

from anomaly.anomaly_detector import predict_login
from nlp.phishing_detector import detect_phishing

app = FastAPI(
    title="DecisionIQ Security Service",
    description="Dual-engine security agent providing login anomaly detection and email phishing analysis.",
    version="1.0.0"
)

class LoginEventRequest(BaseModel):
    login_hour: float = Field(..., description="Hour of the day (0-23)")
    failed_attempts: float = Field(..., description="Failed attempts prior to login")
    new_device: int = Field(..., description="1 if unrecognised hardware/user-agent, 0 otherwise")
    login_frequency: float = Field(..., description="Average logins per hour")
    location_difference: int = Field(..., description="1 if novel geo-IP location, 0 otherwise")

class PhishingRequest(BaseModel):
    text: str = Field(..., description="Email body or message text to inspect")

class SecurityScoreRequest(BaseModel):
    login: Optional[LoginEventRequest] = None
    email_text: Optional[str] = None

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "decisioniq-security-service",
        "models": {
            "isolation_forest": True,
            "phishing_nlp": True
        }
    }

@app.post("/anomaly/score")
def score_login(login: LoginEventRequest):
    """Anisa's Isolation Forest login anomaly detector."""
    res = predict_login(login.model_dump())
    return res

@app.post("/nlp/classify")
def classify_text(req: PhishingRequest):
    """Jiya's DistilBERT / CEAS08 Phishing NLP detector."""
    res = detect_phishing(req.text)
    return res

@app.post("/security/score")
def calculate_security_score(req: SecurityScoreRequest):
    """
    Combined endpoint returning aggregated security threat score [0.0 - 1.0].
    Used by Finance Service Risk Aggregator & API Gateway.
    """
    scores = []
    anomaly_res = None
    phishing_res = None

    if req.login:
        anomaly_res = predict_login(req.login.model_dump())
        scores.append(anomaly_res["anomaly_score"])

    if req.email_text:
        phishing_res = detect_phishing(req.email_text)
        scores.append(phishing_res["phishing_probability"])

    if not scores:
        # Default nominal secure baseline if no specific event provided
        sec_score = 0.05
    else:
        # Max threat across vectors
        sec_score = float(max(scores))

    return {
        "security_score": round(sec_score, 4),
        "security_risk_percent": round(sec_score * 100, 1),
        "anomaly": anomaly_res,
        "phishing": phishing_res,
        "threat_level": "critical" if sec_score >= 0.75 else "elevated" if sec_score >= 0.50 else "watch" if sec_score >= 0.25 else "low"
    }

@app.get("/security/summary")
def get_security_summary():
    """Default baseline security telemetry for dashboard initialization."""
    default_login = {"login_hour": 10.0, "failed_attempts": 0.0, "new_device": 0, "login_frequency": 4.0, "location_difference": 0}
    res = predict_login(default_login)
    return {
        "status": "active",
        "baseline_threat_score": res["anomaly_score"],
        "recent_login_status": "normal" if not res["is_anomaly"] else "anomalous",
        "default_sample": res
    }
