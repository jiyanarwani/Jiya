import os
from pathlib import Path
import joblib

NLP_DIR = Path(__file__).parent
DISTILBERT_MODEL_PATH = NLP_DIR / "phishing_distilbert"
FAST_MODEL_PATH = NLP_DIR / "phishing_fast_model.joblib"

_distilbert_tokenizer = None
_distilbert_model = None
_fast_model = None


def _load_distilbert():
    global _distilbert_tokenizer, _distilbert_model
    if _distilbert_model is None and DISTILBERT_MODEL_PATH.exists():
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            _distilbert_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_MODEL_PATH))
            _distilbert_model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_MODEL_PATH))
            _distilbert_model.eval()
        except Exception:
            _distilbert_model = None
    return _distilbert_model, _distilbert_tokenizer


def _load_fast_model():
    global _fast_model
    if _fast_model is None and FAST_MODEL_PATH.exists():
        _fast_model = joblib.load(str(FAST_MODEL_PATH))
    return _fast_model


def detect_phishing(text: str) -> dict:
    """
    Detects phishing email/text using DistilBERT or the high-accuracy trained model.
    Matches the schema designed in Group 3 Architecture.
    """
    if not text or not text.strip():
        return {
            "is_phishing": False,
            "confidence": 1.0,
            "phishing_probability": 0.0,
            "risk_score": 0,
            "model_type": "none",
            "message": "Empty text provided"
        }

    # Try DistilBERT if available
    db_model, db_tokenizer = _load_distilbert()
    if db_model is not None and db_tokenizer is not None:
        import torch
        inputs = db_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            outputs = db_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        phishing_prob = float(probabilities[0][1].item())
        is_phishing = phishing_prob >= 0.5
        confidence = phishing_prob if is_phishing else 1.0 - phishing_prob
        return {
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "phishing_probability": round(phishing_prob, 4),
            "risk_score": round(phishing_prob * 100),
            "model_type": "DistilBERT"
        }

    # Otherwise use fast CEAS08 model
    fast_m = _load_fast_model()
    if fast_m is not None:
        prob = float(fast_m.predict_proba([text])[0][1])
        is_phishing = prob >= 0.5
        confidence = prob if is_phishing else 1.0 - prob
        return {
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "phishing_probability": round(prob, 4),
            "risk_score": round(prob * 100),
            "model_type": "CEAS08-Classifier"
        }

    # Fallback heuristic
    suspicious_keywords = ["urgent", "verify your account", "password", "bank transfer", "suspended", "click here", "wire"]
    hits = sum(1 for kw in suspicious_keywords if kw in text.lower())
    prob = min(0.95, hits * 0.25)
    return {
        "is_phishing": prob >= 0.5,
        "confidence": 0.8,
        "phishing_probability": round(prob, 4),
        "risk_score": round(prob * 100),
        "model_type": "Heuristic-Fallback"
    }
