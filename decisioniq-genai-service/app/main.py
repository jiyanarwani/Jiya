import os
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

SERVICE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SERVICE_DIR))

from summarizer.risk_narrator import generate_risk_narration
from rag_chat.rag_engine import DecisionIQRAGEngine

app = FastAPI(
    title="DecisionIQ GenAI Service",
    description="Explainable GenAI narration and RAG data chat service powered by Gemini and SHAP synthesis.",
    version="1.0.0"
)

rag_engine = DecisionIQRAGEngine()

class NarrationRequest(BaseModel):
    company_name: str
    composite_score: float
    finance_score: float
    security_score: float
    shap_features: List[Dict[str, Any]] = []
    warnings: List[str] = []
    company_profile: Optional[Dict[str, Any]] = None

class IndexContextRequest(BaseModel):
    company_profile: Dict[str, Any]
    finance_data: Dict[str, Any]
    security_data: Dict[str, Any]
    warnings: List[str] = []

class ChatRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "decisioniq-genai-service",
        "has_gemini_api_key": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    }

@app.post("/genai/narrate")
def narrate_risk(req: NarrationRequest):
    """Generates plain-English business health narration from SHAP & risk scores."""
    result = generate_risk_narration(
        company_name=req.company_name,
        composite_score=req.composite_score,
        finance_score=req.finance_score,
        security_score=req.security_score,
        shap_features=req.shap_features,
        warnings=req.warnings,
        company_profile=req.company_profile
    )
    return result

@app.post("/genai/index")
def index_data(req: IndexContextRequest):
    """Indexes telemetry chunks for RAG chat retrieval."""
    rag_engine.index_company_context(
        company_profile=req.company_profile,
        finance_data=req.finance_data,
        security_data=req.security_data,
        warnings=req.warnings
    )
    return {"status": "indexed", "chunk_count": len(rag_engine.knowledge_base)}

@app.post("/genai/chat")
def chat(req: ChatRequest):
    """Ground-truth Q&A with business data."""
    return rag_engine.answer_query(req.query)
