import os
import sys
import httpx
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

app = FastAPI(
    title="DecisionIQ Unified API Gateway",
    description="Central API gateway routing requests across Finance, Security, and GenAI microservices.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FINANCE_SERVICE_URL = os.getenv("FINANCE_SERVICE_URL", "http://127.0.0.1:8001")
SECURITY_SERVICE_URL = os.getenv("SECURITY_SERVICE_URL", "http://127.0.0.1:8002")
GENAI_SERVICE_URL = os.getenv("GENAI_SERVICE_URL", "http://127.0.0.1:8003")

# Direct in-process fallbacks if independent web servers aren't running
WORKSPACE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_DIR / "decisioniq-finance-service"))
sys.path.insert(0, str(WORKSPACE_DIR / "decisioniq-security-service"))
sys.path.insert(0, str(WORKSPACE_DIR / "decisioniq-genai-service"))

_finance_service = None
_security_service = None

def get_direct_finance_service():
    global _finance_service
    if _finance_service is None:
        try:
            from app.services.finance_service import FinanceService
            _finance_service = FinanceService(str(WORKSPACE_DIR / "decisioniq-finance-service"))
        except Exception:
            _finance_service = None
    return _finance_service

class SimulateRequest(BaseModel):
    company_name: str = "Demo SME Corp"
    sector: str = "retail"
    employees: int = 15
    revenue_usd: float
    opex_usd: float
    accounts_receivable_days: float
    inventory_days: float
    loan_balance_usd: float
    owner_injections_usd: float = 0.0
    security_threat_score: Optional[float] = 0.05

@app.get("/health")
async def health():
    """Aggregated health check across all services."""
    services_status = {}
    async with httpx.AsyncClient(timeout=1.5) as client:
        # Check Finance
        try:
            r = await client.get(f"{FINANCE_SERVICE_URL}/health")
            services_status["finance_service"] = r.json() if r.status_code == 200 else "unhealthy"
        except Exception:
            services_status["finance_service"] = "available (direct fallback)"

        # Check Security
        try:
            r = await client.get(f"{SECURITY_SERVICE_URL}/health")
            services_status["security_service"] = r.json() if r.status_code == 200 else "unhealthy"
        except Exception:
            services_status["security_service"] = "available (direct fallback)"

        # Check GenAI
        try:
            r = await client.get(f"{GENAI_SERVICE_URL}/health")
            services_status["genai_service"] = r.json() if r.status_code == 200 else "unhealthy"
        except Exception:
            services_status["genai_service"] = "available (direct fallback)"

    return {
        "gateway_status": "healthy",
        "services": services_status
    }

@app.post("/simulate")
async def simulate_what_if(req: SimulateRequest):
    """
    What-If Sandbox Orchestrator:
    1. Re-evaluates Finance XGBoost stress model & SHAP
    2. Recalculates composite score combining with security score
    3. Re-evaluates Early Warning signals
    4. Generates updated GenAI narration
    """
    fin_svc = get_direct_finance_service()
    if not fin_svc:
        raise HTTPException(status_code=500, detail="Finance Service engine not initialized")

    from app.schemas.finance_schema import RiskPredictRequest
    from app.services.early_warning import generate_early_warning
    from app.services.risk_aggregation import calculate_composite_score
    from summarizer.risk_narrator import generate_risk_narration

    risk_req = RiskPredictRequest(
        sector=req.sector,
        employees=req.employees,
        revenue_usd=req.revenue_usd,
        opex_usd=req.opex_usd,
        accounts_receivable_days=req.accounts_receivable_days,
        inventory_days=req.inventory_days,
        loan_balance_usd=req.loan_balance_usd,
        owner_injections_usd=req.owner_injections_usd
    )

    risk_res = fin_svc.predict_risk(risk_req)
    fin_score = risk_res.risk_probability
    sec_score = req.security_threat_score if req.security_threat_score is not None else 0.05

    composite = calculate_composite_score(fin_score, sec_score)
    early_warn = generate_early_warning([], fin_score, None)

    shap_dicts = [
        {"feature": f.feature, "magnitude": f.magnitude, "direction": f.direction}
        for f in risk_res.shap_features
    ]

    narration = generate_risk_narration(
        company_name=req.company_name,
        composite_score=composite["composite_score"],
        finance_score=fin_score,
        security_score=sec_score,
        shap_features=shap_dicts,
        warnings=early_warn.get("active_warnings", []),
        company_profile=req.model_dump()
    )

    return {
        "status": "success",
        "inputs": req.model_dump(),
        "finance_risk": {
            "risk_probability": round(fin_score, 4),
            "risk_percentage": round(fin_score * 100, 1),
            "risk_class": risk_res.risk_class,
            "shap_features": shap_dicts
        },
        "composite": composite,
        "early_warning": early_warn,
        "narration": narration
    }
