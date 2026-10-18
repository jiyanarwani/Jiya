import os
from typing import List, Dict, Any, Optional

def generate_risk_narration(
    company_name: str,
    composite_score: float,
    finance_score: float,
    security_score: float,
    shap_features: List[Dict[str, Any]],
    warnings: List[str],
    company_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates explainable plain-English narrative from SHAP features, financial distress probability,
    and security threat vectors using Gemini API (with deterministic fallback).
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    profile = company_profile or {}

    # Build structured narrative prompt / context
    drivers_up = [f["feature"].replace("_", " ").title() for f in shap_features if f.get("direction") == "increases_risk"]
    drivers_down = [f["feature"].replace("_", " ").title() for f in shap_features if f.get("direction") == "decreases_risk"]

    # Try Gemini if API key is provided
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""
You are DecisionIQ's AI Business Advisor for SME executives.
Analyze this business health report and provide a concise, sharp executive briefing.

Company: {company_name}
Sector: {profile.get('sector', 'General Enterprise')}
Revenue: ${profile.get('revenue', 0):,.0f} | OPEX: ${profile.get('opex', 0):,.0f} | Loan: ${profile.get('loan', 0):,.0f}
Composite Business Vulnerability Score: {composite_score * 100:.1f}% (Weight: 70% Financial Stress, 30% Security Threat)
Financial Cash-Stress Risk: {finance_score * 100:.1f}%
Cybersecurity Threat Score: {security_score * 100:.1f}%
Top Factors Driving Risk Up (SHAP): {', '.join(drivers_up) if drivers_up else 'None identified'}
Top Factors Mitigating Risk (SHAP): {', '.join(drivers_down) if drivers_down else 'None identified'}
Active Early Warnings: {', '.join(warnings) if warnings else 'None'}

Please output JSON with:
"executive_summary": "2-3 sentences summarizing overall business health and vulnerability.",
"financial_analysis": "Key financial bottlenecks explained in plain language.",
"security_analysis": "Assessment of cyber risk posture and vulnerability.",
"recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"]
"""
            response = model.generate_content(prompt)
            text = response.text
            # Basic cleanup if markdown json returned
            import json, re
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return {
                    "source": "Gemini-1.5-Flash",
                    "executive_summary": parsed.get("executive_summary", ""),
                    "financial_analysis": parsed.get("financial_analysis", ""),
                    "security_analysis": parsed.get("security_analysis", ""),
                    "recommendations": parsed.get("recommendations", [])
                }
        except Exception:
            pass  # Fall through to deterministic intelligence engine

    # High-quality deterministic intelligence engine
    health_status = (
        "CRITICAL VULNERABILITY" if composite_score >= 0.70
        else "ELEVATED CONCERN" if composite_score >= 0.45
        else "MODERATE WATCH" if composite_score >= 0.25
        else "STABLE & HEALTHY"
    )

    summary_sentences = [
        f"{company_name} is currently assessed at **{health_status}** with a composite vulnerability index of {composite_score * 100:.1f}%."
    ]

    if finance_score >= 0.50:
        summary_sentences.append(
            f"The primary strain stems from cash-flow stress ({finance_score * 100:.1f}%), where working capital pressures and operational overhead threaten short-term solvency."
        )
    else:
        summary_sentences.append(
            f"Financial operations remain relatively resilient with cash-flow stress contained at {finance_score * 100:.1f}%."
        )

    if security_score >= 0.40:
        summary_sentences.append(
            f"Concurrently, elevated cyber exposure ({security_score * 100:.1f}%) introduces operational disruption risk that could amplify financial losses."
        )
    else:
        summary_sentences.append(
            "Cybersecurity perimeter appears stable with minimal anomalous threat signals detected."
        )

    # Financial analysis from SHAP
    fin_analysis_parts = []
    if drivers_up:
        fin_analysis_parts.append(
            f"Key factors driving liquidity pressure include: **{', '.join(drivers_up[:3])}**. These metrics are restricting available cash buffer."
        )
    if drivers_down:
        fin_analysis_parts.append(
            f"Stabilizing cushions include: **{', '.join(drivers_down[:2])}**, helping mitigate deeper cash-flow deterioration."
        )
    if not fin_analysis_parts:
        fin_analysis_parts.append("Financial metrics indicate balanced operational cash conversion.")

    # Recommendations
    recs = []
    if "Accounts Receivable Days" in drivers_up or (profile.get("ar_days", 0) > 45):
        recs.append("Tighten credit terms: Implement milestone billing and early-settlement discounts to accelerate cash collection from delayed accounts receivable.")
    if "Opex Usd" in drivers_up or "Operating Expense" in drivers_up:
        recs.append("Restructure operational expenditures: Audit discretionary monthly burn to extend cash runway beyond the critical threshold.")
    if security_score >= 0.35:
        recs.append("Enforce multi-factor authentication (MFA) and initiate employee security awareness training against credential harvesting.")
    if "Inventory Days" in drivers_up:
        recs.append("Rationalize working capital tied up in inventory through just-in-time stock replenishment.")
    if len(recs) < 3:
        recs.append("Establish a 90-day rolling cash-flow forecast review cadence with quarterly credit line pre-arrangements.")

    return {
        "source": "DecisionIQ-GenAI-Engine",
        "executive_summary": " ".join(summary_sentences),
        "financial_analysis": " ".join(fin_analysis_parts),
        "security_analysis": f"Security risk telemetry scored at {security_score * 100:.1f}%. " + (
            "Anomalous access patterns or suspicious communications warrant immediate endpoint triage."
            if security_score >= 0.40 else "Access controls and communication channels remain within normal operational baselines."
        ),
        "recommendations": recs[:3]
    }
