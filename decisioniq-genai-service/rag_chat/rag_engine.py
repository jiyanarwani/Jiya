import os
import json
from typing import List, Dict, Any, Optional

class DecisionIQRAGEngine:
    """
    RAG Chat Engine for DecisionIQ.
    Retrieves grounded context from company financials, forecast horizon, and security event logs,
    and produces actionable answers via Gemini API or grounded expert synthesis.
    """
    def __init__(self):
        self.knowledge_base: List[Dict[str, Any]] = []

    def index_company_context(
        self,
        company_profile: Dict[str, Any],
        finance_data: Dict[str, Any],
        security_data: Dict[str, Any],
        warnings: List[str]
    ):
        """Builds in-memory chunk store for the current SME company."""
        chunks = []
        c_name = company_profile.get("company", "Selected Company")
        sector = company_profile.get("sector", "Enterprise")

        # Chunk 1: Company Profile & Core Metrics
        chunks.append({
            "topic": "profile",
            "text": f"Company: {c_name}. Sector: {sector}. Employees: {company_profile.get('employees')}. "
                    f"Annual Revenue: ${company_profile.get('revenue', 0):,.0f}. Operating Expenses (OPEX): ${company_profile.get('opex', 0):,.0f}. "
                    f"Outstanding Loan Balance: ${company_profile.get('loan', 0):,.0f}. "
                    f"Accounts Receivable (AR) Days: {company_profile.get('ar_days')}. Inventory Days: {company_profile.get('inventory_days')}."
        })

        # Chunk 2: Financial Distress & Forecasting
        risk_pct = finance_data.get("risk_probability", 0) * 100
        forecast_trend = finance_data.get("forecast_trend", "stable")
        chunks.append({
            "topic": "finance_risk",
            "text": f"{c_name} financial stress analysis: Cash-flow stress risk probability is {risk_pct:.1f}%. "
                    f"Forecast 90-day trajectory indicates {forecast_trend} trend. "
                    f"Top financial risk factors from SHAP analysis: {', '.join([f['feature'] for f in finance_data.get('shap_features', [])])}."
        })

        # Chunk 3: Security Telemetry & Anomalies
        sec_score = security_data.get("security_score", 0.05) * 100
        chunks.append({
            "topic": "security",
            "text": f"{c_name} cybersecurity posture: Overall threat index is {sec_score:.1f}%. "
                    f"Login anomaly status: {security_data.get('anomaly_status', 'normal')}. "
                    f"Recent phishing exposure: {security_data.get('phishing_status', 'monitored')}."
        })

        # Chunk 4: Warnings & Health
        chunks.append({
            "topic": "warnings",
            "text": f"Early warning alerts for {c_name}: {'; '.join(warnings) if warnings else 'No critical warnings triggered'}. "
                    f"Composite Business Vulnerability: {finance_data.get('composite_score', 0) * 100:.1f}%."
        })

        self.knowledge_base = chunks

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Simple keyword & semantic overlap retriever."""
        if not self.knowledge_base:
            return ["No business telemetry currently loaded in context."]

        q_terms = set(query.lower().split())
        scored = []
        for chunk in self.knowledge_base:
            text = chunk["text"]
            t_words = set(text.lower().split())
            overlap = len(q_terms.intersection(t_words))
            scored.append((overlap, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    def answer_query(self, query: str) -> Dict[str, Any]:
        """Answers user question grounded in retrieved company data."""
        retrieved_texts = self.retrieve_relevant_chunks(query)
        context_str = "\n".join(f"- {t}" for t in retrieved_texts)

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"""
You are DecisionIQ Business Assistant. Answer the executive question strictly using the provided ground truth context.
If the answer is not contained in the context, say so concisely.

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:
"""
                resp = model.generate_content(prompt)
                return {
                    "answer": resp.text.strip(),
                    "source": "Gemini-RAG",
                    "retrieved_context": retrieved_texts
                }
            except Exception:
                pass

        # Grounded rule-based reasoning engine
        q_lower = query.lower()
        if "revenue" in q_lower or "sales" in q_lower:
            ans = f"Based on the financial records, the annual revenue is recorded along with a 90-day forward forecast. {retrieved_texts[0]}"
        elif "risk" in q_lower or "score" in q_lower or "health" in q_lower:
            ans = f"The composite business risk evaluation synthesizes 70% financial stress and 30% cyber risk. {retrieved_texts[0]}"
        elif "security" in q_lower or "phishing" in q_lower or "login" in q_lower or "hack" in q_lower:
            sec_chunk = next((t for t in retrieved_texts if "cybersecurity" in t.lower() or "threat" in t.lower()), retrieved_texts[0])
            ans = f"Cybersecurity telemetry indicates: {sec_chunk}"
        elif "warning" in q_lower or "alert" in q_lower or "runway" in q_lower:
            warn_chunk = next((t for t in retrieved_texts if "warning" in t.lower()), retrieved_texts[0])
            ans = f"Active system warnings: {warn_chunk}"
        elif "what if" in q_lower or "simulate" in q_lower or "reduce" in q_lower or "cut" in q_lower:
            ans = "You can simulate operational changes in real-time using the What-If Sandbox. For instance, adjusting OPEX or AR days will instantly re-evaluate the XGBoost distress model and Prophet forecast."
        else:
            ans = f"According to current company records:\n{retrieved_texts[0]}\nFeel free to ask about specific risk drivers, revenue forecasts, or security anomalies."

        return {
            "answer": ans,
            "source": "DecisionIQ-RAG-Engine",
            "retrieved_context": retrieved_texts
        }
