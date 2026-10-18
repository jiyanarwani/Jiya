# DecisionIQ — AI-Powered SME Financial & Cybersecurity Intelligence Platform

> **Final-Year Engineering Project**  
> A unified decision-support platform designed for SMEs combining **90-Day Cash Flow Forecasting**, **Cash-Stress Risk Modeling**, **Explainable AI (SHAP)**, **Dual-Engine Cybersecurity Monitoring**, and **GenAI Executive Narrative Synthesis**.

---

## 1. System Architecture

```text
                             DecisionIQ Dashboard (Streamlit :8501)
                                               │
                                  API Gateway (FastAPI :8000)
                                               │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   Finance Service              Security Service               GenAI Service
   (FastAPI :8001)               (FastAPI :8002)              (FastAPI :8003)
   ├── Prophet Forecaster        ├── Isolation Forest Anomaly  ├── SHAP Risk Narrator
   ├── XGBoost Risk Model        ├── DistilBERT / CEAS08 NLP   ├── Gemini Synthesizer
   ├── SHAP Explainer            └── Unified /security/score   └── RAG Chat with Data
   └── Risk Aggregator (70/30)
```

---

## 2. Core Modules & Ownership

| Service / Subsystem | Core Technologies | Description |
| :--- | :--- | :--- |
| **Finance Service** | Prophet, XGBoost, SHAP, Scikit-learn | 90-day daily cashflow forecasting, bankruptcy/distress risk classification, directional SHAP attribution, rule-based runway alerts, and solvency-weighted risk aggregation (70% finance, 30% security). |
| **Security Service** | Isolation Forest, DistilBERT, CEAS08 NLP | Dual-engine threat detection: unsupervised login behavior anomaly scoring ($K=15$ sigmoid decision margin) and email phishing text classification. |
| **GenAI Service** | Google Gemini 1.5, RAG Knowledge Store | Synthesizes SHAP feature impacts, financial stress, and cybersecurity vectors into executive summaries with 3 actionable recommendations. RAG chat answers questions grounded in company telemetry. |
| **API Gateway** | FastAPI, Uvicorn, HTTPX | Central API routing, health telemetry aggregation across microservices, and orchestrator for the real-time `/simulate` What-If sandbox. |
| **Unified Dashboard** | Streamlit, Matplotlib | Dark-themed executive dashboard featuring live scenario simulation, interactive Prophet charts with confidence intervals, SHAP impact bar charts, and data Q&A. |

---

## 3. Quick Start Guide

### Prerequisites
- Python 3.10 – 3.14
- Git

### Installation
```bash
git clone https://github.com/jiyanarwani/DECISIONIQ.git
cd DECISIONIQ
pip install -r decisioniq-finance-service/requirements.txt
pip install streamlit
```

### Option A: Launch Interactive Dashboard (Recommended)
Double-click `run_dashboard.bat` or run:
```bash
python -m streamlit run app_unified.py --server.port 8501
```
Open your browser to: **http://localhost:8501**

### Option B: Launch All Microservices
```bash
python run_all_services.py
```
- **API Gateway Docs:** `http://localhost:8000/docs`
- **Finance Service Docs:** `http://localhost:8001/docs`
- **Security Service Docs:** `http://localhost:8002/docs`
- **GenAI Service Docs:** `http://localhost:8003/docs`

### Option C: Run Test Suite
```bash
pytest decisioniq-finance-service/tests/test_finance_service.py
```

---

## 4. Key Formulas & Mathematics

1. **Composite Business Health Index:**
   $$\text{Composite Score} = (0.70 \times \text{Finance Stress Risk}) + (0.30 \times \text{Cybersecurity Threat})$$

2. **Isolation Forest Anomaly Score:**
   $$\text{Anomaly Score} = \frac{1}{1 + e^{15 \times \text{decision\_function}(X)}}$$

3. **What-If Simulation:**
   In-memory re-evaluation of Prophet forecasting and XGBoost distress probabilities with real-time SHAP recalculation and GenAI re-narration without modifying persistent storage.
