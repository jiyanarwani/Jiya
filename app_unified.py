"""
DecisionIQ — Comprehensive AI Business Health Dashboard
=========================================================
Fully integrated system combining:
1. Finance Agent (Prophet 90-day forecasting + XGBoost distress classifier + SHAP explainability)
2. Security Agent (Isolation Forest login anomaly detection + Phishing text classification)
3. GenAI Layer (Gemini explainability, risk narration, and RAG chat grounded in business data)
4. What-If Sandbox (Real-time multi-variable scenario simulation)

Built according to the Group 3 System Architecture & strictly adopting the
Finance Branch design language, theme, and styling.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
FINANCE_DIR = BASE_DIR / "decisioniq-finance-service"
SECURITY_DIR = BASE_DIR / "decisioniq-security-service"
GENAI_DIR = BASE_DIR / "decisioniq-genai-service"

sys.path.insert(0, str(FINANCE_DIR))
sys.path.insert(0, str(SECURITY_DIR))
sys.path.insert(0, str(GENAI_DIR))

# Import finance modules
from app.schemas.finance_schema import RiskPredictRequest
from app.services.finance_service import FinanceService
from app.services.early_warning import generate_early_warning
from app.services.risk_aggregation import calculate_composite_score

# Import security modules
from anomaly.anomaly_detector import predict_login
from nlp.phishing_detector import detect_phishing

# Import GenAI modules
from summarizer.risk_narrator import generate_risk_narration
from rag_chat.rag_engine import DecisionIQRAGEngine

# ---------------------------------------------------------------------------
# Page configuration & Theme CSS (strictly matching Finance Branch UI)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DecisionIQ — AI Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ---------- Global ---------- */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ---------- Header Banner ---------- */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 14px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .hero-banner h1 { margin: 0; font-size: 2.2rem; font-weight: 700; color: #ffffff; }
    .hero-banner p  { margin: 0.5rem 0 0; opacity: 0.9; font-size: 1.05rem; }

    /* ---------- Section Cards ---------- */
    .card {
        background: #ffffff0a;
        border: 1px solid #ffffff15;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .card-accent {
        background: #1e293b55;
        border: 1px solid #38bdf844;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }

    /* ---------- Risk Badges ---------- */
    .risk-high {
        display: inline-block;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: #fff;
        padding: 6px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .risk-medium {
        display: inline-block;
        background: linear-gradient(135deg, #e67e22, #d35400);
        color: #fff;
        padding: 6px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    .risk-low {
        display: inline-block;
        background: linear-gradient(135deg, #27ae60, #1e8449);
        color: #fff;
        padding: 6px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }

    /* ---------- Warning Badges ---------- */
    .warn-critical { color: #e74c3c; font-weight: 700; }
    .warn-elevated { color: #e67e22; font-weight: 700; }
    .warn-watch    { color: #f1c40f; font-weight: 700; }
    .warn-normal   { color: #2ecc71; font-weight: 700; }

    /* ---------- SHAP Bar Labels ---------- */
    .shap-up   { color: #e74c3c; font-weight: 600; margin-right: 12px; }
    .shap-down { color: #27ae60; font-weight: 600; margin-right: 12px; }

    /* ---------- Metric Tiles ---------- */
    .stat-number {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0.2rem 0;
    }
    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
    }

    /* ---------- Recommendation list ---------- */
    .rec-item {
        background: #ffffff08;
        border-left: 3px solid #38bdf8;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }

    /* ---------- Disclaimer ---------- */
    .disclaimer {
        background: #1a1a2e;
        border-left: 4px solid #f1c40f;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        font-size: 0.82rem;
        color: #ccc;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Finance ML Models (Prophet, XGBoost, SHAP)...")
def get_finance_service():
    return FinanceService(str(FINANCE_DIR))

@st.cache_resource(show_spinner="Initializing GenAI & RAG Engine...")
def get_rag_engine():
    return DecisionIQRAGEngine()

finance_svc = get_finance_service()
rag_engine = get_rag_engine()

# ---------------------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>DecisionIQ — Business Intelligence Platform</h1>
    <p>Unified SME decision engine combining <strong>Prophet Cashflow Forecasting</strong>, 
    <strong>XGBoost Risk & SHAP Explainability</strong>, <strong>Security Threat Intelligence</strong>, 
    and <strong>GenAI Narrative Synthesis</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: Company Selection & System Health
# ---------------------------------------------------------------------------
csv_path = FINANCE_DIR / "demo_sme_companies.csv"
if not csv_path.exists():
    st.error(f"❌ Demo CSV not found at `{csv_path}`.")
    st.stop()

df_companies = pd.read_csv(csv_path)

with st.sidebar:
    st.header("🏢 SME Enterprise Selector")
    company_name = st.selectbox("Active Company", df_companies["company"].tolist())
    row = df_companies[df_companies["company"] == company_name].iloc[0]

    st.markdown("---")
    st.subheader("Financial Profile")
    st.markdown(f"**Sector:** `{row['sector'].title()}`")
    st.markdown(f"**Workforce:** `{row['employees']}` employees")
    st.markdown(f"**Annual Revenue:** `${row['revenue']:,.0f}`")
    st.markdown(f"**Monthly OPEX:** `${row['opex']:,.0f}`")
    st.markdown(f"**AR Days:** `{row['ar_days']}` days")
    st.markdown(f"**Inventory Days:** `{row['inventory_days']}` days")
    st.markdown(f"**Loan Balance:** `${row['loan']:,.0f}`")

    st.markdown("---")
    st.subheader("⚙️ System Status")
    st.success("● Finance Service (Port 8001)")
    st.success("● Security Agent (Isolation Forest + NLP)")
    st.success("● GenAI Explainer & RAG")

# ---------------------------------------------------------------------------
# Initialize Session State for Interactive Security & What-If
# ---------------------------------------------------------------------------
if "sec_login_hour" not in st.session_state:
    st.session_state.sec_login_hour = 10.0
    st.session_state.sec_failed_attempts = 0.0
    st.session_state.sec_new_device = 0
    st.session_state.sec_login_freq = 4.0
    st.session_state.sec_location_diff = 0

# Run Security Anomaly Detection
login_sample = {
    "login_hour": st.session_state.sec_login_hour,
    "failed_attempts": st.session_state.sec_failed_attempts,
    "new_device": int(st.session_state.sec_new_device),
    "login_frequency": st.session_state.sec_login_freq,
    "location_difference": int(st.session_state.sec_location_diff),
}
security_anomaly_res = predict_login(login_sample)
sec_threat_score = security_anomaly_res["anomaly_score"]

# ---------------------------------------------------------------------------
# Predict Financial Cash-Stress Risk & SHAP
# ---------------------------------------------------------------------------
fin_req = RiskPredictRequest(
    sector=str(row["sector"]),
    employees=int(row["employees"]),
    revenue_usd=float(row["revenue"]),
    opex_usd=float(row["opex"]),
    accounts_receivable_days=float(row["ar_days"]),
    inventory_days=float(row["inventory_days"]),
    loan_balance_usd=float(row["loan"]),
    owner_injections_usd=0.0,
)

fin_res = finance_svc.predict_risk(fin_req)
fin_risk_prob = fin_res.risk_probability

# ---------------------------------------------------------------------------
# Composite Business Health Calculation (70% Finance + 30% Security)
# ---------------------------------------------------------------------------
composite_res = calculate_composite_score(
    finance_score=fin_risk_prob,
    security_score=sec_threat_score,
    w_finance=0.70,
    w_security=0.30
)
comp_score = composite_res["composite_score"]

# Fetch Forecast for Early Warning calculation
try:
    summary = finance_svc.get_summary()
    historical = summary.forecast.historical
    future = summary.forecast.future
    fut_dicts = [f.model_dump() if hasattr(f, "model_dump") else f for f in future]
except Exception:
    historical, future, fut_dicts = [], [], []

early_warning = generate_early_warning(
    forecast_future=fut_dicts,
    cash_stress_probability=fin_risk_prob,
    distress_probability=None
)

# ---------------------------------------------------------------------------
# Generate GenAI Narration & Index Context for RAG
# ---------------------------------------------------------------------------
shap_dicts = [
    {"feature": f.feature, "magnitude": f.magnitude, "direction": f.direction}
    for f in fin_res.shap_features
]

company_profile_dict = {
    "company": company_name,
    "sector": str(row["sector"]),
    "employees": int(row["employees"]),
    "revenue": float(row["revenue"]),
    "opex": float(row["opex"]),
    "ar_days": float(row["ar_days"]),
    "inventory_days": float(row["inventory_days"]),
    "loan": float(row["loan"])
}

narration_data = generate_risk_narration(
    company_name=company_name,
    composite_score=comp_score,
    finance_score=fin_risk_prob,
    security_score=sec_threat_score,
    shap_features=shap_dicts,
    warnings=early_warning.get("active_warnings", []),
    company_profile=company_profile_dict
)

# Index in RAG engine
rag_engine.index_company_context(
    company_profile=company_profile_dict,
    finance_data={
        "risk_probability": fin_risk_prob,
        "composite_score": comp_score,
        "forecast_trend": "stable 90-day trajectory",
        "shap_features": shap_dicts
    },
    security_data={
        "security_score": sec_threat_score,
        "anomaly_status": "anomalous login" if security_anomaly_res["is_anomaly"] else "normal authentication",
        "phishing_status": "continuous inbox surveillance active"
    },
    warnings=early_warning.get("active_warnings", [])
)

# ---------------------------------------------------------------------------
# 1. EXECUTIVE COMPOSITE HEALTH & GENAI NARRATION CARD
# ---------------------------------------------------------------------------
comp_pct = comp_score * 100
badge_comp_cls = "risk-high" if comp_pct >= 60 else "risk-medium" if comp_pct >= 35 else "risk-low"
badge_comp_lbl = "CRITICAL VULNERABILITY" if comp_pct >= 60 else "MODERATE VULNERABILITY" if comp_pct >= 35 else "HEALTHY SOLVENCY"

st.markdown(f"""
<div class="card-accent">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <span class="stat-label">Composite Business Health Index</span>
            <div class="stat-number">{comp_pct:.1f}%</div>
            <span class="{badge_comp_cls}">{badge_comp_lbl}</span>
            <span style="font-size:0.85rem; color:#94a3b8; margin-left: 12px;">Weighted: 70% Financial Stress + 30% Security Threat</span>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.85rem; color:#38bdf8;">DecisionIQ GenAI Synthesizer</div>
            <div style="font-size: 0.78rem; color:#94a3b8;">Source: {narration_data.get('source', 'DecisionIQ Engine')}</div>
        </div>
    </div>
    <div style="margin-top: 1.2rem; font-size: 1rem; line-height: 1.6; color: #f1f5f9;">
        <strong>Executive Summary:</strong> {narration_data.get('executive_summary', '')}
    </div>
    <div style="margin-top: 0.8rem; font-size: 0.95rem; color: #cbd5e1;">
        <strong>Key Risk Analysis:</strong> {narration_data.get('financial_analysis', '')}
    </div>
    <div style="margin-top: 1rem;">
        <span style="font-size: 0.88rem; text-transform: uppercase; color: #38bdf8; font-weight: 600;">Strategic Action Plan:</span>
        <div style="margin-top: 0.4rem;">
            {''.join([f'<div class="rec-item">✦ {r}</div>' for r in narration_data.get('recommendations', [])])}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TABS: Full Navigation according to Group 3 Architecture
# ---------------------------------------------------------------------------
tab_fin, tab_sec, tab_whatif, tab_chat = st.tabs([
    "📊 Financial Intelligence",
    "Cybersecurity Agent",
    "🔄 What-If Sandbox",
    "Business Advisor (RAG Chat)"
])

# ===========================================================================
# TAB 1: FINANCIAL INTELLIGENCE (Exact layout & widgets from Finance branch)
# ===========================================================================
with tab_fin:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader(f"{company_name} — Financial Stress")
        st.markdown("#### Cash-Flow Stress Probability")
        fin_pct = fin_risk_prob * 100
        badge_cls = "risk-high" if fin_res.risk_class == 1 else "risk-low"
        badge_lbl = "HIGH RISK" if fin_res.risk_class == 1 else "LOW RISK"
        
        st.markdown(f"<h1 style='margin:0; font-size: 2.8rem;'>{fin_pct:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown(f'<span class="{badge_cls}">{badge_lbl}</span>', unsafe_allow_html=True)
        st.caption("Predicted cash-flow stress probability from the trained XGBoost model.")

    with col_right:
        st.markdown("#### 🔍 Why is the risk at this level? (SHAP)")
        shap_features = fin_res.shap_features
        if shap_features:
            names = [f.feature.replace("_", " ").title() for f in shap_features]
            magnitudes = [f.magnitude for f in shap_features]
            colors = ["#e74c3c" if f.direction == "increases_risk" else "#27ae60" for f in shap_features]

            fig_shap, ax_shap = plt.subplots(figsize=(6, max(2.2, 0.55 * len(names))))
            fig_shap.patch.set_facecolor("#0e1117")
            ax_shap.set_facecolor("#0e1117")

            y_pos = np.arange(len(names))
            ax_shap.barh(y_pos, magnitudes, color=colors, height=0.55, edgecolor="none")
            ax_shap.set_yticks(y_pos)
            ax_shap.set_yticklabels(names, fontsize=10, color="#ddd")
            ax_shap.invert_yaxis()
            ax_shap.set_xlabel("Impact on prediction", fontsize=10, color="#aaa")
            ax_shap.tick_params(axis="x", colors="#aaa", labelsize=9)
            ax_shap.spines["top"].set_visible(False)
            ax_shap.spines["right"].set_visible(False)
            ax_shap.spines["bottom"].set_color("#444")
            ax_shap.spines["left"].set_color("#444")
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)

            legend_md = "  ".join(
                f'<span class="shap-{"up" if f.direction == "increases_risk" else "down"}">'
                f'{"↑" if f.direction == "increases_risk" else "↓"} {f.feature.replace("_"," ").title()}</span>'
                for f in shap_features
            )
            st.markdown(legend_md, unsafe_allow_html=True)
        else:
            st.info("No SHAP explanation available for this input.")

    st.markdown("---")

    # 90-Day Revenue Forecast Graph (Prophet)
    st.subheader("📈 90-Day Revenue Forecast (Prophet)")
    if not historical and not future:
        st.warning("No forecast data available. Ensure Prophet model is trained.")
    else:
        def items_to_df(items):
            rows = [item.model_dump() if hasattr(item, "model_dump") else item for item in items]
            d = pd.DataFrame(rows)
            if not d.empty:
                d["ds"] = pd.to_datetime(d["ds"])
            return d

        df_hist = items_to_df(historical)
        df_fut = items_to_df(future)

        fig_p, ax_p = plt.subplots(figsize=(14, 5))
        fig_p.patch.set_facecolor("#0e1117")
        ax_p.set_facecolor("#0e1117")

        if not df_hist.empty:
            ax_p.plot(df_hist["ds"], df_hist["yhat"], color="#3498db", linewidth=1.2, label="Historical Revenue")

        if not df_fut.empty:
            ax_p.plot(df_fut["ds"], df_fut["yhat"], color="#e67e22", linewidth=2, label="Forecast Revenue")
            if "yhat_lower" in df_fut.columns and "yhat_upper" in df_fut.columns:
                ax_p.fill_between(
                    df_fut["ds"],
                    df_fut["yhat_lower"],
                    df_fut["yhat_upper"],
                    color="#e67e22",
                    alpha=0.15,
                    label="Confidence Interval",
                )

        if not df_hist.empty and not df_fut.empty:
            boundary = df_hist["ds"].max()
            ax_p.axvline(boundary, color="#95a5a6", linestyle="--", linewidth=1, alpha=0.7)
            ax_p.text(
                boundary, ax_p.get_ylim()[1] * 0.95,
                "  Forecast →", color="#95a5a6", fontsize=9, va="top",
            )

        ax_p.set_title("90-Day Revenue Forecast (Prophet)", fontsize=14, color="#eee", pad=12)
        ax_p.set_xlabel("Date", fontsize=11, color="#aaa")
        ax_p.set_ylabel("Revenue ($)", fontsize=11, color="#aaa")
        ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_p.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        fig_p.autofmt_xdate(rotation=30)
        ax_p.tick_params(colors="#aaa", labelsize=9)
        ax_p.legend(loc="upper left", fontsize=9, framealpha=0.3, labelcolor="#ddd")
        for spine in ax_p.spines.values():
            spine.set_color("#333")
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close(fig_p)

        st.caption(f"Forecast horizon: **90 days** · Historical data points: **{len(df_hist)}** · Forecast points: **{len(df_fut)}**")

        with st.expander("📋 View Forecast Data Table"):
            if not df_fut.empty:
                display_df = df_fut[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                display_df.columns = ["Date", "Forecast Revenue ($)", "Lower Bound ($)", "Upper Bound ($)"]
                display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Early Warning Signals
    st.subheader("⚠️ Rule-Based Early Warning Signals")
    sig = early_warning.get("overall_financial_signal", "normal")
    sig_cls = {
        "critical": "warn-critical",
        "elevated": "warn-elevated",
        "watch": "warn-watch",
        "normal": "warn-normal",
    }.get(sig, "warn-normal")

    st.markdown(f'Overall Signal: <span class="{sig_cls}">{sig.upper()}</span>', unsafe_allow_html=True)
    st.markdown(f"**Diagnostic Message:** {early_warning.get('overall_message', '')}")

    if early_warning.get("active_warnings"):
        for w in early_warning["active_warnings"]:
            st.markdown(f"- `{w}`")

# ===========================================================================
# TAB 2: CYBERSECURITY INTELLIGENCE (Isolation Forest + Phishing NLP)
# ===========================================================================
with tab_sec:
    st.subheader("Cybersecurity Agent Overview")
    st.markdown("Dual-engine security monitoring inspecting **authentication login behavior** and **inbound messaging text**.")

    sec_col1, sec_col2 = st.columns(2, gap="large")

    # Engine A: Isolation Forest Login Anomaly
    with sec_col1:
        st.markdown("#### 1. Login Anomaly Detector (Isolation Forest)")
        st.caption("Trained on unsupervised enterprise authentication telemetry.")

        lh = st.slider("Login Hour (24h format)", 0.0, 23.0, float(st.session_state.sec_login_hour), 1.0)
        fa = st.slider("Failed Attempts Prior to Login", 0.0, 15.0, float(st.session_state.sec_failed_attempts), 1.0)
        lf = st.slider("Login Frequency (attempts/hour)", 0.5, 20.0, float(st.session_state.sec_login_freq), 0.5)
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            nd = st.checkbox("New / Unrecognized Device", value=bool(st.session_state.sec_new_device))
        with c_sub2:
            ld = st.checkbox("Novel Geo-Location Difference", value=bool(st.session_state.sec_location_diff))

        st.session_state.sec_login_hour = lh
        st.session_state.sec_failed_attempts = fa
        st.session_state.sec_login_freq = lf
        st.session_state.sec_new_device = int(nd)
        st.session_state.sec_location_diff = int(ld)

        # Dynamic real-time calculation
        curr_login = {
            "login_hour": lh,
            "failed_attempts": fa,
            "new_device": int(nd),
            "login_frequency": lf,
            "location_difference": int(ld)
        }
        res_login = predict_login(curr_login)

        a_badge = "risk-high" if res_login["is_anomaly"] else "risk-low"
        a_lbl = "ANOMALY DETECTED" if res_login["is_anomaly"] else "NORMAL AUTHENTICATION"

        st.markdown(f"""
        <div class="card" style="margin-top: 1rem;">
            <div style="font-size: 0.85rem; color: #94a3b8;">ANOMALY THREAT SCORE</div>
            <div style="font-size: 2.2rem; font-weight: 700;">{res_login['risk_score']:.1f}%</div>
            <span class="{a_badge}">{a_lbl}</span>
            <div style="font-size: 0.8rem; color: #aaa; margin-top: 8px;">
                Isolation Forest decision margin: <code>{res_login['raw_decision']:.4f}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Engine B: Phishing NLP Classifier
    with sec_col2:
        st.markdown("#### 2. Email & Message Phishing Scanner")
        st.caption("Trained on CEAS08 corpus for phishing and credential-harvesting classification.")

        default_phish_text = "URGENT: Your company bank account is temporarily restricted. Click here to verify credentials."
        sample_email = st.text_area("Inbound Message Body", value=default_phish_text, height=120)

        if sample_email:
            p_res = detect_phishing(sample_email)
            p_badge = "risk-high" if p_res["is_phishing"] else "risk-low"
            p_lbl = "PHISHING DETECTED" if p_res["is_phishing"] else "LEGITIMATE COMMUNICATION"

            st.markdown(f"""
            <div class="card" style="margin-top: 0.6rem;">
                <div style="font-size: 0.85rem; color: #94a3b8;">PHISHING PROBABILITY</div>
                <div style="font-size: 2.2rem; font-weight: 700;">{p_res['risk_score']}%</div>
                <span class="{p_badge}">{p_lbl}</span>
                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 10px;">
                    Model Confidence: <strong>{p_res['confidence'] * 100:.1f}%</strong> · Engine: <code>{p_res.get('model_type', 'NLP')}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================================
# TAB 3: WHAT-IF SCENARIO SANDBOX (Exact UI from Finance branch, extended)
# ===========================================================================
with tab_whatif:
    st.subheader("🔄 What-If Scenario Simulation")
    st.markdown("Adjust one or more operational parameters below. The **same XGBoost model** re-evaluates risk in-memory without persistent DB mutations.")

    wi_col1, wi_col2 = st.columns(2)

    with wi_col1:
        st.markdown("**Financial Levers**")
        new_rev = st.slider(
            "Revenue ($)",
            min_value=0.0,
            max_value=float(row["revenue"]) * 3,
            value=float(row["revenue"]),
            step=1000.0,
            format="$%,.0f",
            key="wi_revenue"
        )
        new_op = st.slider(
            "OPEX ($)",
            min_value=0.0,
            max_value=float(row["opex"]) * 3,
            value=float(row["opex"]),
            step=1000.0,
            format="$%,.0f",
            key="wi_opex"
        )
        new_ln = st.slider(
            "Loan Balance ($)",
            min_value=0.0,
            max_value=float(row["loan"]) * 3,
            value=float(row["loan"]),
            step=1000.0,
            format="$%,.0f",
            key="wi_loan"
        )

    with wi_col2:
        st.markdown("**Working Capital & Cyber Levers**")
        new_ar_val = st.slider(
            "AR Days",
            min_value=0.0,
            max_value=180.0,
            value=float(row["ar_days"]),
            step=5.0,
            key="wi_ar"
        )
        new_inv_val = st.slider(
            "Inventory Days",
            min_value=0.0,
            max_value=180.0,
            value=float(row["inventory_days"]),
            step=5.0,
            key="wi_inv"
        )
        new_sec_sim = st.slider(
            "Simulated Cyber Threat Level",
            min_value=0.0,
            max_value=1.0,
            value=float(sec_threat_score),
            step=0.05,
            format="%.2f",
            key="wi_sec"
        )

    if st.button("▶  Run Multi-Variable What-If Scenario", type="primary"):
        try:
            scenario_req = RiskPredictRequest(
                sector=str(row["sector"]),
                employees=int(row["employees"]),
                revenue_usd=new_rev,
                opex_usd=new_op,
                accounts_receivable_days=new_ar_val,
                inventory_days=new_inv_val,
                loan_balance_usd=new_ln,
                owner_injections_usd=0.0,
            )
            scenario_res = finance_svc.predict_risk(scenario_req)

            scen_fin_prob = scenario_res.risk_probability
            curr_fin_prob = fin_risk_prob

            # Recalculate composite
            scen_comp = calculate_composite_score(scen_fin_prob, new_sec_sim)

            curr_pct = curr_fin_prob * 100
            scen_pct = scen_fin_prob * 100
            diff = scen_pct - curr_pct

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline Financial Risk", f"{curr_pct:.1f}%")
            m2.metric("Scenario Financial Risk", f"{scen_pct:.1f}%", delta=f"{diff:+.1f} pp", delta_color="inverse")
            m3.metric("New Composite Health Risk", f"{scen_comp['composite_score'] * 100:.1f}%")

            if diff > 0.5:
                st.warning("📈 **Financial vulnerability increases** under this simulation.")
            elif diff < -0.5:
                st.success("📉 **Financial vulnerability decreases** under this simulation.")
            else:
                st.info("➡️ **Risk remains approximately unchanged.**")

            # Scenario SHAP
            with st.expander("🔍 Scenario SHAP Explanations"):
                for feat in scenario_res.shap_features:
                    arrow = "↑" if feat.direction == "increases_risk" else "↓"
                    col = "red" if feat.direction == "increases_risk" else "green"
                    st.markdown(f":{col}[{arrow} **{feat.feature.replace('_',' ').title()}**] — impact magnitude `{feat.magnitude:.4f}`")

            # Live GenAI Narration update
            scen_shap_dicts = [
                {"feature": f.feature, "magnitude": f.magnitude, "direction": f.direction}
                for f in scenario_res.shap_features
            ]
            scen_narr = generate_risk_narration(
                company_name=company_name,
                composite_score=scen_comp["composite_score"],
                finance_score=scen_fin_prob,
                security_score=new_sec_sim,
                shap_features=scen_shap_dicts,
                warnings=early_warning.get("active_warnings", []),
                company_profile={
                    "sector": str(row["sector"]),
                    "revenue": new_rev,
                    "opex": new_op,
                    "ar_days": new_ar_val,
                    "loan": new_ln
                }
            )

            st.markdown(f"""
            <div class="card" style="margin-top: 1rem; border-color: #38bdf8;">
                <div style="font-size: 0.85rem; color: #38bdf8; font-weight: 600;">SIMULATED GENAI STRATEGIC ASSESSMENT</div>
                <p style="margin-top: 0.5rem; line-height: 1.5;">{scen_narr.get('executive_summary', '')}</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Scenario simulation error: {e}")

# ===========================================================================
# TAB 4: BUSINESS ADVISOR (RAG Chat with Business Data)
# ===========================================================================
with tab_chat:
    st.subheader("AI Business Advisor & Data Q&A")
    st.markdown("Ask contextual questions regarding **financial metrics, 90-day forecasts, security exposures, or operational levers**.")

    user_query = st.text_input(
        "Ask a question regarding the business:",
        placeholder="e.g. Why is our cash-flow stress high? How can we reduce our composite risk score?"
    )

    if st.button("Send Query", type="primary") or user_query:
        if user_query:
            with st.spinner("Retrieving business chunks and generating grounded guidance..."):
                chat_res = rag_engine.answer_query(user_query)
                st.markdown(f"""
                <div class="card-accent" style="margin-top: 1rem;">
                    <div style="font-size: 0.85rem; color: #38bdf8; margin-bottom: 0.5rem;">
                        <strong>DecisionIQ Advisor Response</strong> (Source: {chat_res.get('source', 'RAG')})
                    </div>
                    <div style="font-size: 1rem; line-height: 1.6; color: #f8fafc;">
                        {chat_res.get('answer', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("🔍 Ground-Truth Retrieved Context"):
                    for c in chat_res.get("retrieved_context", []):
                        st.markdown(f"- {c}")

# ---------------------------------------------------------------------------
# Disclaimer Footer (Strictly preserved from Finance branch)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer">
    <strong>Note:</strong> These risk scores and forecasts are AI-assisted estimates meant to help guide your everyday business planning. They should be used alongside your team's real-world judgment and do not replace professional financial or security advice.
</div>
""", unsafe_allow_html=True)
