@echo off
echo ========================================================
echo Starting DecisionIQ AI Business Intelligence Dashboard...
echo ========================================================
python -m streamlit run app_unified.py --server.port 8501
pause
