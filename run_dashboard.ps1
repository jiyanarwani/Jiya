Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting DecisionIQ AI Business Intelligence Dashboard..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
& python -m streamlit run app_unified.py --server.port 8501
