"""
DecisionIQ Microservices Runner
===============================
Starts API Gateway and all microservices concurrently using uvicorn.
- API Gateway:      http://127.0.0.1:8000
- Finance Service:  http://127.0.0.1:8001
- Security Service: http://127.0.0.1:8002
- GenAI Service:    http://127.0.0.1:8003
"""

import sys
import subprocess
import time
import os

python_exe = sys.executable

print("========================================================")
print("🚀 Launching DecisionIQ Microservices Architecture")
print("========================================================")

processes = []

try:
    print("Starting Finance Service on port 8001...")
    p_fin = subprocess.Popen([python_exe, "-m", "uvicorn", "app.main:app", "--port", "8001"], cwd="decisioniq-finance-service")
    processes.append(p_fin)

    print("Starting Security Service on port 8002...")
    p_sec = subprocess.Popen([python_exe, "-m", "uvicorn", "app.main:app", "--port", "8002"], cwd="decisioniq-security-service")
    processes.append(p_sec)

    print("Starting GenAI Service on port 8003...")
    p_gen = subprocess.Popen([python_exe, "-m", "uvicorn", "app.main:app", "--port", "8003"], cwd="decisioniq-genai-service")
    processes.append(p_gen)

    time.sleep(2)

    print("Starting API Gateway on port 8000...")
    p_gw = subprocess.Popen([python_exe, "-m", "uvicorn", "main:app", "--port", "8000"], cwd="decisioniq-gateway")
    processes.append(p_gw)

    print("\n✅ All services running! Check Gateway Docs at http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to terminate all services.\n")

    for p in processes:
        p.wait()

except KeyboardInterrupt:
    print("\nStopping all services...")
    for p in processes:
        p.terminate()
    print("All services stopped.")
