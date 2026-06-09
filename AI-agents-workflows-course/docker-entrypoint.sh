#!/usr/bin/env bash
set -e

echo ""
echo "============================================================"
echo " AI Prompt Simulator - host URLs"
echo "============================================================"
echo " Gradio UI:  ${PUBLIC_GRADIO_URL:-http://localhost:7860/}"
echo " MLflow UI:  ${PUBLIC_MLFLOW_URL:-http://127.0.0.1:5002/}"
echo ""
echo " Internal container bindings:"
echo " Gradio:     http://${GRADIO_SERVER_NAME:-0.0.0.0}:${GRADIO_SERVER_PORT:-7860}"
echo " MLflow:     http://${MLFLOW_UI_HOST:-0.0.0.0}:${MLFLOW_UI_PORT:-5001}"
echo "============================================================"
echo ""

exec python "${APP_SCRIPT:-ai-prompt-sim.py}"
