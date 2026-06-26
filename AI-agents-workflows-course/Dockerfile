# Lightweight image that runs the full Agentic AI Weather Pipeline.
# The core pipeline (Prompt -> Weather -> Excel -> Email) runs in deterministic
# rule-based mode here. The optional CUDA / OpenVINO LLM backends are NOT
# installed in this image (they need a GPU or extra runtimes); see the README
# "Run with Docker" section for how to enable them.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Core runtime dependencies, plus mlflow (used both for background metric
# logging by the app and to serve the MLflow UI service in docker-compose).
# LLM backends (torch/transformers/openvino) remain opt-in.
RUN pip install --no-cache-dir \
    "gradio>=4.44" \
    "requests>=2.32" \
    "pandas>=2.2" \
    "openpyxl>=3.1" \
    "python-dotenv>=1.0" \
    "mlflow>=2.12"

# Application code. app.py is copied unchanged; .env is provided at runtime via
# docker-compose (env_file), never baked into the image.
COPY app.py ./

# Generated Excel reports land here; mount a volume to persist them on the host.
RUN mkdir -p /app/Outputs

EXPOSE 7861

CMD ["python", "app.py"]
