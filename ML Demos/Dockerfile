FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# libgomp1 is required by XGBoost / scikit-learn (OpenMP runtime).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install the CPU build of PyTorch first so the large CUDA wheels aren't pulled in.
# For a GPU image, use an nvidia/cuda base and the matching cuXXX torch index instead.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Only the entrypoint is baked into the image. app.py and the datasets are
# bind-mounted at runtime (see docker-compose.yml), so app.py stays untouched.
COPY docker_entrypoint.py ./

EXPOSE 7860 8080

CMD ["python", "docker_entrypoint.py"]
