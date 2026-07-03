ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG INSTALL_QWEN_FAST=0
ARG RUNTIME_PROFILE=cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV GRADIO_ANALYTICS_ENABLED=False
ENV HF_HOME=/cache/huggingface
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    ninja-build \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
RUN python -m pip install --upgrade pip setuptools wheel packaging

RUN if [ "${RUNTIME_PROFILE}" = "openvino" ]; then \
        pip install torch torchvision; \
    else \
        pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision; \
    fi

COPY requirements-docker.txt .
COPY requirements.txt .
COPY requirements-agent.txt .
COPY requirements-app.txt .
COPY requirements-openvino.txt .
RUN if [ "${RUNTIME_PROFILE}" = "openvino" ]; then \
        pip install -r requirements-openvino.txt; \
    else \
        pip install -r requirements-docker.txt && \
        pip install --no-deps optimum==1.27.0 optimum-intel==1.27.0 optimum-onnx==0.1.0; \
    fi

RUN if [ "${RUNTIME_PROFILE}" != "openvino" ] && [ "${INSTALL_QWEN_FAST}" = "1" ]; then \
        pip install "flash-linear-attention[cuda]" && \
        pip install --no-build-isolation causal-conv1d; \
    fi

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
