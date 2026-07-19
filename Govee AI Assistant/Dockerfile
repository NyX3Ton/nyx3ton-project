# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Govee AI Assistant — GPU image for unsloth/gemma-4-E4B-it
#
# Requirements on the HOST (not inside the container):
#   • NVIDIA driver >= 560.28  (minimum for CUDA 12.6 runtime)
#     Check: nvidia-smi | grep "Driver Version"
#   • NVIDIA Container Toolkit  https://docs.nvidia.com/datacenter/cloud-native/
#
# WHY CUDA 12.6 (not 12.4):
#   requirements-agent.txt pins torch>=2.7. PyTorch's cu124 wheel index only
#   carries torch up to 2.6.0. Torch 2.7+ is built against CUDA 12.6, so we
#   use the cu126 wheel index and the matching CUDA 12.6 base image.
#
# WHY THE OPENVINO STACK IS FILTERED OUT (see Phase 2):
#   openvino + optimum-intel are only used by the OpenVINO backend, which does
#   not exist in this GPU container (that code path was removed from agent.py).
#   They are never imported at runtime here. They also actively break the build:
#   optimum-intel pins optimum~=2.2, which pins transformers back to 5.0.0 — a
#   release that predates the `gemma4` architecture, producing at runtime:
#     ValueError: model type `gemma4` ... Transformers does not recognize ...
#   Removing them frees transformers to install a gemma4-capable version.
#
# Build:  docker compose build
# Run:    docker compose up
#
# First run downloads ~16 GB of model weights into the hf_cache volume.
# Subsequent runs reuse the cache — startup takes ~30 s instead of ~20 min.
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

# ── Environment ────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    GOVEE_LLM_MODEL=unsloth/gemma-4-E4B-it \
    QUANTIZE_BITS=4

# ── System packages ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
# Phase 1: torch + torchvision from the pytorch cu126 index, versions frozen
# into a constraints file so nothing downgrades them to a CPU-only wheel later
# (which would cause "operator torchvision::nms does not exist").
RUN pip install \
        torch \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu126 \
    && pip freeze | grep -E "^(torch|torchvision)==" > /tmp/torch-pins.txt \
    && echo "=== pinned torch versions ===" && cat /tmp/torch-pins.txt

# bitsandbytes + accelerate (for 4/8-bit quantization), constrained
RUN pip install \
        "bitsandbytes>=0.43.0" \
        "accelerate>=0.30.0" \
        -c /tmp/torch-pins.txt

# Phase 2: project requirements, with the OpenVINO stack filtered out.
# grep -viE drops any line starting with openvino* or optimum-intel from
# requirements-agent.txt (case-insensitive), writing a CUDA-only variant.
COPY requirements.txt requirements-agent.txt requirements-app.txt ./
RUN grep -viE '^(openvino|optimum-intel)' requirements-agent.txt > requirements-agent.cuda.txt \
    && echo "=== requirements-agent (CUDA, OpenVINO filtered out) ===" \
    && cat requirements-agent.cuda.txt

RUN pip install \
        -r requirements.txt \
        -r requirements-agent.cuda.txt \
        -r requirements-app.txt \
        -c /tmp/torch-pins.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126

# gemma-4 is brand new. After filtering optimum-intel, transformers installs at
# its latest release. If that release still predates gemma4 support, fall back
# to transformers main — the exact fix the HF error message recommends.
# github.com / codeload.github.com are reachable from the build network.
RUN python -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as m; import sys; sys.exit(0 if 'gemma4' in m else 1)" \
    && echo "transformers already recognizes gemma4" \
    || (echo "gemma4 missing from release transformers — installing from source" \
        && pip install --upgrade "git+https://github.com/huggingface/transformers.git" -c /tmp/torch-pins.txt)

# ── Sanity check ───────────────────────────────────────────────────────────
# Fail the build loudly (not at container runtime) if torch is CPU-only OR if
# transformers still can't recognize gemma4. Single line: a Dockerfile RUN only
# continues across newlines when the line ends with a backslash.
RUN python -c "import torch, torchvision; from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as m; print('torch', torch.__version__, '| torchvision', torchvision.__version__); assert '+cu' in torch.__version__, 'ERROR: CPU-only torch installed (%s)' % torch.__version__; assert 'gemma4' in m, 'ERROR: transformers does not recognize gemma4'; print('OK: CUDA torch + gemma4 support confirmed')"

# ── Application code ────────────────────────────────────────────────────────
COPY . .
RUN mkdir -p /app/.cache/huggingface

# ── Runtime ─────────────────────────────────────────────────────────────────
EXPOSE 7860

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=180s \
    --retries=5 \
    CMD curl -sf http://localhost:7860/ > /dev/null || exit 1

CMD ["python", "app.py"]
