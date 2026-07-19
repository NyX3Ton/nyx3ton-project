#config.py
#
# Single source of truth for every environment variable the assistant reads.
# Other modules do `from . import config` and reference `config.NAME` rather
# than calling os.getenv(...) themselves, so every setting and its default
# lives in exactly one place.

from __future__ import annotations

import os, re
from dotenv import load_dotenv

# Must run before anything below reads os.environ, and before app.py's
# `import gradio` (GRADIO_ANALYTICS_ENABLED has to be set pre-import) -
# this module is deliberately the first project import in every entry point.
load_dotenv()
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# ── Govee API ────────────────────────────────────────────────────────────
GOVEE_API_KEY = os.environ.get("GOVEE_API_KEY")

# ── LLM backend (agent.py) ──────────────────────────────────────────────

# GOVEE_MODEL_ID is the legacy name (pre Qwen->Gemma migration). It's honored
# as a fallback so existing .env files that still use it aren't silently
# ignored - otherwise the app would quietly load the default model instead of
# the one the user configured. Prefer GOVEE_LLM_MODEL going forward.
GOVEE_LLM_MODEL = (os.getenv("GOVEE_LLM_MODEL") or os.getenv("GOVEE_MODEL_ID") or "unsloth/gemma-4-E4B-it")

# Selects the model-loading implementation.  `transformers` is the current
# production backend; `nemo` is accepted as an explicit future-facing option
# so configuration can be shared before the NeMo backend is introduced.
GOVEE_MODEL_RUNTIME = os.getenv("GOVEE_MODEL_RUNTIME", "transformers").strip().lower()
if GOVEE_MODEL_RUNTIME not in {"transformers", "nemo"}:
    raise ValueError("GOVEE_MODEL_RUNTIME must be either 'transformers' or 'nemo'")

# Optional per-GPU cap passed to Transformers/Accelerate's `max_memory` map,
# e.g. "10GiB" or "12288MiB". Empty lets the loader use available VRAM.
_max_gpu_memory = os.getenv("GOVEE_MAX_GPU_MEMORY", "").strip()
if _max_gpu_memory and not re.fullmatch(r"[1-9]\d*(?:MiB|GiB|MB|GB)", _max_gpu_memory):
    raise ValueError("GOVEE_MAX_GPU_MEMORY must look like '10GiB' or '12288MiB'")
GOVEE_MAX_GPU_MEMORY: str | None = _max_gpu_memory or None

# Optional secondary model tried only if the primary fails to load (e.g. the
# primary is too large for the available VRAM). Empty = no fallback.
GOVEE_FALLBACK_MODEL = os.getenv("GOVEE_FALLBACK_MODEL_ID", "").strip()
QUANTIZE_BITS = int(os.getenv("QUANTIZE_BITS", "0"))

# Agent architecture: "single" = the built-in tool-calling loop (GoveeAgent);
# "workflow" = the LlamaIndex multi-agent AgentWorkflow (orchestrator.py).
# Single is the default because multi-agent ReAct is less reliable on a small
# local model; the workflow is opt-in. See orchestrator.py.
GOVEE_AGENT_MODE = os.getenv("GOVEE_AGENT_MODE", "workflow").strip().lower()

# Optional bounded writer--critic refinement.  The critic shares the already
# loaded local model and can only review/rewrite the final natural-language
# answer; it cannot call device or information tools.  This keeps a critique
# pass from repeating a side-effecting action such as turning a light on.
GOVEE_CRITIQUE_ENABLED = os.getenv("GOVEE_CRITIQUE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
GOVEE_CRITIQUE_MAX_PASSES = max(0, int(os.getenv("GOVEE_CRITIQUE_MAX_PASSES", "1")))

# ── Semantic device/scene matching (semantic_match.py) ──────────────────
GOVEE_EMBEDDING_MODEL = os.getenv("GOVEE_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# ── Speech-to-text (speech_to_text.py) ───────────────────────────────────
GOVEE_STT_MODEL = os.getenv("GOVEE_STT_MODEL", "openai/whisper-base")
GOVEE_STT_DEVICE = os.getenv("GOVEE_STT_DEVICE", "auto").strip().lower()
GOVEE_STT_LANGUAGE = os.getenv("GOVEE_STT_LANGUAGE", "").strip()
GOVEE_STT_SILENCE_RMS = float(os.getenv("GOVEE_STT_SILENCE_RMS", "0.002"))

# ── Weather / news / memory (weather_client.py, news_client.py, memory_store.py) ──
GOVEE_DEFAULT_LOCATION = os.getenv("GOVEE_DEFAULT_LOCATION", "").strip()
GOVEE_NEWS_FEEDS = os.getenv("GOVEE_NEWS_FEEDS", "").strip()
GOVEE_MEMORY_DB = os.getenv("GOVEE_MEMORY_DB", "./chroma_memory")

# Optional relevance floor for RAG memory recall (0-1). Empty = disabled
# (return the top matches regardless of score). The right value depends on the
# embedding model, so it ships off by default; see memory_store.MemoryStore.
_memory_cutoff = os.getenv("GOVEE_MEMORY_SIMILARITY_CUTOFF", "").strip()
GOVEE_MEMORY_SIMILARITY_CUTOFF: float | None = float(_memory_cutoff) if _memory_cutoff else None

# ── Gradio server (app.py) ───────────────────────────────────────────────
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
