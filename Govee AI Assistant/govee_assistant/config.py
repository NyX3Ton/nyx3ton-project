#config.py
#
# Single source of truth for every environment variable the assistant reads.
# Other modules do `from . import config` and reference `config.NAME` rather
# than calling os.getenv(...) themselves, so every setting and its default
# lives in exactly one place.

from __future__ import annotations

import os

from dotenv import load_dotenv

# Must run before anything below reads os.environ, and before app.py's
# `import gradio` (GRADIO_ANALYTICS_ENABLED has to be set pre-import) -
# this module is deliberately the first project import in every entry point.
load_dotenv()
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# ── Govee API ────────────────────────────────────────────────────────────
GOVEE_API_KEY = os.environ.get("GOVEE_API_KEY")

# ── LLM backend (agent.py) ──────────────────────────────────────────────
GOVEE_LLM_MODEL = os.getenv("GOVEE_LLM_MODEL", "google/gemma-4-E4B-it")
QUANTIZE_BITS = int(os.getenv("QUANTIZE_BITS", "0"))

# ── Semantic device/scene matching (semantic_match.py) ──────────────────
GOVEE_EMBEDDING_MODEL = os.getenv(
    "GOVEE_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

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
