#speech_to_text.py

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger("speech_to_text")

# Whisper is multilingual with automatic language detection across ~99
# languages - a superset of the 50+ languages the semantic-matching
# embedding model (semantic_match.py) covers, so speech in any language the
# rest of the pipeline understands should transcribe fine without forcing a
# language up front. "small" balances accuracy against size/speed (~500MB,
# comparable to the multilingual embedding model); override via
# GOVEE_STT_MODEL, e.g. "openai/whisper-base" (~150MB, lighter/faster) or
# "openai/whisper-large-v3" (most accurate, slowest).
_MODEL_NAME = os.getenv("GOVEE_STT_MODEL", "openai/whisper-large-v3")
_pipeline = None


def _default_transcribe(audio_path: str) -> str:
    global _pipeline
    if _pipeline is None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading speech-to-text model %s (first call only)...", _MODEL_NAME)
        _pipeline = pipeline("automatic-speech-recognition", model=_MODEL_NAME, device=device)

    # No "language" kwarg passed -> Whisper auto-detects the spoken
    # language instead of assuming English.
    result = _pipeline(audio_path, generate_kwargs={"task": "transcribe"})
    return result["text"]


def transcribe(audio_path: Optional[str], transcribe_fn: Optional[Callable[[str], str]] = None) -> str:
    if not audio_path:
        return ""

    fn = transcribe_fn or _default_transcribe
    try:
        text = fn(audio_path)
    except Exception:
        logger.exception("Speech-to-text model unavailable or failed, skipping transcription")
        return ""
    return (text or "").strip()
