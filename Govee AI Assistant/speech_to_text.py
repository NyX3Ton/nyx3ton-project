#speech_to_text.py

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger("speech_to_text")

# Whisper is multilingual with automatic language detection across ~99
# languages - a superset of the 50+ languages the semantic-matching
# embedding model (semantic_match.py) covers, so speech in any language the
# rest of the pipeline understands should transcribe fine without forcing a
# language up front. "small" balances accuracy against size/speed (~500MB,
# comparable to the multilingual embedding model); override via
# GOVEE_STT_MODEL, e.g. "openai/whisper-base" (~150MB, lighter/faster) or
# "openai/whisper-large-v3" (most accurate, slowest - and heavy on CPU).
_MODEL_NAME = os.getenv("GOVEE_STT_MODEL", "openai/whisper-small")

# The lazily-loaded transformers pipeline. Typed Any on purpose: the pipeline
# object and its return value are dynamically typed, and pinning it to Any is
# what keeps the type checker from mis-inferring the call result as a list and
# rejecting result["text"] below.
_pipeline: Any = None


def _result_to_text(result: Any) -> str:
    """Pull the transcript string out of whatever shape the ASR pipeline returns.

    For a single input the pipeline returns {"text": ...} (plus "chunks" when
    timestamps are on); for a list of inputs it returns a list of such dicts.
    Handle both, plus a bare string, so a shape change can't crash the caller.
    """
    if isinstance(result, dict):
        return str(result.get("text", ""))
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
    if isinstance(result, str):
        return result
    return ""


def _default_transcribe(audio_path: str) -> str:
    global _pipeline
    if _pipeline is None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading speech-to-text model %s (first call only)...", _MODEL_NAME)
        _pipeline = pipeline("automatic-speech-recognition", model=_MODEL_NAME, device=device)

    # chunk_length_s turns on Whisper's chunked long-form algorithm. Whisper's
    # receptive field is only 30s; without this, a recording longer than that
    # raises (or silently truncates) instead of transcribing - a common cause
    # of "the mic button produces nothing". Splitting into 30s chunks makes any
    # length work. No "language" kwarg -> Whisper auto-detects the spoken one.
    result = _pipeline(audio_path, chunk_length_s=30, generate_kwargs={"task": "transcribe"})
    return _result_to_text(result)


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