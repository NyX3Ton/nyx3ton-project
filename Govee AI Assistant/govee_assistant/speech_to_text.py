#speech_to_text.py

from __future__ import annotations

import logging, math, os, shutil, wave
from typing import Any, Callable, Optional

import numpy as np

from . import config

logger = logging.getLogger("speech_to_text")

# Whisper is multilingual with automatic language detection across ~99
# languages - a superset of the 50+ languages the semantic-matching
# embedding model (semantic_match.py) covers, so speech in any language the
# rest of the pipeline understands should transcribe fine without forcing a
# language up front. "small" balances accuracy against size/speed (~500MB,
# comparable to the multilingual embedding model); override via
# GOVEE_STT_MODEL, e.g. "openai/whisper-base" (~150MB, lighter/faster) or
# "openai/whisper-large-v3" (most accurate, slowest - and heavy on CPU).
_MODEL_NAME = config.GOVEE_STT_MODEL
_DEVICE_SETTING = config.GOVEE_STT_DEVICE
_LANGUAGE = config.GOVEE_STT_LANGUAGE
_SILENCE_RMS = config.GOVEE_STT_SILENCE_RMS
_TARGET_SR = 16000  # Whisper's expected sample rate.
_pipeline: Any = None


def _audio_path_value(audio_path: Any) -> Optional[str]:
    if isinstance(audio_path, str):
        return audio_path
    if isinstance(audio_path, dict):
        for key in ("path", "name"):
            value = audio_path.get(key)
            if isinstance(value, str):
                return value
    if isinstance(audio_path, (list, tuple)) and audio_path:
        return _audio_path_value(audio_path[0])
    return None

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

def _pipeline_device() -> int:
    if _DEVICE_SETTING in {"cpu", "-1"}:
        return -1

    import torch

    if _DEVICE_SETTING in {"cuda", "gpu", "0"}:
        if torch.cuda.is_available():
            return 0
        logger.warning("GOVEE_STT_DEVICE=%s requested, but CUDA is unavailable; using CPU", _DEVICE_SETTING)
        return -1

    if _DEVICE_SETTING != "auto":
        logger.warning("Unknown GOVEE_STT_DEVICE=%s; expected auto, cpu, or cuda", _DEVICE_SETTING)
    return 0 if torch.cuda.is_available() else -1


def _decode_wav_pcm(path: str) -> Optional[tuple[np.ndarray, int]]:
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, OSError):
        return None

    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:  # 8-bit PCM WAV is unsigned
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return None

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr

def _resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    if sr == _TARGET_SR or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        g = math.gcd(sr, _TARGET_SR)
        return resample_poly(samples, _TARGET_SR // g, sr // g).astype(np.float32)
    except Exception:
        duration = samples.shape[0] / float(sr)
        n_target = int(round(duration * _TARGET_SR))
        if n_target <= 0:
            return samples.astype(np.float32, copy=False)
        x_old = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)

def _result_to_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("text", ""))
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
    if isinstance(result, str):
        return result
    return ""

def _generate_kwargs() -> dict[str, str]:
    kwargs = {"task": "transcribe"}
    if _LANGUAGE:
        kwargs["language"] = _LANGUAGE
    return kwargs

def _default_transcribe(audio_path: str) -> str:
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline

        device = _pipeline_device()
        device_name = "cuda" if device >= 0 else "cpu"
        logger.info("Loading speech-to-text model %s on %s (first call only)...", _MODEL_NAME, device_name)
        _pipeline = pipeline("automatic-speech-recognition", model=_MODEL_NAME, device=device)

        # Whisper-via-transformers prints several advisory warnings on every
        # call (chunk_length_s "experimental", duplicate logits processors, BPE
        # clean_up). They're harmless console clutter, so drop transformers'
        # logger to error level. Note: this affects transformers advisories
        # process-wide; remove these two lines if you want those warnings back.
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

    # Decode the recording ourselves (stdlib `wave` + numpy) and hand the
    # pipeline a ready 16 kHz mono array. This sidesteps transformers' hard
    # dependency on the external ffmpeg binary - the most common reason the mic
    # "produces nothing" on a fresh machine. Anything we can't decode this way
    # falls back to the ffmpeg path if ffmpeg is present, else a clear error.
    decoded = _decode_wav_pcm(audio_path)
    duration = 0.0
    if decoded is not None:
        samples, sr = decoded
        duration = samples.size / float(sr or _TARGET_SR)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        logger.info(
                    "Decoded voice recording %s: %.2fs at %s Hz, peak=%.4f, rms=%.4f",
                    os.path.basename(audio_path), duration, sr, peak, rms,
                    )
        if samples.size and rms < _SILENCE_RMS:
            logger.warning("Voice recording is silent or very quiet; skipping transcription")
            return ""
        pipe_input: Any = {"raw": _resample_to_16k(samples, sr), "sampling_rate": _TARGET_SR}
    elif ffmpeg_available():
        logger.info("Using ffmpeg-backed decode for voice recording %s", os.path.basename(audio_path))
        pipe_input = audio_path  # let transformers decode via ffmpeg
    else:
        raise RuntimeError(
                            "Couldn't decode the recording without ffmpeg, and ffmpeg wasn't found "
                            "on PATH. Install ffmpeg (Windows: `winget install Gyan.FFmpeg`; macOS: "
                            "`brew install ffmpeg`; Linux: `apt install ffmpeg`) and restart the app."
                            )

    # Whisper only needs chunked long-form mode for recordings over its 30s
    # receptive field. Short command clips are more reliable on the normal path.
    kwargs = {"generate_kwargs": _generate_kwargs()}
    if duration > 30:
        kwargs["chunk_length_s"] = 30
    result = _pipeline(pipe_input, **kwargs)
    text = _result_to_text(result).strip()
    if text:
        logger.info("Speech-to-text transcript: %s", text)
    else:
        logger.warning("Speech-to-text returned an empty transcript for %s", os.path.basename(audio_path))
    return text

def transcribe(audio_path: Any, transcribe_fn: Optional[Callable[[str], str]] = None) -> str:
    audio_path = _audio_path_value(audio_path)
    if not audio_path:
        logger.warning("Speech-to-text received no audio file path")
        return ""

    fn = transcribe_fn or _default_transcribe
    try:
        text = fn(audio_path)
    except Exception:
        logger.exception("Speech-to-text model unavailable or failed, skipping transcription")
        return ""
    return (text or "").strip()
