# main script: validator_utils.py of nyx3ton-project\Few-shots\CV Validator

from __future__ import annotations

import hashlib, json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# 1. UTILS
# -----------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    return text.strip()

def safe_json_loads(text: str, fallback: Any) -> Any:
    if not text:
        return fallback

    cleaned = strip_thinking(text).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    parsed_candidates = []

    for i, ch in enumerate(cleaned):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
            parsed_candidates.append(obj)
        except Exception:
            continue

    if parsed_candidates:
        return parsed_candidates[-1]

    return fallback

def normalize_space(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def trim_text(text: str, max_chars: int = 24000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[...SKRATENE...]\n\n" + text[-half:]

def file_ext(path: str) -> str:
    return Path(path).suffix.lower().replace(".", "")

def weighted_average(rows: List[Dict[str, Any]]) -> float:
    total_w = 0.0
    total = 0.0
    for r in rows:
        try:
            w = float(r.get("weight", 1.0))
            s = float(r.get("score", 0.0))
        except Exception:
            continue
        total_w += max(w, 0.01)
        total += max(0.0, min(100.0, s)) * max(w, 0.01)
    return round(total / total_w, 2) if total_w else 0.0

def remove_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))

def cache_key(prefix: str, text: str, model_id: str = "") -> str:
    raw = f"{prefix}||{model_id}||{normalize_space(text)}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

def parse_boolish(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = normalize_space(str(value)).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "on", "ano"}

def normalized_lookup_value(value: Any) -> str:
    return remove_diacritics(normalize_space(str(value or "")).lower())


def safe_float_value(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def dedupe_text_list(items: List[str], limit: int = 12) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = normalize_space(str(item))
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result
