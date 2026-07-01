#semantic_match.py

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence
import numpy as np

logger = logging.getLogger("semantic_match")

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None 

def _default_encode(texts: Sequence[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s (first call only)...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return np.asarray(_model.encode(list(texts), normalize_embeddings=True))

def best_match(query: str,candidates: list[str],threshold: float = 0.45,margin: float = 0.03,encode_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None) -> Optional[tuple[str, float]]:

    if not candidates:
        return None

    encode = encode_fn or _default_encode
    try:
        query_vec = encode([query])[0]
        cand_vecs = encode(candidates)
    except Exception:
        logger.exception("Embedding model unavailable, skipping semantic match")
        return None

    scores = np.asarray(cand_vecs) @ np.asarray(query_vec)
    order = np.argsort(scores)[::-1]
    top_idx = int(order[0])
    top_score = float(scores[top_idx])

    if top_score < threshold:
        return None
    if len(order) > 1 and (top_score - float(scores[int(order[1])])) < margin:
        return None

    return candidates[top_idx], top_score
