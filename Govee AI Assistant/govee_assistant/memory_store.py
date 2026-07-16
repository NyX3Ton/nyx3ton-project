#memory_store.py
#
# Chroma-backed long-term memory / RAG store. Reuses the same multilingual
# embedding model semantic_match.py already loads for device-name matching
# (via semantic_match.encode) rather than pulling in Chroma's default
# embedding function, which would download a second, different model.

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

import chromadb
from chromadb import EmbeddingFunction

from . import config as app_config

logger = logging.getLogger("memory_store")

COLLECTION_NAME = "assistant_memory"
DEFAULT_PATH = app_config.GOVEE_MEMORY_DB


class _SemanticMatchEmbeddingFunction(EmbeddingFunction):
    """Adapts semantic_match.encode to Chroma's embedding-function interface,
    reusing the same lazily-loaded multilingual model semantic_match.py uses
    for device-name matching instead of downloading a second model."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "semantic_match"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "_SemanticMatchEmbeddingFunction":
        return _SemanticMatchEmbeddingFunction()

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        from . import semantic_match

        return semantic_match.encode(list(input)).tolist()


def _row(doc: str, metadata: dict) -> dict:
    return {
        "when": metadata.get("created_at", ""),
        "category": metadata.get("category", ""),
        "content": doc,
    }


class MemoryStore:
    def __init__(
        self,
        path: Optional[str] = None,
        embedding_fn: Optional[Callable[[Sequence[str]], list[list[float]]]] = None,
        client: Optional[Any] = None,
    ):
        self.path = path or DEFAULT_PATH
        embedding_fn = embedding_fn or _SemanticMatchEmbeddingFunction()

        if client is None:
            client = chromadb.PersistentClient(path=self.path)
        self._client = client
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )

    def add(self, category: str, content: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._collection.add(
            documents=[content],
            metadatas=[{"category": category, "created_at": now}],
            ids=[uuid.uuid4().hex],
        )

    def recent(self, limit: int = 10, category: Optional[str] = None) -> list[dict]:
        where = {"category": category} if category else None
        try:
            result = self._collection.get(where=where)
        except Exception:  # noqa: BLE001
            logger.exception("MemoryStore.recent failed")
            return []

        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        rows = [_row(doc, meta or {}) for doc, meta in zip(docs, metas)]
        rows.sort(key=lambda r: r["when"], reverse=True)
        return rows[:limit]

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> list[dict]:
        where = {"category": category} if category else None
        try:
            result = self._collection.query(query_texts=[query], n_results=top_k, where=where)
        except Exception:  # noqa: BLE001
            logger.exception("MemoryStore.search failed, falling back to empty result")
            return []

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        return [_row(doc, meta or {}) for doc, meta in zip(docs, metas)]
