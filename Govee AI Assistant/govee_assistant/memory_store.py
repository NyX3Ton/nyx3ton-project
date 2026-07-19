#memory_store.py
#
# Long-term memory / RAG store, backed by a LlamaIndex VectorStoreIndex over
# ChromaDB. The public API (add / recent / search) is unchanged from the
# earlier raw-Chroma version, so the agent loop and InfoTools are unaffected -
# only the internals moved to LlamaIndex:
#
#   * search()  -> a LlamaIndex retriever (semantic RAG over the memories)
#   * recent()  -> a plain metadata listing straight off the Chroma collection
#                  (no embedding call, so it's cheap and works before any model
#                  is loaded)
#
# Embeddings reuse the same lazily-loaded multilingual model semantic_match.py
# already loads for device-name matching (via SemanticMatchEmbedding), so this
# store never downloads a second embedding model.

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import config as app_config

logger = logging.getLogger("memory_store")

COLLECTION_NAME = "assistant_memory"
DEFAULT_PATH = app_config.GOVEE_MEMORY_DB

# Sentinel so callers can pass similarity_cutoff=None to mean "explicitly
# disabled" distinct from "not passed, fall back to the configured default".
_USE_CONFIG_CUTOFF = object()


class SemanticMatchEmbedding(BaseEmbedding):
    """LlamaIndex embedding backed by semantic_match.encode.

    Reuses the same multilingual model semantic_match.py already loads for
    device-name matching, rather than pulling in a second embedding model for
    the memory store.
    """

    def _get_query_embedding(self, query: str) -> list[float]:
        from . import semantic_match

        return semantic_match.encode([query])[0].tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        from . import semantic_match

        return semantic_match.encode([text])[0].tolist()

    def _get_text_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        from . import semantic_match

        return [vec.tolist() for vec in semantic_match.encode(list(texts))]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


def _row(content: str, metadata: Optional[dict]) -> dict:
    metadata = metadata or {}
    return {
        "when": metadata.get("created_at", ""),
        "category": metadata.get("category", ""),
        "content": content,
    }


class MemoryStore:
    def __init__(
        self,
        path: Optional[str] = None,
        embed_model: Optional[BaseEmbedding] = None,
        client: Optional[Any] = None,
        similarity_cutoff: Any = _USE_CONFIG_CUTOFF,
    ):
        """
        similarity_cutoff: if set, search() drops retrieved memories whose
        relevance score is below this value. Left off by default because the
        score scale depends on the embedding model, and a too-aggressive cutoff
        would silently hide legitimate recalls; enable it deliberately once
        tuned for your model. When not passed, falls back to
        config.GOVEE_MEMORY_SIMILARITY_CUTOFF (env var GOVEE_MEMORY_SIMILARITY_CUTOFF);
        pass an explicit None to force it off regardless of config.
        """
        self.path = path or DEFAULT_PATH
        self._embed_model = embed_model or SemanticMatchEmbedding()
        self._similarity_cutoff = (
            app_config.GOVEE_MEMORY_SIMILARITY_CUTOFF
            if similarity_cutoff is _USE_CONFIG_CUTOFF
            else similarity_cutoff
        )

        if client is None:
            client = chromadb.PersistentClient(path=self.path)
        self._client = client
        # cosine space keeps retriever scores comparable across queries.
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self._embed_model
        )

    def add(self, category: str, content: str) -> None:
        node = TextNode(
            id_=uuid.uuid4().hex,
            text=content,
            metadata={"category": category, "created_at": datetime.now(timezone.utc).isoformat()},
        )
        self._index.insert_nodes([node])

    def recent(self, limit: int = 10, category: Optional[str] = None) -> list[dict]:
        where = {"category": category} if category else None
        try:
            result = self._collection.get(where=where)
        except Exception:  # noqa: BLE001
            logger.exception("MemoryStore.recent failed")
            return []

        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        rows = [_row(doc, meta) for doc, meta in zip(docs, metas)]
        rows.sort(key=lambda r: r["when"], reverse=True)
        return rows[:limit]

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> list[dict]:
        filters = None
        if category:
            filters = MetadataFilters(filters=[MetadataFilter(key="category", value=category)])
        try:
            retriever = self._index.as_retriever(similarity_top_k=top_k, filters=filters)
            nodes = retriever.retrieve(query)
        except Exception:  # noqa: BLE001
            logger.exception("MemoryStore.search failed, falling back to empty result")
            return []

        rows: list[dict] = []
        for scored in nodes:
            if (
                self._similarity_cutoff is not None
                and scored.score is not None
                and scored.score < self._similarity_cutoff
            ):
                continue
            rows.append(_row(scored.node.get_content(), scored.node.metadata))
        return rows
