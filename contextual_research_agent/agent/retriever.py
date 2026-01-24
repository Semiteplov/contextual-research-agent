from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from contextual_research_agent.common import logging
from contextual_research_agent.ingestion.domain.entities import RetrievedChunk
from contextual_research_agent.ingestion.embeddings.base import Embedder
from contextual_research_agent.ingestion.vectorstores.qdrant_store import QdrantStore

logger = logging.get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float

    top_k: int = 0
    filters: dict[str, Any] = field(default_factory=dict)

    @property
    def texts(self) -> list[str]:
        return [rc.chunk.text for rc in self.chunks]

    @property
    def context(self) -> str:
        parts = []
        for rc in self.chunks:
            chunk_id = rc.chunk.id
            text = rc.chunk.text
            score = rc.score
            parts.append(f"[{chunk_id}] (score: {score:.3f})\n{text}")
        return "\n\n---\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "num_chunks": len(self.chunks),
            "top_k": self.top_k,
            "latency_ms": self.latency_ms,
            "filters": self.filters,
        }


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: QdrantStore,
        default_top_k: int = 10,
        default_score_threshold: float | None = None,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._default_score_threshold = default_score_threshold

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self._default_top_k
        score_threshold = score_threshold or self._default_score_threshold

        start = time.perf_counter()

        search_filters = filters.copy() if filters else {}
        if document_ids:
            search_filters["document_id"] = document_ids

        query_embedding = await self._embedder.embed_query(query)

        results = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=search_filters if search_filters else None,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        retrieved_chunks = [
            RetrievedChunk(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(results)
        ]

        logger.debug(
            f"Retrieved {len(retrieved_chunks)} chunks for query "
            f"({elapsed_ms:.0f}ms, top_k={top_k})"
        )

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            latency_ms=elapsed_ms,
            top_k=top_k,
            filters=search_filters,
        )

    async def retrieve_for_document(
        self,
        query: str,
        document_id: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        return await self.retrieve(
            query=query,
            top_k=top_k,
            document_ids=[document_id],
        )

    async def retrieve_by_section(
        self,
        query: str,
        section: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filters={"section": section},
        )

    async def multi_query_retrieve(
        self,
        queries: list[str],
        top_k_per_query: int = 5,
        deduplicate: bool = True,
    ) -> RetrievalResult:
        start = time.perf_counter()

        all_chunks: dict[str, RetrievedChunk] = {}

        for query in queries:
            result = await self.retrieve(query, top_k=top_k_per_query)

            for rc in result.chunks:
                chunk_id = rc.chunk.id

                if deduplicate:
                    if chunk_id not in all_chunks or all_chunks[chunk_id].score < rc.score:
                        all_chunks[chunk_id] = rc
                else:
                    key = f"{chunk_id}_{len(all_chunks)}"
                    all_chunks[key] = rc

        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)

        ranked_chunks = [
            RetrievedChunk(chunk=rc.chunk, score=rc.score, rank=rank)
            for rank, rc in enumerate(sorted_chunks)
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=" | ".join(queries),
            chunks=ranked_chunks,
            latency_ms=elapsed_ms,
            top_k=len(ranked_chunks),
            filters={"multi_query": True, "num_queries": len(queries)},
        )


def create_retriever(
    embedder: Embedder,
    vector_store: QdrantStore,
    default_top_k: int = 10,
    default_score_threshold: float | None = None,
) -> Retriever:
    return Retriever(
        embedder=embedder,
        vector_store=vector_store,
        default_top_k=default_top_k,
        default_score_threshold=default_score_threshold,
    )
