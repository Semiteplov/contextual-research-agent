from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from qdrant_client import models

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.ingestion.vectorstores.qdrant_store import (
    _build_filter,
    _payload_to_chunk,
)
from contextual_research_agent.retrieval.channels import RetrievalChannel
from contextual_research_agent.retrieval.config import SparseChannelConfig
from contextual_research_agent.retrieval.types import (
    ChannelName,
    ChannelResult,
    ScoredCandidate,
)

logger = get_logger(__name__)


class SparseEncoder(Protocol):
    """Protocol for sparse query encoding."""

    def encode_query(self, query: str) -> dict[int, float]:
        """Encode query to sparse vector {token_id: weight}."""
        ...


class QdrantSparseBackend:
    """
    Sparse retrieval.
    """

    def __init__(
        self,
        client,
        collection_name: str,
        sparse_vector_name: str = "sparse",
        sparse_encoder: SparseEncoder | None = None,
    ):
        self._client = client
        self._collection = collection_name
        self._sparse_name = sparse_vector_name
        self._encoder = sparse_encoder

    async def search(
        self,
        query: str,
        top_k: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Any, float]]:
        """
        Sparse vector search. Returns (payload, score) pairs.
        """
        if self._encoder is None:
            raise RuntimeError(
                "SparseEncoder not configured. "
                "Install fastembed and configure sparse_model in SparseChannelConfig."
            )

        sparse_vector = self._encoder.encode_query(query)

        query_vector = models.SparseVector(
            indices=list(sparse_vector.keys()),
            values=list(sparse_vector.values()),
        )

        qdrant_filter = None
        if filters:
            qdrant_filter = _build_filter(filters)

        def _search():
            return self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                using=self._sparse_name,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        response = await asyncio.to_thread(_search)

        return [(point.payload, float(point.score or 0.0)) for point in response.points]


class SparseChannel(RetrievalChannel):
    """
    Sparse (BM25) retrieval channel.
    """

    def __init__(
        self,
        backend: QdrantSparseBackend,
        config: SparseChannelConfig | None = None,
    ):
        self._backend = backend
        self._config = config or SparseChannelConfig()

    @property
    def name(self) -> ChannelName:
        return ChannelName.SPARSE

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChannelResult:
        top_k = top_k or self._config.top_k
        t0 = time.perf_counter()

        try:
            search_filters = dict(filters) if filters else {}
            section_types = kwargs.get("section_types") or self._config.section_types
            if section_types:
                search_filters["metadata.section_type"] = section_types

            results = await self._backend.search(
                query=query,
                top_k=top_k,
                filters=search_filters if search_filters else None,
            )

            candidates = [
                ScoredCandidate(
                    chunk=_payload_to_chunk(payload),
                    score=score,
                    rank=rank,
                    channel=ChannelName.SPARSE,
                )
                for rank, (payload, score) in enumerate(results)
            ]

            latency_ms = (time.perf_counter() - t0) * 1000

            logger.debug(
                "Sparse channel: %d candidates in %.0fms",
                len(candidates),
                latency_ms,
            )

            return ChannelResult(
                channel=ChannelName.SPARSE,
                candidates=candidates,
                latency_ms=latency_ms,
                metadata={"top_k": top_k, "backend": self._config.backend},
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Sparse channel failed: %s", e)
            return ChannelResult(
                channel=ChannelName.SPARSE,
                candidates=[],
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )
