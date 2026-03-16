from __future__ import annotations

import time
from typing import Any

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.ingestion.embeddings.base import Embedder
from contextual_research_agent.ingestion.vectorstores.qdrant_store import QdrantStore
from contextual_research_agent.retrieval.channels import RetrievalChannel
from contextual_research_agent.retrieval.config import DenseChannelConfig
from contextual_research_agent.retrieval.types import (
    ChannelName,
    ChannelResult,
    ScoredCandidate,
)

logger = get_logger(__name__)


class DenseChannel(RetrievalChannel):
    """
    Bi-encoder dense retrieval over Qdrant vector index.

    Supports:
      - Section-type pre-filtering
      - Document-scoped retrieval
      - Score threshold filtering
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        embedder: Embedder,
        config: DenseChannelConfig | None = None,
    ):
        self._store = vector_store
        self._embedder = embedder
        self._config = config or DenseChannelConfig()

    @property
    def name(self) -> ChannelName:
        return ChannelName.DENSE

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
            if query_embedding is None:
                query_embedding = await self._embedder.embed_query(query)

            search_filters = dict(filters) if filters else {}

            section_types = kwargs.get("section_types") or self._config.section_types
            if section_types:
                search_filters["metadata.section_type"] = section_types

            results, _search_metrics = await self._store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                score_threshold=self._config.score_threshold,
                filters=search_filters if search_filters else None,
            )

            candidates = [
                ScoredCandidate(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    channel=ChannelName.DENSE,
                )
                for rank, (chunk, score) in enumerate(results)
            ]

            latency_ms = (time.perf_counter() - t0) * 1000

            logger.debug(
                "Dense channel: %d candidates in %.0fms",
                len(candidates),
                latency_ms,
            )

            return ChannelResult(
                channel=ChannelName.DENSE,
                candidates=candidates,
                latency_ms=latency_ms,
                metadata={
                    "top_k": top_k,
                    "filters": search_filters,
                    "score_threshold": self._config.score_threshold,
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Dense channel failed: %s", e)
            return ChannelResult(
                channel=ChannelName.DENSE,
                candidates=[],
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )
