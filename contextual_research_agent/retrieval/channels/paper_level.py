from __future__ import annotations

import time
from typing import Any

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.ingestion.embeddings.base import Embedder
from contextual_research_agent.ingestion.vectorstores.qdrant_store import QdrantStore
from contextual_research_agent.retrieval.channels import RetrievalChannel
from contextual_research_agent.retrieval.config import PaperLevelConfig
from contextual_research_agent.retrieval.types import (
    ChannelName,
    ChannelResult,
    ScoredCandidate,
)

logger = get_logger(__name__)


class PaperLevelChannel(RetrievalChannel):
    """
    Paper-level → chunk expansion channel.
    """

    def __init__(
        self,
        paper_store: QdrantStore,
        chunk_store: QdrantStore,
        embedder: Embedder,
        config: PaperLevelConfig | None = None,
    ):
        self._paper_store = paper_store
        self._chunk_store = chunk_store
        self._embedder = embedder
        self._config = config or PaperLevelConfig()

    @property
    def name(self) -> ChannelName:
        return ChannelName.PAPER_LEVEL

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChannelResult:
        t0 = time.perf_counter()

        try:
            if query_embedding is None:
                query_embedding = await self._embedder.embed_query(query)

            paper_results, _ = await self._paper_store.search(
                query_embedding=query_embedding,
                top_k=self._config.top_k_papers,
                filters=filters,
            )

            if not paper_results:
                return ChannelResult(
                    channel=ChannelName.PAPER_LEVEL,
                    candidates=[],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

            candidates: list[ScoredCandidate] = []
            for paper_chunk, paper_score in paper_results:
                doc_id = paper_chunk.document_id

                chunk_results, _ = await self._chunk_store.search(
                    query_embedding=query_embedding,
                    top_k=self._config.chunks_per_paper,
                    filters={"document_id": doc_id},
                )

                for chunk, chunk_score in chunk_results:
                    combined_score = paper_score * 0.3 + chunk_score * 0.7
                    candidates.append(
                        ScoredCandidate(
                            chunk=chunk,
                            score=combined_score,
                            rank=len(candidates),
                            channel=ChannelName.PAPER_LEVEL,
                            metadata={
                                "paper_score": paper_score,
                                "chunk_score": chunk_score,
                                "source_paper": doc_id,
                            },
                        )
                    )

            candidates.sort(key=lambda c: c.score, reverse=True)
            candidates = [
                ScoredCandidate(
                    chunk=c.chunk,
                    score=c.score,
                    rank=i,
                    channel=c.channel,
                    metadata=c.metadata,
                )
                for i, c in enumerate(candidates)
            ]

            latency_ms = (time.perf_counter() - t0) * 1000

            logger.debug(
                "Paper-level: %d papers → %d chunks (%.0fms)",
                len(paper_results),
                len(candidates),
                latency_ms,
            )

            return ChannelResult(
                channel=ChannelName.PAPER_LEVEL,
                candidates=candidates,
                latency_ms=latency_ms,
                metadata={
                    "papers_found": len(paper_results),
                    "chunks_per_paper": self._config.chunks_per_paper,
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Paper-level channel failed: %s", e)
            return ChannelResult(
                channel=ChannelName.PAPER_LEVEL,
                candidates=[],
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )
