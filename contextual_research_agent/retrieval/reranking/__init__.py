from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

import torch
from sentence_transformers import CrossEncoder

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.retrieval.config import RerankConfig
from contextual_research_agent.retrieval.types import (
    RerankResult,
    ScoredCandidate,
)

logger = get_logger(__name__)


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        top_k: int = 10,
    ) -> RerankResult:
        """
        Re-score and re-rank candidates using (query, passage) scoring.

        Args:
            query: Original query text.
            candidates: Pre-fusion candidates to rerank.
            top_k: Number of candidates to return after reranking.

        Returns:
            RerankResult with re-scored, re-ranked candidates.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using sentence-transformers CrossEncoder.
    """

    def __init__(self, config: RerankConfig | None = None):
        self._config = config or RerankConfig()
        self._model: CrossEncoder | None = None

    def _ensure_model(self):
        if self._model is not None:
            return

        device = self._config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading reranker: %s on %s", self._config.model, device)
        self._model = CrossEncoder(
            self._config.model,
            device=device,
            max_length=self._config.max_length,
        )
        logger.info("Reranker loaded: %s", self._config.model)

    @property
    def model_name(self) -> str:
        return self._config.model

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        top_k: int | None = None,
    ) -> RerankResult:
        top_k = top_k or self._config.top_k
        t0 = time.perf_counter()

        if not candidates:
            return RerankResult(
                candidates=[],
                latency_ms=0.0,
                model_name=self._config.model,
            )

        self._ensure_model()
        model = self._model
        assert model is not None

        pairs = [(query, c.chunk.text) for c in candidates]

        def _score():
            return model.predict(
                pairs,
                batch_size=self._config.batch_size,
                show_progress_bar=False,
            )

        scores = await asyncio.to_thread(_score)

        scored = list(zip(candidates, scores, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: list[ScoredCandidate] = []
        for new_rank, (original, new_score) in enumerate(scored[:top_k]):
            reranked.append(
                ScoredCandidate(
                    chunk=original.chunk,
                    score=float(new_score),
                    rank=new_rank,
                    channel=original.channel,
                    metadata={
                        **original.metadata,
                        "pre_rerank_score": original.score,
                        "pre_rerank_rank": original.rank,
                        "reranker_score": float(new_score),
                    },
                )
            )

        rank_changes = []
        for c in reranked:
            pre_rank = c.metadata.get("pre_rerank_rank", c.rank)
            rank_changes.append(abs(c.rank - pre_rank))
        mean_rank_change = sum(rank_changes) / len(rank_changes) if rank_changes else 0.0

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "Reranker: %d → %d candidates, mean rank change %.1f (%.0fms)",
            len(candidates),
            len(reranked),
            mean_rank_change,
            latency_ms,
        )

        return RerankResult(
            candidates=reranked,
            latency_ms=latency_ms,
            model_name=self._config.model,
            rank_changes=mean_rank_change,
        )


class NoOpReranker(Reranker):
    @property
    def model_name(self) -> str:
        return "noop"

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredCandidate],
        top_k: int = 10,
    ) -> RerankResult:
        return RerankResult(
            candidates=candidates[:top_k],
            latency_ms=0.0,
            model_name="noop",
            rank_changes=0.0,
        )


def create_reranker(config: RerankConfig | None = None) -> Reranker:
    config = config or RerankConfig()
    if not config.enabled:
        return NoOpReranker()
    return CrossEncoderReranker(config)
