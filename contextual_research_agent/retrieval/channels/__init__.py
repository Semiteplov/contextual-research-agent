from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from contextual_research_agent.retrieval.types import ChannelName, ChannelResult


class RetrievalChannel(ABC):
    """
    Base interface for a retrieval channel.

    Contract:
      - `retrieve()` is async and stateless (no side effects between calls).
      - Returns ChannelResult with scored, ranked candidates.
      - Channel is responsible for its own error handling:
        on failure, return empty ChannelResult with error in metadata.
    """

    @property
    @abstractmethod
    def name(self) -> ChannelName:
        """Unique channel identifier."""
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 50,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChannelResult:
        """
        Execute retrieval for the given query.

        Args:
            query: Raw query text.
            query_embedding: Pre-computed query embedding (avoids redundant encoding).
            top_k: Maximum candidates to return.
            filters: Channel-specific filters (e.g., section_type, document_id).
            **kwargs: Channel-specific parameters.

        Returns:
            ChannelResult with ranked ScoredCandidates.
        """
        ...

    async def warmup(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError
