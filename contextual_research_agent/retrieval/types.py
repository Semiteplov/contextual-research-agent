from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contextual_research_agent.ingestion.domain.entities import Chunk


class ChannelName(str, Enum):
    """Registered retrieval channel identifiers."""

    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH_CITATION = "graph_citation"
    GRAPH_ENTITY = "graph_entity"
    PAPER_LEVEL = "paper_level"


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    chunk: Chunk
    score: float
    rank: int
    channel: ChannelName
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id


@dataclass
class ChannelResult:
    channel: ChannelName
    candidates: list[ScoredCandidate]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "num_candidates": self.num_candidates,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }


@dataclass
class FusionResult:
    candidates: list[ScoredCandidate]
    channel_contributions: dict[str, int]
    channel_overlaps: dict[str, int]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_candidates": len(self.candidates),
            "channel_contributions": self.channel_contributions,
            "channel_overlaps": self.channel_overlaps,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class RerankResult:
    candidates: list[ScoredCandidate]
    latency_ms: float
    model_name: str = ""
    rank_changes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_candidates": len(self.candidates),
            "latency_ms": round(self.latency_ms, 2),
            "model_name": self.model_name,
            "mean_rank_change": round(self.rank_changes, 2),
        }


@dataclass
class RetrievalResult:
    query: str
    candidates: list[ScoredCandidate]
    context: str

    channel_results: list[ChannelResult] = field(default_factory=list)
    fusion_result: FusionResult | None = None
    rerank_result: RerankResult | None = None

    total_latency_ms: float = 0.0
    query_analysis_ms: float = 0.0

    intent: str = ""
    active_channels: list[str] = field(default_factory=list)
    filters_applied: dict[str, Any] = field(default_factory=dict)

    @property
    def texts(self) -> list[str]:
        return [c.chunk.text for c in self.candidates]

    @property
    def num_results(self) -> int:
        return len(self.candidates)

    @property
    def document_ids(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for c in self.candidates:
            if c.document_id not in seen:
                seen.add(c.document_id)
                result.append(c.document_id)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "num_results": self.num_results,
            "num_documents": len(self.document_ids),
            "active_channels": self.active_channels,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "query_analysis_ms": round(self.query_analysis_ms, 2),
            "channels": [cr.to_dict() for cr in self.channel_results],
            "fusion": self.fusion_result.to_dict() if self.fusion_result else None,
            "rerank": self.rerank_result.to_dict() if self.rerank_result else None,
            "filters_applied": self.filters_applied,
        }
