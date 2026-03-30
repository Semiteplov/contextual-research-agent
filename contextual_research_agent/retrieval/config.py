from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from contextual_research_agent.retrieval.types import ChannelName


@dataclass
class DenseChannelConfig:
    """Dense (bi-encoder) retrieval channel."""

    enabled: bool = True
    top_k: int = 50
    score_threshold: float | None = None
    collection_name: str = "documents"
    section_types: list[str] | None = None


@dataclass
class SparseChannelConfig:
    """Sparse (BM25 / SPLADE) retrieval channel."""

    enabled: bool = True
    top_k: int = 50
    backend: Literal["qdrant_sparse", "rank_bm25"] = "qdrant_sparse"
    sparse_model: str = "Qdrant/bm25"
    collection_name: str = "documents"
    section_types: list[str] | None = None


@dataclass
class GraphChannelConfig:
    """Graph retrieval channel (citation + entity)."""

    citation_enabled: bool = True
    entity_enabled: bool = True
    seed_top_k: int = 5
    citation_depth: int = 1
    max_papers: int = 10
    chunks_per_paper: int = 3


@dataclass
class PaperLevelConfig:
    """Paper-level → chunk expansion channel."""

    enabled: bool = True
    top_k_papers: int = 5
    chunks_per_paper: int = 5
    paper_collection: str = "documents_papers"


@dataclass
class FusionConfig:
    strategy: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    top_n: int = 50

    channel_weights: dict[str, float] = field(
        default_factory=lambda: {
            ChannelName.DENSE.value: 1.0,
            ChannelName.SPARSE.value: 0.3,
            ChannelName.GRAPH_CITATION.value: 0.7,
            ChannelName.GRAPH_ENTITY.value: 0.5,
            ChannelName.PAPER_LEVEL.value: 0.5,
        }
    )


@dataclass
class RerankConfig:
    """Cross-encoder reranking configuration."""

    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 20
    batch_size: int = 16
    device: str | None = None
    max_length: int = 512


@dataclass
class QueryAnalysisConfig:
    """Query understanding and routing."""

    enabled: bool = True
    method: Literal["rule", "llm"] = "rule"


@dataclass
class ContextAssemblyConfig:
    """Post-retrieval context."""

    max_tokens: int = 4096
    ordering: Literal["document_then_chunk", "score"] = "document_then_chunk"
    include_metadata: bool = True
    separator: str = "\n\n---\n\n"
    chunk_template: str = "[{chunk_id}] [{section_type}] (score: {score:.3f})\n{text}"


@dataclass
class RetrievalConfig:
    dense: DenseChannelConfig = field(default_factory=DenseChannelConfig)
    sparse: SparseChannelConfig = field(default_factory=SparseChannelConfig)
    graph: GraphChannelConfig = field(default_factory=GraphChannelConfig)
    paper_level: PaperLevelConfig = field(default_factory=PaperLevelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    query_analysis: QueryAnalysisConfig = field(default_factory=QueryAnalysisConfig)
    context: ContextAssemblyConfig = field(default_factory=ContextAssemblyConfig)

    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    def to_mlflow_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "embedding_model": self.embedding_model,
            # Dense
            "dense/enabled": self.dense.enabled,
            "dense/top_k": self.dense.top_k,
            # Sparse
            "sparse/enabled": self.sparse.enabled,
            "sparse/top_k": self.sparse.top_k,
            "sparse/backend": self.sparse.backend,
            "sparse/model": self.sparse.sparse_model,
            # Graph
            "graph/citation_enabled": self.graph.citation_enabled,
            "graph/entity_enabled": self.graph.entity_enabled,
            "graph/seed_top_k": self.graph.seed_top_k,
            "graph/citation_depth": self.graph.citation_depth,
            "graph/max_papers": self.graph.max_papers,
            # Paper-level
            "paper_level/enabled": self.paper_level.enabled,
            "paper_level/top_k_papers": self.paper_level.top_k_papers,
            # Fusion
            "fusion/strategy": self.fusion.strategy,
            "fusion/rrf_k": self.fusion.rrf_k,
            "fusion/top_n": self.fusion.top_n,
            # Rerank
            "rerank/enabled": self.rerank.enabled,
            "rerank/model": self.rerank.model,
            "rerank/top_k": self.rerank.top_k,
            # Query
            "query_analysis/enabled": self.query_analysis.enabled,
            "query_analysis/method": self.query_analysis.method,
            # Context
            "context/max_tokens": self.context.max_tokens,
            "context/ordering": self.context.ordering,
        }
        for ch, w in self.fusion.channel_weights.items():
            params[f"fusion/weight_{ch}"] = w

        return params

    def active_channels(self) -> list[ChannelName]:
        channels: list[ChannelName] = []
        if self.dense.enabled:
            channels.append(ChannelName.DENSE)
        if self.sparse.enabled:
            channels.append(ChannelName.SPARSE)
        if self.graph.citation_enabled:
            channels.append(ChannelName.GRAPH_CITATION)
        if self.graph.entity_enabled:
            channels.append(ChannelName.GRAPH_ENTITY)
        if self.paper_level.enabled:
            channels.append(ChannelName.PAPER_LEVEL)
        return channels
