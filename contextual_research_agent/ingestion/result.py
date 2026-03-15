from dataclasses import dataclass, field
from typing import Any

from contextual_research_agent.ingestion.domain.entities import Document
from contextual_research_agent.ingestion.embeddings.metrics import EmbeddingMetrics
from contextual_research_agent.ingestion.extraction.citation_extractor import (
    CitationExtractionResult,
)
from contextual_research_agent.ingestion.extraction.entity_extractor import EntityExtractionResult
from contextual_research_agent.ingestion.parsers.metrics import ChunkingMetrics
from contextual_research_agent.ingestion.vectorstores.metrics import StoreOperationMetrics


@dataclass
class StageLatency:
    """Timing for each pipeline stage in milliseconds."""

    parse_ms: float = 0.0
    chunk_ms: float = 0.0
    enrich_ms: float = 0.0  # section classify + citation extract
    entity_extract_ms: float = 0.0  # LLM entity extraction
    embed_ms: float = 0.0  # chunk embeddings
    paper_embed_ms: float = 0.0  # paper-level embedding
    index_ms: float = 0.0
    graph_store_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.parse_ms
            + self.chunk_ms
            + self.enrich_ms
            + self.entity_extract_ms
            + self.embed_ms
            + self.paper_embed_ms
            + self.index_ms
            + self.graph_store_ms
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "parse_ms": round(self.parse_ms, 1),
            "chunk_ms": round(self.chunk_ms, 1),
            "enrich_ms": round(self.enrich_ms, 1),
            "entity_extract_ms": round(self.entity_extract_ms, 1),
            "embed_ms": round(self.embed_ms, 1),
            "paper_embed_ms": round(self.paper_embed_ms, 1),
            "index_ms": round(self.index_ms, 1),
            "graph_store_ms": round(self.graph_store_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@dataclass
class ExtractionMetrics:
    """Metrics from the enrichment stages (section classification + citations)."""

    # Section classification
    section_classify_ms: float = 0.0
    section_type_distribution: dict[str, int] = field(default_factory=dict)
    unknown_section_rate: float = 0.0

    # Citation extraction
    citation_extract_ms: float = 0.0
    total_references: int = 0
    resolved_references: int = 0
    resolution_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_classify_ms": round(self.section_classify_ms, 1),
            "section_type_distribution": self.section_type_distribution,
            "unknown_section_rate": round(self.unknown_section_rate, 3),
            "citation_extract_ms": round(self.citation_extract_ms, 1),
            "total_references": self.total_references,
            "resolved_references": self.resolved_references,
            "resolution_rate": round(self.resolution_rate, 3),
        }


@dataclass
class IngestionMetrics:
    """Aggregated metrics from all pipeline stages."""

    latency: StageLatency = field(default_factory=StageLatency)
    chunking: ChunkingMetrics | None = None
    extraction: ExtractionMetrics | None = None
    entity_extraction: EntityExtractionResult | None = None
    embedding: EmbeddingMetrics | None = None
    indexing: StoreOperationMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"latency": self.latency.to_dict()}
        if self.chunking:
            result["chunking"] = self.chunking.to_dict()
        if self.extraction:
            result["extraction"] = self.extraction.to_dict()
        if self.entity_extraction:
            result["entity_extraction"] = self.entity_extraction.to_dict()
        if self.embedding:
            result["embedding"] = self.embedding.to_dict()
        if self.indexing:
            result["indexing"] = self.indexing.to_dict()
        return result


@dataclass
class IngestionResult:
    """Typed result of a single document ingestion."""

    run_id: str
    file_path: str
    status: str  # "indexed" | "failed"
    document: Document | None = None
    chunk_count: int = 0
    metrics: IngestionMetrics = field(default_factory=IngestionMetrics)
    citation_result: CitationExtractionResult | None = None
    error: str | None = None
    completed_at: str = ""

    @property
    def success(self) -> bool:
        return self.status == "indexed"

    @property
    def document_id(self) -> str | None:
        return self.document.id if self.document else None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "run_id": self.run_id,
            "file_path": self.file_path,
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "success": self.success,
            "error": self.error,
            "completed_at": self.completed_at,
            "metrics": self.metrics.to_dict(),
        }
        if self.citation_result:
            result["citations"] = self.citation_result.to_dict()
        return result


@dataclass
class BatchResult:
    """Typed result of batch ingestion."""

    results: list[IngestionResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return self.total - self.succeeded

    @property
    def total_chunks(self) -> int:
        return sum(r.chunk_count for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "total_chunks": self.total_chunks,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "results": [r.to_dict() for r in self.results],
        }
