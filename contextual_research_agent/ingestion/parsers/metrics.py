from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contextual_research_agent.ingestion.domain.entities import (
    Chunk,
    ChunkType,
    DoclingParserConfig,
    Document,
)


@dataclass
class ChunkingMetrics:
    """
    Quality metrics for a single document's chunking result.

    Use for: ablation studies on chunk size / merge strategy / context inclusion,
    monitoring ingestion pipeline health, comparing parser configurations.
    """

    document_id: str
    total_chunks: int
    total_tokens: int

    # Token length distribution
    token_lengths: list[int] = field(default_factory=list)
    mean_tokens: float = 0.0
    median_tokens: float = 0.0
    std_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    p5_tokens: float = 0.0
    p95_tokens: float = 0.0

    # Quality indicators
    empty_chunk_count: int = 0
    oversized_chunk_count: int = 0
    type_distribution: dict[str, int] = field(default_factory=dict)

    # Structural coverage
    sections_in_doc: int = 0
    sections_covered: int = 0
    section_coverage_ratio: float = 0.0

    # Context overhead (only meaningful when include_context=True)
    mean_context_overhead: float = 0.0  # avg (contextualized - raw) / contextualized

    # Fallback indicator
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation for logging / MLflow."""
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "mean_tokens": round(self.mean_tokens, 1),
            "median_tokens": round(self.median_tokens, 1),
            "std_tokens": round(self.std_tokens, 1),
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "p5_tokens": round(self.p5_tokens, 1),
            "p95_tokens": round(self.p95_tokens, 1),
            "empty_chunk_count": self.empty_chunk_count,
            "oversized_chunk_count": self.oversized_chunk_count,
            "type_distribution": self.type_distribution,
            "sections_in_doc": self.sections_in_doc,
            "sections_covered": self.sections_covered,
            "section_coverage_ratio": round(self.section_coverage_ratio, 3),
            "mean_context_overhead": round(self.mean_context_overhead, 3),
            "used_fallback": self.used_fallback,
        }


def compute_chunking_metrics(
    document: Document,
    chunks: list[Chunk],
    config: DoclingParserConfig,
    context_overheads: list[float] | None = None,
    used_fallback: bool = False,
) -> ChunkingMetrics:
    """
    Compute chunking quality metrics for a single document.

    Args:
        document: Parsed document with section metadata.
        chunks: Produced chunks.
        config: Parser config (for thresholds).
        context_overheads: Per-chunk ratio of (ctx_tokens - raw_tokens) / ctx_tokens.
                           None if include_context=False.
        used_fallback: Whether fallback chunking was used.
    """
    if not chunks:
        return ChunkingMetrics(
            document_id=document.id,
            total_chunks=0,
            total_tokens=0,
            used_fallback=used_fallback,
        )

    token_lengths = [c.token_count for c in chunks]
    arr = np.array(token_lengths, dtype=np.float64)

    type_dist: dict[str, int] = {}
    for c in chunks:
        t = c.metadata.get("chunk_type", ChunkType.TEXT.value)
        type_dist[t] = type_dist.get(t, 0) + 1

    sections_in_doc = len(document.metadata.get("sections", []))
    sections_in_chunks = len({c.section for c in chunks if c.section})
    coverage = sections_in_chunks / sections_in_doc if sections_in_doc > 0 else 1.0

    mean_overhead = 0.0
    if context_overheads:
        mean_overhead = float(np.mean([o for o in context_overheads if o > 0]))

    return ChunkingMetrics(
        document_id=document.id,
        total_chunks=len(chunks),
        total_tokens=int(arr.sum()),
        token_lengths=token_lengths,
        mean_tokens=float(arr.mean()),
        median_tokens=float(np.median(arr)),
        std_tokens=float(arr.std()),
        min_tokens=int(arr.min()),
        max_tokens=int(arr.max()),
        p5_tokens=float(np.percentile(arr, 5)),
        p95_tokens=float(np.percentile(arr, 95)),
        empty_chunk_count=int((arr < config.min_chunk_tokens).sum()),
        oversized_chunk_count=int((arr > config.max_tokens).sum()),
        type_distribution=type_dist,
        sections_in_doc=sections_in_doc,
        sections_covered=sections_in_chunks,
        section_coverage_ratio=coverage,
        mean_context_overhead=mean_overhead,
        used_fallback=used_fallback,
    )


def aggregate_corpus_metrics(metrics_list: list[ChunkingMetrics]) -> dict[str, Any]:
    """
    Aggregate chunking metrics across a corpus.

    Useful for comparing configurations in ablation studies:
    e.g. max_tokens=256 vs 512 vs 1024, merge_peers on/off, etc.
    """
    if not metrics_list:
        return {"num_documents": 0}

    all_lengths: list[int] = []
    total_empty = 0
    total_oversized = 0
    total_fallback = 0
    all_section_coverage: list[float] = []
    all_overhead: list[float] = []

    for m in metrics_list:
        all_lengths.extend(m.token_lengths)
        total_empty += m.empty_chunk_count
        total_oversized += m.oversized_chunk_count
        total_fallback += int(m.used_fallback)
        all_section_coverage.append(m.section_coverage_ratio)
        if m.mean_context_overhead > 0:
            all_overhead.append(m.mean_context_overhead)

    arr = np.array(all_lengths, dtype=np.float64) if all_lengths else np.array([0.0])
    n_chunks = len(all_lengths)

    return {
        "num_documents": len(metrics_list),
        "total_chunks": n_chunks,
        "chunks_per_doc_mean": round(n_chunks / len(metrics_list), 1),
        "corpus_mean_tokens": round(float(arr.mean()), 1),
        "corpus_median_tokens": round(float(np.median(arr)), 1),
        "corpus_std_tokens": round(float(arr.std()), 1),
        "corpus_p5": round(float(np.percentile(arr, 5)), 1),
        "corpus_p95": round(float(np.percentile(arr, 95)), 1),
        "total_empty_chunks": total_empty,
        "total_oversized_chunks": total_oversized,
        "empty_rate": round(total_empty / n_chunks, 4) if n_chunks else 0,
        "oversized_rate": round(total_oversized / n_chunks, 4) if n_chunks else 0,
        "fallback_rate": round(total_fallback / len(metrics_list), 4),
        "mean_section_coverage": round(float(np.mean(all_section_coverage)), 3),
        "mean_context_overhead": round(float(np.mean(all_overhead)), 3) if all_overhead else 0.0,
    }
