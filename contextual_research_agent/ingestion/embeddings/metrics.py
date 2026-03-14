from dataclasses import dataclass
from typing import Any


@dataclass
class EmbeddingMetrics:
    """Metrics for a single embedding operation."""

    operation: str  # "embed_passages" | "embed_query"
    model_name: str
    num_texts: int
    batch_size: int
    duration_ms: float
    throughput_texts_per_sec: float
    dimension: int
    success: bool
    error: str | None = None

    mean_text_length: float = 0.0
    max_text_length: int = 0
    empty_text_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "model_name": self.model_name,
            "num_texts": self.num_texts,
            "batch_size": self.batch_size,
            "duration_ms": round(self.duration_ms, 2),
            "throughput_texts_per_sec": round(self.throughput_texts_per_sec, 2),
            "dimension": self.dimension,
            "success": self.success,
            "error": self.error,
            "mean_text_length": round(self.mean_text_length, 1),
            "max_text_length": self.max_text_length,
            "empty_text_count": self.empty_text_count,
        }
