from dataclasses import dataclass
from typing import Any


@dataclass
class StoreOperationMetrics:
    """Metrics for a single store operation. Logged per-call."""

    operation: str  # "upsert", "search", "delete", "get_by_ids"
    collection: str
    duration_ms: float
    num_items: int  # Points upserted / results returned / deleted
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "collection": self.collection,
            "duration_ms": round(self.duration_ms, 2),
            "num_items": self.num_items,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class SearchMetrics(StoreOperationMetrics):
    """Extended metrics for search operations."""

    top_k_requested: int = 0
    results_returned: int = 0
    min_score: float = 0.0
    max_score: float = 0.0
    mean_score: float = 0.0
    above_threshold_count: int = 0  # Results above score_threshold

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "top_k_requested": self.top_k_requested,
                "results_returned": self.results_returned,
                "min_score": round(self.min_score, 4),
                "max_score": round(self.max_score, 4),
                "mean_score": round(self.mean_score, 4),
                "above_threshold_count": self.above_threshold_count,
            }
        )
        return base
