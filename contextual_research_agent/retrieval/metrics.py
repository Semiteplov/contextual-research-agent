from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from contextual_research_agent.retrieval.types import (
    ChannelResult,
    FusionResult,
    RerankResult,
    RetrievalResult,
    ScoredCandidate,
)


@dataclass
class ChannelOperationalMetrics:
    channel: str
    latency_ms: float
    num_candidates: int
    min_score: float = 0.0
    max_score: float = 0.0
    mean_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "latency_ms": round(self.latency_ms, 2),
            "num_candidates": self.num_candidates,
            "min_score": round(self.min_score, 4),
            "max_score": round(self.max_score, 4),
            "mean_score": round(self.mean_score, 4),
        }


@dataclass
class RetrievalOperationalMetrics:
    query: str
    intent: str
    active_channels: list[str]

    # Timing
    total_latency_ms: float = 0.0
    query_analysis_ms: float = 0.0
    channels_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    context_assembly_ms: float = 0.0

    # Counts
    total_candidates_pre_fusion: int = 0
    total_candidates_post_fusion: int = 0
    total_candidates_post_rerank: int = 0
    final_result_count: int = 0
    unique_documents: int = 0

    # Per-channel
    channel_metrics: list[ChannelOperationalMetrics] = field(default_factory=list)

    # Fusion analysis
    channel_contributions: dict[str, int] = field(default_factory=dict)
    channel_overlap_pairs: dict[str, int] = field(default_factory=dict)

    # Rerank analysis
    rerank_mean_rank_change: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "active_channels": self.active_channels,
            "timing": {
                "total_ms": round(self.total_latency_ms, 2),
                "query_analysis_ms": round(self.query_analysis_ms, 2),
                "channels_ms": round(self.channels_latency_ms, 2),
                "fusion_ms": round(self.fusion_latency_ms, 2),
                "rerank_ms": round(self.rerank_latency_ms, 2),
                "context_assembly_ms": round(self.context_assembly_ms, 2),
            },
            "counts": {
                "pre_fusion": self.total_candidates_pre_fusion,
                "post_fusion": self.total_candidates_post_fusion,
                "post_rerank": self.total_candidates_post_rerank,
                "final": self.final_result_count,
                "unique_documents": self.unique_documents,
            },
            "channels": [cm.to_dict() for cm in self.channel_metrics],
            "fusion_contributions": self.channel_contributions,
            "rerank_mean_rank_change": round(self.rerank_mean_rank_change, 2),
        }

    def to_mlflow_metrics(self, prefix: str = "retrieval") -> dict[str, float]:
        m: dict[str, float] = {
            f"{prefix}/total_latency_ms": self.total_latency_ms,
            f"{prefix}/query_analysis_ms": self.query_analysis_ms,
            f"{prefix}/channels_latency_ms": self.channels_latency_ms,
            f"{prefix}/fusion_latency_ms": self.fusion_latency_ms,
            f"{prefix}/rerank_latency_ms": self.rerank_latency_ms,
            f"{prefix}/pre_fusion_candidates": float(self.total_candidates_pre_fusion),
            f"{prefix}/post_fusion_candidates": float(self.total_candidates_post_fusion),
            f"{prefix}/final_results": float(self.final_result_count),
            f"{prefix}/unique_documents": float(self.unique_documents),
            f"{prefix}/rerank_mean_rank_change": self.rerank_mean_rank_change,
        }
        for cm in self.channel_metrics:
            m[f"{prefix}/channel/{cm.channel}/latency_ms"] = cm.latency_ms
            m[f"{prefix}/channel/{cm.channel}/candidates"] = float(cm.num_candidates)
        for ch, cnt in self.channel_contributions.items():
            m[f"{prefix}/contribution/{ch}"] = float(cnt)
        return m


def compute_operational_metrics(result: RetrievalResult) -> RetrievalOperationalMetrics:
    channel_metrics: list[ChannelOperationalMetrics] = []
    for cr in result.channel_results:
        scores = [c.score for c in cr.candidates]
        channel_metrics.append(
            ChannelOperationalMetrics(
                channel=cr.channel.value,
                latency_ms=cr.latency_ms,
                num_candidates=cr.num_candidates,
                min_score=min(scores) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
            )
        )

    total_pre_fusion = sum(cr.num_candidates for cr in result.channel_results)

    return RetrievalOperationalMetrics(
        query=result.query,
        intent=result.intent,
        active_channels=result.active_channels,
        total_latency_ms=result.total_latency_ms,
        query_analysis_ms=result.query_analysis_ms,
        channels_latency_ms=(
            max(cr.latency_ms for cr in result.channel_results) if result.channel_results else 0.0
        ),
        fusion_latency_ms=(result.fusion_result.latency_ms if result.fusion_result else 0.0),
        rerank_latency_ms=(result.rerank_result.latency_ms if result.rerank_result else 0.0),
        total_candidates_pre_fusion=total_pre_fusion,
        total_candidates_post_fusion=(
            len(result.fusion_result.candidates) if result.fusion_result else 0
        ),
        total_candidates_post_rerank=(
            len(result.rerank_result.candidates) if result.rerank_result else 0
        ),
        final_result_count=result.num_results,
        unique_documents=len(result.document_ids),
        channel_metrics=channel_metrics,
        channel_contributions=(
            result.fusion_result.channel_contributions if result.fusion_result else {}
        ),
        rerank_mean_rank_change=(
            result.rerank_result.rank_changes if result.rerank_result else 0.0
        ),
    )


@dataclass
class IRQualityMetrics:
    query: str
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])

    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    map_at_k: dict[int, float] = field(default_factory=dict)

    mrr: float = 0.0
    num_relevant: int = 0
    num_retrieved: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "query": self.query,
            "mrr": round(self.mrr, 4),
            "num_relevant": self.num_relevant,
            "num_retrieved": self.num_retrieved,
        }
        for k in self.k_values:
            d[f"recall@{k}"] = round(self.recall_at_k.get(k, 0.0), 4)
            d[f"precision@{k}"] = round(self.precision_at_k.get(k, 0.0), 4)
            d[f"ndcg@{k}"] = round(self.ndcg_at_k.get(k, 0.0), 4)
            d[f"hit_rate@{k}"] = round(self.hit_rate_at_k.get(k, 0.0), 4)
            d[f"map@{k}"] = round(self.map_at_k.get(k, 0.0), 4)
        return d

    def to_mlflow_metrics(self, prefix: str = "ir") -> dict[str, float]:
        m: dict[str, float] = {f"{prefix}/mrr": self.mrr}
        for k in self.k_values:
            m[f"{prefix}/recall@{k}"] = self.recall_at_k.get(k, 0.0)
            m[f"{prefix}/precision@{k}"] = self.precision_at_k.get(k, 0.0)
            m[f"{prefix}/ndcg@{k}"] = self.ndcg_at_k.get(k, 0.0)
            m[f"{prefix}/hit_rate@{k}"] = self.hit_rate_at_k.get(k, 0.0)
            m[f"{prefix}/map@{k}"] = self.map_at_k.get(k, 0.0)
        return m


def _dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def compute_ir_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    relevance_grades: dict[str, float] | None = None,
    k_values: list[int] | None = None,
    query: str = "",
) -> IRQualityMetrics:
    """
    Compute standard IR quality metrics.

    Args:
        retrieved_ids: Ordered list of chunk/document IDs from retrieval.
        relevant_ids: Set of ground-truth relevant IDs.
        relevance_grades: Optional graded relevance (ID → grade).
            If None, binary relevance (1.0 if in relevant_ids, else 0.0).
        k_values: Cutoff values for metrics.
        query: Query string (for logging).

    Returns:
        IRQualityMetrics with all metrics computed.
    """
    k_values = k_values or [1, 3, 5, 10]
    if relevance_grades is None:
        relevance_grades = {rid: 1.0 for rid in relevant_ids}

    binary_rels = [1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids]
    graded_rels = [relevance_grades.get(rid, 0.0) for rid in retrieved_ids]
    ideal_rels = sorted(relevance_grades.values(), reverse=True)

    metrics = IRQualityMetrics(
        query=query,
        k_values=k_values,
        num_relevant=len(relevant_ids),
        num_retrieved=len(retrieved_ids),
    )

    # MRR
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            metrics.mrr = 1.0 / (i + 1)
            break

    total_relevant = len(relevant_ids) if relevant_ids else 1

    for k in k_values:
        top_k_binary = binary_rels[:k]
        hits = sum(top_k_binary)

        # Recall@K
        metrics.recall_at_k[k] = hits / total_relevant

        # Precision@K
        metrics.precision_at_k[k] = hits / k if k > 0 else 0.0

        # Hit Rate@K
        metrics.hit_rate_at_k[k] = 1.0 if hits > 0 else 0.0

        # NDCG@K
        dcg = _dcg(graded_rels, k)
        idcg = _dcg(ideal_rels, k)
        metrics.ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0

        # MAP@K
        ap = 0.0
        running_hits = 0.0
        for i in range(min(k, len(binary_rels))):
            if binary_rels[i] > 0:
                running_hits += 1
                ap += running_hits / (i + 1)
        metrics.map_at_k[k] = ap / min(total_relevant, k) if total_relevant > 0 else 0.0

    return metrics


@dataclass
class AggregatedIRMetrics:
    num_queries: int = 0
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])

    mean_recall_at_k: dict[int, float] = field(default_factory=dict)
    mean_precision_at_k: dict[int, float] = field(default_factory=dict)
    mean_ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mean_hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    mean_map_at_k: dict[int, float] = field(default_factory=dict)
    mean_mrr: float = 0.0

    std_recall_at_k: dict[int, float] = field(default_factory=dict)
    std_ndcg_at_k: dict[int, float] = field(default_factory=dict)

    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "num_queries": self.num_queries,
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }
        for k in self.k_values:
            d[f"mean_recall@{k}"] = round(self.mean_recall_at_k.get(k, 0.0), 4)
            d[f"mean_precision@{k}"] = round(self.mean_precision_at_k.get(k, 0.0), 4)
            d[f"mean_ndcg@{k}"] = round(self.mean_ndcg_at_k.get(k, 0.0), 4)
            d[f"mean_hit_rate@{k}"] = round(self.mean_hit_rate_at_k.get(k, 0.0), 4)
            d[f"mean_map@{k}"] = round(self.mean_map_at_k.get(k, 0.0), 4)
        return d

    def to_mlflow_metrics(self, prefix: str = "ir_agg") -> dict[str, float]:
        m: dict[str, float] = {
            f"{prefix}/mean_mrr": self.mean_mrr,
            f"{prefix}/num_queries": float(self.num_queries),
            f"{prefix}/mean_latency_ms": self.mean_latency_ms,
            f"{prefix}/p95_latency_ms": self.p95_latency_ms,
        }
        for k in self.k_values:
            m[f"{prefix}/mean_recall@{k}"] = self.mean_recall_at_k.get(k, 0.0)
            m[f"{prefix}/mean_ndcg@{k}"] = self.mean_ndcg_at_k.get(k, 0.0)
            m[f"{prefix}/mean_precision@{k}"] = self.mean_precision_at_k.get(k, 0.0)
            m[f"{prefix}/mean_hit_rate@{k}"] = self.mean_hit_rate_at_k.get(k, 0.0)
            m[f"{prefix}/mean_map@{k}"] = self.mean_map_at_k.get(k, 0.0)
        return m


def aggregate_ir_metrics(
    per_query: list[IRQualityMetrics],
    latencies: list[float] | None = None,
) -> AggregatedIRMetrics:
    if not per_query:
        return AggregatedIRMetrics()

    n = len(per_query)
    k_values = per_query[0].k_values

    agg = AggregatedIRMetrics(num_queries=n, k_values=k_values)
    agg.mean_mrr = sum(q.mrr for q in per_query) / n

    for k in k_values:
        recalls = [q.recall_at_k.get(k, 0.0) for q in per_query]
        ndcgs = [q.ndcg_at_k.get(k, 0.0) for q in per_query]

        agg.mean_recall_at_k[k] = sum(recalls) / n
        agg.mean_precision_at_k[k] = sum(q.precision_at_k.get(k, 0.0) for q in per_query) / n
        agg.mean_ndcg_at_k[k] = sum(ndcgs) / n
        agg.mean_hit_rate_at_k[k] = sum(q.hit_rate_at_k.get(k, 0.0) for q in per_query) / n
        agg.mean_map_at_k[k] = sum(q.map_at_k.get(k, 0.0) for q in per_query) / n

        mean_r = agg.mean_recall_at_k[k]
        mean_n = agg.mean_ndcg_at_k[k]
        agg.std_recall_at_k[k] = (sum((r - mean_r) ** 2 for r in recalls) / n) ** 0.5
        agg.std_ndcg_at_k[k] = (sum((nd - mean_n) ** 2 for nd in ndcgs) / n) ** 0.5

    if latencies:
        agg.mean_latency_ms = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        agg.p95_latency_ms = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

    return agg
