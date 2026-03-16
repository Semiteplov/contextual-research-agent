from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.retrieval.config import RetrievalConfig
from contextual_research_agent.retrieval.metrics import (
    AggregatedIRMetrics,
    IRQualityMetrics,
    RetrievalOperationalMetrics,
)

logger = get_logger(__name__)


class RetrievalTracker:
    def __init__(
        self,
        experiment_name: str = "retrieval",
        tracking_uri: str | None = None,
    ):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self._experiment_name = experiment_name

    def log_evaluation(  # noqa: PLR0913
        self,
        config: RetrievalConfig,
        agg_metrics: AggregatedIRMetrics,
        per_query: list[IRQualityMetrics] | None = None,
        operational: list[RetrievalOperationalMetrics] | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        eval_set_path: str | None = None,
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(config.to_mlflow_params())

            mlflow.set_tag("pipeline", "retrieval")
            mlflow.set_tag("channels", ",".join(ch.value for ch in config.active_channels()))
            if tags:
                mlflow.set_tags(tags)
            if eval_set_path:
                mlflow.set_tag("eval_set", eval_set_path)

            mlflow.log_metrics(agg_metrics.to_mlflow_metrics())

            if operational:
                latencies = [o.total_latency_ms for o in operational]
                mlflow.log_metrics(
                    {
                        "ops/mean_latency_ms": sum(latencies) / len(latencies),
                        "ops/max_latency_ms": max(latencies),
                        "ops/mean_channels_ms": (
                            sum(o.channels_latency_ms for o in operational) / len(operational)
                        ),
                        "ops/mean_rerank_ms": (
                            sum(o.rerank_latency_ms for o in operational) / len(operational)
                        ),
                        "ops/mean_unique_docs": (
                            sum(o.unique_documents for o in operational) / len(operational)
                        ),
                    }
                )

            if per_query:
                self._log_json_artifact(
                    [m.to_dict() for m in per_query],
                    "per_query_metrics.json",
                )

            if operational:
                self._log_json_artifact(
                    [o.to_dict() for o in operational],
                    "operational_metrics.json",
                )

            summary = {
                "config": config.to_mlflow_params(),
                "aggregated": agg_metrics.to_dict(),
                "num_queries": agg_metrics.num_queries,
            }
            self._log_json_artifact(summary, "evaluation_summary.json")

            logger.info(
                "Logged retrieval evaluation to MLflow: %s (MRR=%.3f, R@5=%.3f)",
                run.info.run_id,
                agg_metrics.mean_mrr,
                agg_metrics.mean_recall_at_k.get(5, 0),
            )
            return run.info.run_id

    def log_single_query(
        self,
        config: RetrievalConfig,
        ir_metrics: IRQualityMetrics,
        operational: RetrievalOperationalMetrics | None = None,
        run_name: str | None = None,
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(config.to_mlflow_params())
            mlflow.log_metrics(ir_metrics.to_mlflow_metrics())

            if operational:
                mlflow.log_metrics(operational.to_mlflow_metrics())

            return run.info.run_id

    @staticmethod
    def _log_json_artifact(data: Any, filename: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            with Path.open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            mlflow.log_artifact(str(path))
