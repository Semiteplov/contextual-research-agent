from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.ingestion.domain.entities import DoclingParserConfig
from contextual_research_agent.ingestion.parsers.metrics import (
    ChunkingMetrics,
    aggregate_corpus_metrics,
)
from contextual_research_agent.ingestion.pipeline import (
    BatchResult,
    IngestionMetrics,
    IngestionResult,
)

logger = get_logger(__name__)


class IngestionTracker:
    """MLflow tracker for ingestion experiments."""

    def __init__(
        self,
        experiment_name: str = "ingestion",
        tracking_uri: str | None = None,
    ):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self._experiment_name = experiment_name

    def log_single(
        self,
        result: IngestionResult,
        config: DoclingParserConfig,
        embedding_model: str = "",
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Log a single file ingestion as an MLflow run.

        Returns the MLflow run_id.
        """
        run_name = result.run_id

        with mlflow.start_run(run_name=run_name) as run:
            # Params
            self._log_config_params(config, embedding_model)
            mlflow.log_param("file_path", result.file_path)
            mlflow.log_param("mode", "single")

            # Tags
            mlflow.set_tag("status", result.status)
            if result.document_id:
                mlflow.set_tag("document_id", result.document_id)
            if tags:
                mlflow.set_tags(tags)

            # Metrics
            self._log_result_metrics(result)

            # Artifact: full result JSON
            self._log_json_artifact(result.to_dict(), "result.json")

            logger.info("Logged single ingestion to MLflow: %s", run.info.run_id)
            return run.info.run_id

    def log_batch(  # noqa: PLR0913
        self,
        batch: BatchResult,
        config: DoclingParserConfig,
        dataset_name: str = "",
        split: str | None = None,
        embedding_model: str = "",
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Log a batch ingestion as an MLflow run with aggregated metrics.

        Returns the MLflow run_id.
        """
        run_name = f"batch_{dataset_name}" if dataset_name else "batch"
        if split:
            run_name += f"_{split}"

        with mlflow.start_run(run_name=run_name) as run:
            # Params
            self._log_config_params(config, embedding_model)
            mlflow.log_param("mode", "batch")
            if dataset_name:
                mlflow.log_param("dataset_name", dataset_name)
            if split:
                mlflow.log_param("split", split)

            # Tags
            mlflow.set_tag("status", "complete")
            if tags:
                mlflow.set_tags(tags)

            # Batch-level metrics
            mlflow.log_metrics(
                {
                    "batch/total": batch.total,
                    "batch/succeeded": batch.succeeded,
                    "batch/failed": batch.failed,
                    "batch/success_rate": batch.succeeded / batch.total if batch.total else 0,
                    "batch/total_chunks": batch.total_chunks,
                    "batch/total_duration_ms": batch.total_duration_ms,
                    "batch/avg_duration_per_doc_ms": (
                        batch.total_duration_ms / batch.total if batch.total else 0
                    ),
                }
            )

            # Per-stage latency aggregation
            self._log_latency_aggregates(batch)

            # Chunking aggregation
            self._log_chunking_aggregates(batch)

            # Embedding aggregation
            self._log_embedding_aggregates(batch)

            # Artifacts
            self._log_json_artifact(batch.to_dict(), "batch_report.json")

            # Per-document metrics table
            self._log_per_doc_table(batch)

            logger.info("Logged batch ingestion to MLflow: %s", run.info.run_id)
            return run.info.run_id

    @staticmethod
    def _log_config_params(config: DoclingParserConfig, embedding_model: str) -> None:
        """Log parser/pipeline config as MLflow params."""
        mlflow.log_params(
            {
                "parser/max_tokens": config.max_tokens,
                "parser/embedding_model": config.embedding_model,
                "parser/merge_peers": config.merge_peers,
                "parser/include_context": config.include_context,
                "parser/do_ocr": config.do_ocr,
                "parser/do_table_structure": config.do_table_structure,
                "parser/min_chunk_tokens": config.min_chunk_tokens,
                "parser/filter_empty_chunks": config.filter_empty_chunks,
            }
        )
        if embedding_model:
            mlflow.log_param("embedding_model", embedding_model)

    @staticmethod
    def _log_result_metrics(result: IngestionResult) -> None:
        """Log metrics from a single IngestionResult."""
        m = result.metrics

        # Latency
        mlflow.log_metrics(
            {
                "latency/parse_ms": m.latency.parse_ms,
                "latency/chunk_ms": m.latency.chunk_ms,
                "latency/embed_ms": m.latency.embed_ms,
                "latency/index_ms": m.latency.index_ms,
                "latency/total_ms": m.latency.total_ms,
            }
        )

        mlflow.log_metric("chunk_count", result.chunk_count)

        # Chunking metrics
        if m.chunking:
            cm = m.chunking
            mlflow.log_metrics(
                {
                    "chunking/total_chunks": cm.total_chunks,
                    "chunking/mean_tokens": cm.mean_tokens,
                    "chunking/median_tokens": cm.median_tokens,
                    "chunking/std_tokens": cm.std_tokens,
                    "chunking/p5_tokens": cm.p5_tokens,
                    "chunking/p95_tokens": cm.p95_tokens,
                    "chunking/empty_count": cm.empty_chunk_count,
                    "chunking/oversized_count": cm.oversized_chunk_count,
                    "chunking/section_coverage": cm.section_coverage_ratio,
                    "chunking/context_overhead": cm.mean_context_overhead,
                }
            )

        # Embedding metrics
        if m.embedding:
            em = m.embedding
            mlflow.log_metrics(
                {
                    "embedding/duration_ms": em.duration_ms,
                    "embedding/throughput": em.throughput_texts_per_sec,
                    "embedding/empty_texts": em.empty_text_count,
                }
            )

        # Indexing metrics
        if m.indexing:
            mlflow.log_metrics(
                {
                    "indexing/duration_ms": m.indexing.duration_ms,
                    "indexing/num_items": m.indexing.num_items,
                }
            )

    @staticmethod
    def _log_latency_aggregates(batch: BatchResult) -> None:
        """Aggregate latency across successful results."""
        ok = [r for r in batch.results if r.success]
        if not ok:
            return

        stages = ["parse_ms", "chunk_ms", "embed_ms", "index_ms", "total_ms"]
        for stage in stages:
            values = [getattr(r.metrics.latency, stage) for r in ok]
            prefix = f"latency_agg/{stage.replace('_ms', '')}"
            mlflow.log_metrics(
                {
                    f"{prefix}_mean": sum(values) / len(values),
                    f"{prefix}_max": max(values),
                    f"{prefix}_min": min(values),
                }
            )

    @staticmethod
    def _log_chunking_aggregates(batch: BatchResult) -> None:
        """Aggregate chunking metrics using corpus-level aggregation."""
        chunking_metrics = [
            r.metrics.chunking
            for r in batch.results
            if r.success and r.metrics.chunking is not None
        ]
        if not chunking_metrics:
            return

        agg = aggregate_corpus_metrics(chunking_metrics)
        prefixed = {
            f"chunking_agg/{k}": float(v) for k, v in agg.items() if isinstance(v, (int, float))
        }
        if prefixed:
            mlflow.log_metrics(prefixed)

    @staticmethod
    def _log_embedding_aggregates(batch: BatchResult) -> None:
        """Aggregate embedding throughput."""
        emb_metrics = [
            r.metrics.embedding
            for r in batch.results
            if r.success and r.metrics.embedding is not None
        ]
        if not emb_metrics:
            return

        throughputs = [e.throughput_texts_per_sec for e in emb_metrics]
        total_texts = sum(e.num_texts for e in emb_metrics)
        total_ms = sum(e.duration_ms for e in emb_metrics)

        mlflow.log_metrics(
            {
                "embedding_agg/mean_throughput": sum(throughputs) / len(throughputs),
                "embedding_agg/total_texts": total_texts,
                "embedding_agg/total_ms": total_ms,
                "embedding_agg/overall_throughput": (
                    total_texts / (total_ms / 1000) if total_ms > 0 else 0
                ),
            }
        )

    @staticmethod
    def _log_json_artifact(data: dict[str, Any], filename: str) -> None:
        """Log a dict as a JSON artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            with Path.open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            mlflow.log_artifact(str(path))

    @staticmethod
    def _log_per_doc_table(batch: BatchResult) -> None:
        """
        Log per-document metrics as a CSV artifact for table comparison.
        """
        rows = []
        for r in batch.results:
            row = {
                "file_path": r.file_path,
                "document_id": r.document_id or "",
                "status": r.status,
                "chunk_count": r.chunk_count,
                "parse_ms": round(r.metrics.latency.parse_ms, 1),
                "chunk_ms": round(r.metrics.latency.chunk_ms, 1),
                "embed_ms": round(r.metrics.latency.embed_ms, 1),
                "index_ms": round(r.metrics.latency.index_ms, 1),
                "total_ms": round(r.metrics.latency.total_ms, 1),
                "error": r.error or "",
            }
            if r.metrics.chunking:
                row["mean_tokens"] = round(r.metrics.chunking.mean_tokens, 1)
                row["oversized"] = r.metrics.chunking.oversized_chunk_count
                row["empty"] = r.metrics.chunking.empty_chunk_count
                row["section_coverage"] = round(r.metrics.chunking.section_coverage_ratio, 3)
            rows.append(row)

        if not rows:
            return

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_document_metrics.csv"
            headers = list(rows[0].keys())

            with Path.open(path, "w") as f:
                f.write(",".join(headers) + "\n")
                for row in rows:
                    values = [str(row.get(h, "")) for h in headers]
                    f.write(",".join(values) + "\n")
            mlflow.log_artifact(str(path))
