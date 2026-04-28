from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from contextual_research_agent.agent.tracing.trace import AgentTrace
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.common.settings import get_settings

logger = get_logger(__name__)


def log_agent_trace(
    trace: AgentTrace,
    experiment_name: str = "multi_agent",
    run_name: str | None = None,
) -> None:
    """Log a single agent trace to MLflow."""
    try:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(experiment_name)

        auto_run_name = run_name or f"query_{trace.resolved_mode}_{trace.complexity}"

        with mlflow.start_run(run_name=auto_run_name):
            # Parameters
            mlflow.log_params(
                {
                    "query": trace.query[:250],
                    "intent": trace.detected_intent,
                    "complexity": trace.complexity,
                    "resolved_mode": trace.resolved_mode,
                    "num_sub_queries": len(trace.sub_queries),
                    "retry_count": trace.retry_count,
                    "status": trace.status,
                }
            )

            # Metrics
            metrics: dict[str, float] = {
                "retrieval_latency_ms": trace.retrieval_latency_ms,
                "generation_latency_ms": trace.generation_latency_ms,
                "total_latency_ms": trace.total_latency_ms,
                "num_chunks_retrieved": len(trace.retrieved_chunks),
                "answer_length": len(trace.final_answer),
            }

            tokens = trace.generation_tokens
            if tokens:
                metrics["prompt_tokens"] = tokens.get("prompt", 0)
                metrics["completion_tokens"] = tokens.get("completion", 0)

            critic_fb = trace.critic_feedback
            if critic_fb:
                if critic_fb.get("faithfulness_score") is not None:
                    metrics["critic_faithfulness"] = critic_fb["faithfulness_score"]
                if critic_fb.get("completeness_score") is not None:
                    metrics["critic_completeness"] = critic_fb["completeness_score"]

            mlflow.log_metrics(metrics)

            # Artifact: full trace
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                prefix="agent_trace_",
            ) as f:
                json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)
                f.flush()
                mlflow.log_artifact(f.name)

    except ImportError:
        logger.debug("MLflow not available, skipping trace logging")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


def log_agent_batch(
    traces: list[AgentTrace],
    experiment_name: str = "multi_agent",
    run_name: str = "batch_eval",
) -> None:
    """Log aggregated metrics from a batch of agent traces."""
    try:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "num_queries": len(traces),
                }
            )

            if not traces:
                return

            # Aggregate metrics
            latencies = [t.total_latency_ms for t in traces]
            retrieval_lats = [t.retrieval_latency_ms for t in traces]
            gen_lats = [t.generation_latency_ms for t in traces]

            completed = [t for t in traces if t.status == "completed"]
            failed = [t for t in traces if t.status == "failed"]
            retried = [t for t in traces if t.retry_count > 0]

            mlflow.log_metrics(
                {
                    "mean_total_latency_ms": sum(latencies) / len(latencies),
                    "mean_retrieval_latency_ms": sum(retrieval_lats) / len(retrieval_lats),
                    "mean_generation_latency_ms": sum(gen_lats) / len(gen_lats),
                    "completed_count": len(completed),
                    "failed_count": len(failed),
                    "retry_count": len(retried),
                    "completion_rate": len(completed) / len(traces),
                }
            )

            # Complexity distribution
            complexity_counts: dict[str, int] = {}
            for t in traces:
                complexity_counts[t.complexity] = complexity_counts.get(t.complexity, 0) + 1
            for c, count in complexity_counts.items():
                mlflow.log_metric(f"complexity_{c}_count", count)

            # Save all traces as artifact
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                prefix="agent_batch_",
            ) as f:
                json.dump(
                    [t.to_dict() for t in traces],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
                f.flush()
                mlflow.log_artifact(f.name)

    except Exception as e:
        logger.warning("MLflow batch logging failed: %s", e)
