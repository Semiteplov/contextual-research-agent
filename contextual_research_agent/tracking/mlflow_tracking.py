from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from contextual_research_agent.agent.config import MLflowConfig
from contextual_research_agent.common import logging

logger = logging.get_logger(__name__)


class MLflowManager:
    def __init__(self, config: MLflowConfig | None = None):
        self._config = config or MLflowConfig()
        self._mlflow: Any | None = None
        self._experiment_id: str | None = None
        self._active_run: Any | None = None
        self._setup_complete = False

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled and self._mlflow is not None)

    @property
    def experiment_id(self) -> str | None:
        return self._experiment_id

    def setup(self) -> bool:
        if not self._config.enabled:
            logger.debug("MLflow disabled in config")
            return False

        if self._setup_complete:
            return True

        try:
            self._mlflow = mlflow

            mlflow.set_tracking_uri(self._config.tracking_uri)
            logger.info(f"MLflow tracking URI: {self._config.tracking_uri}")

            experiment = mlflow.get_experiment_by_name(self._config.experiment_name)

            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    name=self._config.experiment_name,
                    artifact_location=self._config.artifact_location,
                )
                logger.info(f"Created MLflow experiment: {self._config.experiment_name}")
            else:
                self._experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self._config.experiment_name}")

            mlflow.set_experiment(self._config.experiment_name)

            if self._config.autolog_enabled:
                self._setup_autolog()

            self._setup_complete = True
            return True

        except ImportError:
            logger.warning("mlflow package not installed, tracking disabled")
            self._config.enabled = False
            return False

        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self._config.enabled = False
            return False

    def _setup_autolog(self) -> None:
        if not self._mlflow:
            return

        try:
            mlflow.langchain.autolog(  # type: ignore
                disable_for_unsupported_versions=True,
                silent=False,
                log_traces=True,
            )
            logger.info("MLflow autolog enabled (mlflow.langchain) - traces will be recorded")

        except ImportError:
            logger.debug("mlflow.langchain not available, autolog disabled")
        except Exception as e:
            logger.warning(f"Failed to setup autolog: {e}")

    def disable_autolog(self) -> None:
        try:
            mlflow.langchain.autolog(disable=True)  # type: ignore
        except Exception:
            return

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
        description: str | None = None,
    ) -> Generator[Any, None, None]:
        if not self.enabled:
            yield None
            return

        try:
            with self._mlflow.start_run(  # type: ignore
                run_name=run_name,
                tags=tags,
                nested=nested,
                description=description,
            ) as run:
                self._active_run = run
                yield run
                self._active_run = None

        except Exception as e:
            logger.warning(f"MLflow run failed: {e}")
            yield None

    def get_active_run(self) -> Any | None:
        """Get the currently active run."""
        if self._mlflow:
            return self._mlflow.active_run()
        return None

    def get_run_id(self) -> str | None:
        """Get current run ID."""
        run = self.get_active_run()
        if run:
            return run.info.run_id
        return None

    def get_run_url(self) -> str | None:
        """Get URL to view current run in MLflow UI."""
        run = self.get_active_run()
        if not run:
            return None

        run_id = run.info.run_id
        tracking_uri = self._config.tracking_uri

        if tracking_uri.startswith("http"):
            return f"{tracking_uri}/#/experiments/{self._experiment_id}/runs/{run_id}"

        return None

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            str_params = {}
            for k, v in params.items():
                str_val = str(v)
                if len(str_val) > 500:  # noqa: PLR2004
                    str_val = str_val[:497] + "..."
                str_params[k] = str_val

            self._mlflow.log_params(str_params)  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to log params: {e}")

    def log_param(self, key: str, value: Any) -> None:
        self.log_params({key: value})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return

        try:
            self._mlflow.log_metrics(metrics, step=step)  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to log metrics: {e}")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return

        try:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to log artifact: {e}")

    def log_artifacts(self, dir_path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return

        try:
            self._mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to log artifacts: {e}")

    def log_dict(self, data: dict[str, Any], filename: str) -> None:
        if not self.enabled:
            return

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
            ) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name

            self._mlflow.log_artifact(temp_path)  # type: ignore
            Path(temp_path).unlink()
        except Exception as e:
            logger.debug(f"Failed to log dict: {e}")

    def log_text(self, text: str, filename: str) -> None:
        if not self.enabled:
            return

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=Path(filename).suffix or ".txt",
                delete=False,
            ) as f:
                f.write(text)
                temp_path = f.name

            self._mlflow.log_artifact(temp_path)  # type: ignore
            Path(temp_path).unlink()
        except Exception as e:
            logger.debug(f"Failed to log text: {e}")

    def set_tag(self, key: str, value: str) -> None:
        if not self.enabled:
            return

        try:
            self._mlflow.set_tag(key, value)  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to set tag: {e}")

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set multiple tags."""
        for key, value in tags.items():
            self.set_tag(key, value)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        **kwargs,
    ) -> None:
        if not self.enabled:
            return

        try:
            if hasattr(self._mlflow, "pyfunc"):
                self._mlflow.pyfunc.log_model(  # type: ignore
                    artifact_path=artifact_path,
                    python_model=model,
                    **kwargs,
                )
            else:
                logger.warning("Model logging not available")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")


@dataclass
class QueryRunData:
    query: str
    mode: str
    answer: str
    citations: list[str]

    num_chunks: int
    chunk_ids: list[str]
    scores: list[float]

    latency_retrieve_ms: float
    latency_generate_ms: float
    latency_total_ms: float

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    top_k: int
    llm_model: str
    embedding_model: str

    relevance_score: float | None = None
    faithfulness_score: float | None = None
    citation_accuracy: float | None = None

    def to_params(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
        }

    def to_metrics(self) -> dict[str, float]:
        metrics = {
            "latency_retrieve_ms": self.latency_retrieve_ms,
            "latency_generate_ms": self.latency_generate_ms,
            "latency_total_ms": self.latency_total_ms,
            "num_chunks": float(self.num_chunks),
            "num_citations": float(len(self.citations)),
            "prompt_tokens": float(self.prompt_tokens),
            "completion_tokens": float(self.completion_tokens),
            "total_tokens": float(self.total_tokens),
        }

        if self.relevance_score is not None:
            metrics["relevance_score"] = self.relevance_score
        if self.faithfulness_score is not None:
            metrics["faithfulness_score"] = self.faithfulness_score
        if self.citation_accuracy is not None:
            metrics["citation_accuracy"] = self.citation_accuracy

        return metrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationBatch:
    name: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    runs: list[QueryRunData] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.latency_total_ms for r in self.runs) / len(self.runs)

    @property
    def avg_relevance(self) -> float | None:
        scores = [r.relevance_score for r in self.runs if r.relevance_score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def avg_faithfulness(self) -> float | None:
        scores = [r.faithfulness_score for r in self.runs if r.faithfulness_score is not None]
        return sum(scores) / len(scores) if scores else None

    def to_aggregate_metrics(self) -> dict[str, float]:
        metrics = {
            "num_queries": float(len(self.runs)),
            "avg_latency_ms": self.avg_latency_ms,
        }

        if self.runs:
            metrics["avg_num_chunks"] = sum(r.num_chunks for r in self.runs) / len(self.runs)
            metrics["avg_tokens"] = sum(r.total_tokens for r in self.runs) / len(self.runs)

        if self.avg_relevance is not None:
            metrics["avg_relevance"] = self.avg_relevance
        if self.avg_faithfulness is not None:
            metrics["avg_faithfulness"] = self.avg_faithfulness

        return metrics


def log_query_run(
    manager: MLflowManager,
    data: QueryRunData,
    run_name: str | None = None,
) -> str | None:
    with manager.start_run(run_name=run_name or f"query_{data.mode}") as run:
        if run is None:
            return None

        manager.log_params(data.to_params())
        manager.log_metrics(data.to_metrics())
        manager.log_dict(data.to_dict(), "query_run.json")
        manager.log_text(data.answer, "answer.md")

        return manager.get_run_id()


def log_evaluation_batch(
    manager: MLflowManager,
    batch: EvaluationBatch,
) -> str | None:
    with manager.start_run(
        run_name=f"eval_{batch.name}",
        tags={"type": "evaluation_batch"},
    ) as parent_run:
        if parent_run is None:
            return None

        manager.log_params(batch.config)
        manager.log_metrics(batch.to_aggregate_metrics())

        for i, run_data in enumerate(batch.runs):
            with manager.start_run(
                run_name=f"query_{i}",
                nested=True,
            ):
                manager.log_params(run_data.to_params())
                manager.log_metrics(run_data.to_metrics())

        batch_data = {
            "name": batch.name,
            "timestamp": batch.timestamp,
            "config": batch.config,
            "aggregate_metrics": batch.to_aggregate_metrics(),
            "num_runs": len(batch.runs),
        }
        manager.log_dict(batch_data, "batch_summary.json")

        return manager.get_run_id()


def create_query_run_data(
    response: Any,  # AgentResponse
    config: dict[str, Any],
) -> QueryRunData:
    return QueryRunData(
        query=response.query,
        mode=response.mode.value,
        answer=response.answer,
        citations=response.citations,
        num_chunks=len(response.retrieval.chunks),
        chunk_ids=[rc.chunk.id for rc in response.retrieval.chunks],
        scores=[rc.score for rc in response.retrieval.chunks],
        latency_retrieve_ms=response.latency.get("retrieve_ms", 0),
        latency_generate_ms=response.latency.get("generate_ms", 0),
        latency_total_ms=response.latency.get("total_ms", 0),
        prompt_tokens=response.tokens.get("prompt", 0),
        completion_tokens=response.tokens.get("completion", 0),
        total_tokens=response.tokens.get("total", 0),
        top_k=response.retrieval.top_k,
        llm_model=config.get("llm", {}).get("model", "unknown"),
        embedding_model=config.get("embedding", {}).get("model", "unknown"),
    )


_manager: MLflowManager | None = None


def setup_mlflow(config: MLflowConfig | None = None) -> MLflowManager:
    global _manager
    _manager = MLflowManager(config)
    _manager.setup()
    return _manager


def get_mlflow_manager() -> MLflowManager:
    global _manager
    if _manager is None:
        _manager = MLflowManager()
    return _manager


def disable_mlflow() -> None:
    global _manager
    if _manager:
        _manager.disable_autolog()
        _manager._config.enabled = False
