import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sentence_transformers import SentenceTransformer

from contextual_research_agent.common import logging
from contextual_research_agent.ingestion.embeddings.base import Embedder
from contextual_research_agent.ingestion.embeddings.metrics import EmbeddingMetrics

logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class ModelInstructionConfig:
    """
    Instruction / prompt configuration for a specific model family.

    Two mechanisms are supported:
      1. prompt_name: uses SentenceTransformer's built-in prompt templates
         (preferred, model defines the template).
      2. prefix: manual string prepended to text
         (fallback for models without prompt support).
    """

    query_prompt_name: str | None = None
    passage_prompt_name: str | None = None
    query_prefix: str | None = None
    passage_prefix: str | None = None


_MODEL_CONFIGS: dict[str, ModelInstructionConfig] = {
    "bge": ModelInstructionConfig(
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "bge-m3": ModelInstructionConfig(),
    "e5": ModelInstructionConfig(
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "qwen3-embedding": ModelInstructionConfig(
        query_prompt_name="query",
        passage_prompt_name="document",
    ),
    "bge-large-en": ModelInstructionConfig(
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}


def _resolve_instruction_config(
    model_name: str,
    query_instruction: str | None,
    passage_instruction: str | None,
) -> ModelInstructionConfig:
    """
    Resolve instruction config: explicit overrides > auto-detection > no instructions.
    """
    if query_instruction is not None or passage_instruction is not None:
        return ModelInstructionConfig(
            query_prefix=query_instruction,
            passage_prefix=passage_instruction,
        )

    model_lower = model_name.lower()
    for key, config in _MODEL_CONFIGS.items():
        if key in model_lower:
            return config

    return ModelInstructionConfig()


_EMPTY_PLACEHOLDER = "[empty]"


def _sanitize_texts(texts: Sequence[str]) -> tuple[list[str], int]:
    """
    Replace empty/whitespace texts with placeholder to preserve index alignment.
    Returns (sanitized_texts, empty_count).
    """
    sanitized = []
    empty_count = 0
    for t in texts:
        if not t or not t.strip():
            sanitized.append(_EMPTY_PLACEHOLDER)
            empty_count += 1
        else:
            sanitized.append(t)
    return sanitized, empty_count


class HuggingFaceEmbedder(Embedder):
    def __init__(  # noqa: PLR0913
        self,
        model: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
        normalize: bool = True,
        query_instruction: str | None = None,
        passage_instruction: str | None = None,
        batch_size: int = 32,
    ):
        self._model_name = model
        self._normalize = normalize
        self._batch_size = batch_size

        self._instructions = _resolve_instruction_config(
            model,
            query_instruction,
            passage_instruction,
        )

        logger.info(f"Loading embedding model: {model}")
        self._model = SentenceTransformer(
            model,
            device=device,
            model_kwargs={"torch_dtype": "float16"},
        )

        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"Cannot determine embedding dimension for model={model}")
        self._dimension = int(dim)

        self._supports_prompts = bool(getattr(self._model, "prompts", None))

        logger.info(
            f"Loaded {model} (dim={self._dimension}, "
            "device={self._model.device}, prompts={self._supports_prompts})"
        )
        self._metrics_log: list[EmbeddingMetrics] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    async def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Embed passage texts. Returns embeddings aligned 1:1 with input.

        Empty/whitespace texts get a placeholder embedding (not filtered).
        """
        if not texts:
            return []

        sanitized, empty_count = _sanitize_texts(texts)
        effective_batch = batch_size or self._batch_size

        embeddings, metrics = await self._encode(
            texts=sanitized,
            batch_size=effective_batch,
            prompt_name=self._instructions.passage_prompt_name,
            prefix=self._instructions.passage_prefix,
            operation="embed_passages",
            empty_count=empty_count,
        )
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query text.

        Raises ValueError for empty queries instead of returning zero vector.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("Cannot embed empty query")

        embeddings, metrics = await self._encode(
            texts=[query],
            batch_size=1,
            prompt_name=self._instructions.query_prompt_name,
            prefix=self._instructions.query_prefix,
            operation="embed_query",
            empty_count=0,
        )
        return embeddings[0]

    async def _encode(  # noqa: PLR0913
        self,
        texts: list[str],
        batch_size: int,
        prompt_name: str | None,
        prefix: str | None,
        operation: str,
        empty_count: int,
    ) -> tuple[list[list[float]], EmbeddingMetrics]:
        """
        Unified encode path. Handles prompt_name vs prefix routing.

        Returns (embeddings, metrics).
        """
        encode_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "normalize_embeddings": self._normalize,
            "show_progress_bar": len(texts) > 100,  # noqa: PLR2004
        }

        # Prompt routing: prompt_name takes priority over prefix
        if prompt_name and self._supports_prompts:
            encode_kwargs["prompt_name"] = prompt_name
        elif prefix:
            texts = [prefix + t for t in texts]

        # Text length stats
        text_lengths = [len(t) for t in texts]
        mean_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        max_len = max(text_lengths) if text_lengths else 0

        t0 = time.monotonic()
        error_msg = None
        success = True
        result: list[list[float]] = []

        try:

            def _do_encode():
                emb = self._model.encode(texts, **encode_kwargs)
                return emb.tolist()

            result = await asyncio.to_thread(_do_encode)

        except Exception as e:
            success = False
            error_msg = str(e)
            logger.exception(
                "Embedding failed",
                extra={
                    "model": self._model_name,
                    "operation": operation,
                    "num_texts": len(texts),
                    "batch_size": batch_size,
                },
            )
            raise

        finally:
            duration_ms = (time.monotonic() - t0) * 1000
            throughput = (len(texts) / (duration_ms / 1000)) if duration_ms > 0 else 0

            metrics = EmbeddingMetrics(
                operation=operation,
                model_name=self._model_name,
                num_texts=len(texts),
                batch_size=batch_size,
                duration_ms=duration_ms,
                throughput_texts_per_sec=throughput,
                dimension=self._dimension,
                success=success,
                error=error_msg,
                mean_text_length=mean_len,
                max_text_length=max_len,
                empty_text_count=empty_count,
            )
            self._metrics_log.append(metrics)

            if success:
                logger.debug(
                    "Embedding complete",
                    extra=metrics.to_dict(),
                )

        return result, metrics

    def get_metrics_log(self) -> list[EmbeddingMetrics]:
        return list(self._metrics_log)

    def clear_metrics_log(self) -> None:
        self._metrics_log.clear()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Aggregate metrics since last clear."""
        if not self._metrics_log:
            return {"total_operations": 0}

        total_texts = sum(m.num_texts for m in self._metrics_log)
        total_duration = sum(m.duration_ms for m in self._metrics_log)
        total_empty = sum(m.empty_text_count for m in self._metrics_log)

        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "total_operations": len(self._metrics_log),
            "total_texts_embedded": total_texts,
            "total_duration_ms": round(total_duration, 2),
            "overall_throughput_texts_per_sec": (
                round(total_texts / (total_duration / 1000), 2) if total_duration > 0 else 0
            ),
            "mean_duration_per_op_ms": round(
                total_duration / len(self._metrics_log),
                2,
            ),
            "total_empty_texts": total_empty,
            "error_count": sum(1 for m in self._metrics_log if not m.success),
        }


def create_hf_embedder(  # noqa: PLR0913
    model: str = "Qwen/Qwen3-Embedding-4B",
    device: str | None = None,
    normalize: bool = True,
    query_instruction: str | None = None,
    passage_instruction: str | None = None,
    batch_size: int = 32,
) -> HuggingFaceEmbedder:
    """
    Factory for HuggingFaceEmbedder.

    Default model: Qwen/Qwen3-Embedding-4B.
    For Qwen3-Embedding: prompts are auto-configured via prompt_name.
    For BGE: query instruction is auto-configured.
    For E5: both query and passage instructions are auto-configured.

    Override query_instruction / passage_instruction to disable auto-detection.
    """
    return HuggingFaceEmbedder(
        model=model,
        device=device,
        normalize=normalize,
        query_instruction=query_instruction,
        passage_instruction=passage_instruction,
        batch_size=batch_size,
    )
