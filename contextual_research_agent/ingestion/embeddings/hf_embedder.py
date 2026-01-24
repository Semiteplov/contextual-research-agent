import asyncio
from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from contextual_research_agent.common import logging
from contextual_research_agent.ingestion.embeddings.base import Embedder

logger = logging.get_logger(__name__)


class HuggingFaceEmbedder(Embedder):
    def __init__(  # noqa: PLR0913
        self,
        model: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
        normalize: bool = True,
        query_instruction: str | None = None,
        passage_instruction: str | None = None,
        batch_size: int = 16,
    ):
        self._model_name = model
        self._normalize = normalize
        self._batch_size = batch_size

        self._query_instruction = query_instruction
        self._passage_instruction = passage_instruction

        if query_instruction is None and "bge" in model.lower():
            self._query_instruction = "Represent this sentence for searching relevant passages: "

        logger.info(f"Loading embedding model: {model}")
        self._model = SentenceTransformer(model, device=device)

        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"Embedding dimension is None for model={model}")
        self._dimension = dim

        logger.info(f"Loaded {model} (dim={self._dimension}, device={self._model.device})")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_batch_size(self) -> int:
        return self._batch_size

    def _apply_prefix(self, texts: Sequence[str], prefix: str | None = None) -> list[str]:
        if not prefix:
            return [t for t in texts]
        return [prefix + t for t in texts]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        texts_list = [t for t in texts if t and t.strip()]
        if not texts_list:
            return []

        texts_list = self._apply_prefix(texts_list, self._passage_instruction)

        def _encode():
            try:
                emb = self._model.encode(
                    texts_list,
                    batch_size=self.max_batch_size,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
                return emb.tolist()
            except Exception:
                logger.exception(
                    f"Embedding texts failed: model={self._model_name}"
                    f"n={len(texts_list)} batch_size={self.max_batch_size}",
                )
                raise

        return await asyncio.to_thread(_encode)

    async def embed_query(self, query: str) -> list[float]:
        query = (query or "").strip()
        if not query:
            return [0.0] * self.dimension

        if self._query_instruction:
            query = self._query_instruction + query

        def _encode():
            try:
                emb = self._model.encode(
                    [query],
                    batch_size=1,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
                vec = emb[0]
                return vec.tolist()
            except Exception:
                logger.exception(f"Embedding query failed: model={self._model_name}")
                raise

        return await asyncio.to_thread(_encode)

    async def embed_texts_batched(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        texts_list = [t for t in texts if t and t.strip()]
        if not texts_list:
            return []

        texts_list = self._apply_prefix(texts_list, self._passage_instruction)
        effective_batch_size = batch_size or self._batch_size

        def _encode():
            return self._model.encode(
                texts_list,
                batch_size=effective_batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=len(texts_list) > 100,  # noqa: PLR2004
            ).tolist()

        return await asyncio.to_thread(_encode)


def create_hf_embedder(  # noqa: PLR0913
    model: str = "Qwen/Qwen3-Embedding-4B",
    device: str | None = None,
    normalize: bool = True,
    query_instruction: str | None = None,
    passage_instruction: str | None = None,
    batch_size: int = 16,
) -> Embedder:
    return HuggingFaceEmbedder(
        model=model,
        device=device,
        normalize=normalize,
        query_instruction=query_instruction,
        passage_instruction=passage_instruction,
        batch_size=batch_size,
    )
