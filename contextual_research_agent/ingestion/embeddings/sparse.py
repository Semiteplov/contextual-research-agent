from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from fastembed import SparseTextEmbedding

from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


class SparseEncoder:
    """
    BM25 sparse encoder using fastembed.

    Produces sparse vectors compatible with Qdrant SparseVector format.
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return

        logger.info("Loading sparse encoder: %s", self._model_name)
        self._model = SparseTextEmbedding(model_name=self._model_name)
        logger.info("Sparse encoder loaded: %s", self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_texts(
        self,
        texts: Sequence[str],
    ) -> list[dict[str, Any]]:
        self._ensure_model()
        assert self._model is not None

        results: list[dict[str, Any]] = []
        for sparse_vec in self._model.embed(list(texts)):
            results.append(
                {
                    "indices": sparse_vec.indices.tolist(),
                    "values": sparse_vec.values.tolist(),
                }
            )

        return results

    async def encode_texts_async(
        self,
        texts: Sequence[str],
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.encode_texts, texts)

    def encode_query(self, query: str) -> dict[str, Any]:
        results = self.encode_texts([query])
        return results[0]

    async def encode_query_async(self, query: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.encode_query, query)


def create_sparse_encoder(
    model_name: str = "Qdrant/bm25",
) -> SparseEncoder:
    return SparseEncoder(model_name=model_name)
