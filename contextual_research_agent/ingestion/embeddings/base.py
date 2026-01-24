from abc import ABC, abstractmethod
from collections.abc import Sequence


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    def max_batch_size(self) -> int:
        return 32

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (documents/chunks).

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors, same order as input.
        """
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text.

        Returns:
            Embedding vector.
        """
        ...

    async def embed_texts_batched(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Embed texts in batches.

        Args:
            texts: Texts to embed.
            batch_size: Batch size (uses max_batch_size if None).

        Returns:
            List of embedding vectors.
        """
        batch_size = batch_size or self.max_batch_size
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
