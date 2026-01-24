import asyncio
import hashlib
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from contextual_research_agent.common import logging
from contextual_research_agent.common.settings import get_settings
from contextual_research_agent.ingestion.domain.entities import Chunk

logger = logging.get_logger(__name__)


class QdrantStore:
    def __init__(
        self,
        embedding_dim: int = 1024,
        distance: str = "cosine",
        on_disk: bool = False,
    ):
        settings = get_settings()

        self.collection_name = settings.qdrant.collection_name
        self.embedding_dim = embedding_dim
        self.distance = distance
        self._on_disk = on_disk

        self._client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except (UnexpectedResponse, Exception):
            distance_map = {
                "cosine": models.Distance.COSINE,
                "euclid": models.Distance.EUCLID,
                "dot": models.Distance.DOT,
            }

            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=distance_map.get(self.distance, models.Distance.COSINE),
                    on_disk=self._on_disk,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,
                ),
            )

            self._create_payload_indexes()

            logger.info(
                f"Created collection: {self.collection_name} "
                f"(dim={self.embedding_dim}, distance={self.distance})"
            )

    def _create_payload_indexes(self) -> None:
        indexes = [
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("chunk_id", models.PayloadSchemaType.KEYWORD),
            ("section", models.PayloadSchemaType.KEYWORD),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
        ]

        for field_name, field_schema in indexes:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as e:
                logger.debug(f"Index creation for {field_name}: {e}")

    async def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
        batch_size: int = 128,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        if not chunks:
            return 0

        points: list[models.PointStruct] = []
        for chunk, vec in zip(chunks, embeddings, strict=True):
            if len(vec) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dim mismatch for chunk={chunk.id}: "
                    f"got {len(vec)}, expected {self.embedding_dim}"
                )

            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "page_numbers": chunk.page_numbers,
                "metadata": chunk.metadata,
            }

            point_id = self._chunk_id_to_uuid(chunk.id)

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        def _upsert_batches() -> None:
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )

        await asyncio.to_thread(_upsert_batches)

        logger.info("Added/updated %d chunks in %s", len(chunks), self.collection_name)
        return len(chunks)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                "Query embedding dim mismatch: "
                f"got {len(query_embedding)}, expected {self.embedding_dim}"
            )

        qdrant_filter = self._build_filter(filters) if filters else None

        def _search():
            return self._client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        response = await asyncio.to_thread(_search)

        output: list[tuple[Chunk, float]] = []

        for point in response.points:
            payload: dict[str, Any] = point.payload or {}
            score: float = point.score if point.score is not None else 0.0

            chunk = Chunk(
                id=str(payload.get("chunk_id", "")),
                document_id=str(payload.get("document_id", "")),
                text=str(payload.get("text", "")),
                token_count=int(payload.get("token_count") or 0),
                chunk_index=int(payload.get("chunk_index") or 0),
                section=str(payload.get("section") or ""),
                page_numbers=list(payload.get("page_numbers") or []),
                metadata=dict(payload.get("metadata") or {}),
            )
            output.append((chunk, float(score)))

        return output

    async def delete_by_document(self, document_id: str) -> int:
        def _delete():
            count_result = self._client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            count_before = count_result.count

            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )

            return count_before

        count = await asyncio.to_thread(_delete)
        logger.info(f"Deleted {count} chunks for document {document_id}")
        return count

    async def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []

        point_ids = [self._chunk_id_to_uuid(cid) for cid in chunk_ids]

        def _retrieve():
            return self._client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=True,
            )

        results = await asyncio.to_thread(_retrieve)

        chunks = []
        for point in results:
            payload = point.payload or {}
            chunk = Chunk(
                id=str(payload.get("chunk_id", "")),
                document_id=str(payload.get("document_id", "")),
                text=str(payload.get("text", "")),
                token_count=int(payload.get("token_count") or 0),
                chunk_index=int(payload.get("chunk_index") or 0),
                section=str(payload.get("section") or ""),
                page_numbers=list(payload.get("page_numbers") or []),
                metadata=dict(payload.get("metadata") or {}),
            )
            chunks.append(chunk)

        return chunks

    async def get_stats(self) -> dict[str, Any]:
        def _get_stats():
            try:
                info = self._client.get_collection(self.collection_name)
                return {
                    "collection_name": self.collection_name,
                    "points_count": info.points_count,
                    "indexed_vectors_count": info.indexed_vectors_count,
                    "status": info.status.value if info.status else "unknown",
                    "embedding_dim": self.embedding_dim,
                    "distance": self.distance,
                }
            except UnexpectedResponse:
                return {
                    "collection_name": self.collection_name,
                    "error": "Collection not found",
                }
            except Exception as e:
                return {
                    "collection_name": self.collection_name,
                    "error": str(e),
                }

        return await asyncio.to_thread(_get_stats)

    async def collection_exists(self) -> bool:
        def _check():
            try:
                self._client.get_collection(self.collection_name)
                return True
            except UnexpectedResponse:
                return False

        return await asyncio.to_thread(_check)

    async def delete_collection(self) -> bool:
        def _delete():
            try:
                self._client.delete_collection(self.collection_name)
                return True
            except Exception:
                return False

        result = await asyncio.to_thread(_delete)
        if result:
            logger.info(f"Deleted collection: {self.collection_name}")
        return result

    async def close(self) -> None:
        self._client.close()

    def _build_filter(self, filters: dict[str, Any]) -> models.Filter | None:
        conditions: list[models.Condition] = []

        for key, value in filters.items():
            if value is None:
                continue

            if isinstance(value, (list, tuple, set)):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=list(value)),
                    )
                )
            elif isinstance(value, dict):
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                                gt=value.get("gt"),
                                lt=value.get("lt"),
                            ),
                        )
                    )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        hash_bytes = hashlib.sha256(chunk_id.encode()).digest()[:16]
        return str(
            uuid4().__class__(
                bytes=hash_bytes,
                version=4,
            )
        )


def create_qdrant_store(
    embedding_dim: int = 1024, distance: str = "cosine", on_disk: bool = False
) -> QdrantStore:
    return QdrantStore(embedding_dim=embedding_dim, distance=distance, on_disk=on_disk)
