import asyncio
import hashlib
import time
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from contextual_research_agent.common import logging
from contextual_research_agent.common.settings import get_settings
from contextual_research_agent.ingestion.domain.entities import Chunk, ChunkType
from contextual_research_agent.ingestion.vectorstores.metrics import (
    SearchMetrics,
    StoreOperationMetrics,
)

logger = logging.get_logger(__name__)


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    """
    Serialize Chunk → Qdrant payload dict.

    Single source of truth for which fields are stored.
    All typed fields from Chunk are stored at top level for indexing.
    """
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "chunk_type": chunk.chunk_type.value,
        "chunk_index": chunk.chunk_index,
        "section": chunk.section,
        "page_numbers": chunk.page_numbers,
        "is_contextualized": chunk.is_contextualized,
        "is_fallback": chunk.is_fallback,
        "is_degenerate": chunk.is_degenerate,
        "is_oversized": chunk.is_oversized,
        "metadata": chunk.metadata,
    }


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """
    Deserialize Qdrant payload → Chunk.

    Handles missing fields gracefully for backward compatibility
    with payloads written before schema changes.
    """
    chunk_type_str = payload.get("chunk_type", "text")
    try:
        chunk_type = ChunkType(chunk_type_str)
    except ValueError:
        chunk_type = ChunkType.TEXT

    return Chunk(
        id=str(payload.get("chunk_id", "")),
        document_id=str(payload.get("document_id", "")),
        text=str(payload.get("text", "")),
        token_count=int(payload.get("token_count") or 0),
        chunk_type=chunk_type,
        chunk_index=int(payload.get("chunk_index") or 0),
        section=str(payload.get("section") or ""),
        page_numbers=list(payload.get("page_numbers") or []),
        is_contextualized=bool(payload.get("is_contextualized", False)),
        is_fallback=bool(payload.get("is_fallback", False)),
        is_degenerate=bool(payload.get("is_degenerate", False)),
        is_oversized=bool(payload.get("is_oversized", False)),
        metadata=dict(payload.get("metadata") or {}),
    )


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id for idempotent upserts."""
    hash_bytes = hashlib.sha256(chunk_id.encode()).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes, version=4))


_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 0.5


async def _retry_async(
    fn,
    *,
    retries: int = _DEFAULT_RETRIES,
    backoff: float = _DEFAULT_BACKOFF,
    operation_name: str = "",
):
    """
    Retry an async-offloaded sync function with exponential backoff.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(fn)
        except UnexpectedResponse:
            raise
        except Exception as e:
            last_exc = e
            wait = backoff * (2**attempt)
            logger.warning(
                "Retry %d/%d for %s: %s (backoff %.1fs)",
                attempt + 1,
                retries,
                operation_name,
                e,
                wait,
            )
            await asyncio.sleep(wait)

    raise last_exc  # type: ignore[misc]


_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "euclid": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}

_PAYLOAD_INDEXES: list[tuple[str, models.PayloadSchemaType]] = [
    ("document_id", models.PayloadSchemaType.KEYWORD),
    ("chunk_id", models.PayloadSchemaType.KEYWORD),
    ("section", models.PayloadSchemaType.KEYWORD),
    ("chunk_type", models.PayloadSchemaType.KEYWORD),
    ("chunk_index", models.PayloadSchemaType.INTEGER),
    ("is_fallback", models.PayloadSchemaType.BOOL),
    ("is_degenerate", models.PayloadSchemaType.BOOL),
]


class QdrantStore:
    """
    Async wrapper around Qdrant for chunk vector storage.

    Prefer `QdrantStore.create()` over direct `__init__` to avoid
    blocking the event loop during collection initialization.
    """

    def __init__(  # noqa: PLR0913
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_dim: int,
        distance: str = "cosine",
        sparse_vector_name: str | None = None,
        dense_vector_name: str | None = None,
    ):
        self._client = client
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance = distance
        self._sparse_name = sparse_vector_name
        self._dense_name = dense_vector_name

        self._metrics_log: list[StoreOperationMetrics] = []

    @classmethod
    async def create(  # noqa: PLR0913
        cls,
        collection_name: str = "documents",
        embedding_dim: int = 1024,
        distance: str = "cosine",
        on_disk: bool = False,
        sparse_vector_name: str | None = None,
        dense_vector_name: str | None = None,
    ):
        """
        Async factory: creates client, ensures collection exists.

        Usage:
            store = await QdrantStore.create(collection_name="papers")
        """
        settings = get_settings()
        client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)

        detected_sparse = sparse_vector_name
        detected_dense = dense_vector_name

        try:
            info = client.get_collection(collection_name)
            sparse_cfg = info.config.params.sparse_vectors
            if sparse_cfg and len(sparse_cfg) > 0:
                detected_sparse = detected_sparse or next(iter(sparse_cfg))
                vectors_cfg = info.config.params.vectors
                if isinstance(vectors_cfg, dict) and len(vectors_cfg) > 0:
                    detected_dense = detected_dense or next(iter(vectors_cfg))
        except Exception:
            pass

        instance = cls(
            client=client,
            collection_name=collection_name,
            embedding_dim=embedding_dim,
            distance=distance,
            sparse_vector_name=detected_sparse,
            dense_vector_name=detected_dense,
        )

        await instance._ensure_collection(on_disk=on_disk)
        return instance

    async def _ensure_collection(self, on_disk: bool = False) -> None:
        """Create collection if it doesn't exist. Idempotent."""

        def _check_and_create():
            try:
                self._client.get_collection(self.collection_name)
                logger.info("Using existing collection: %s", self.collection_name)
                return
            except (UnexpectedResponse, Exception):
                pass

            qdrant_distance = _DISTANCE_MAP.get(self.distance, models.Distance.COSINE)

            if self._sparse_name:
                vectors_config = {
                    (self._dense_name or "dense"): models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_distance,
                        on_disk=on_disk,
                    ),
                }
                sparse_config = {
                    self._sparse_name: models.SparseVectorParams(),
                }
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_config,
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,
                    ),
                )
                logger.info(
                    "Created hybrid collection: %s (dim=%d, sparse=%s)",
                    self.collection_name,
                    self.embedding_dim,
                    self._sparse_name,
                )
            else:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_distance,
                        on_disk=on_disk,
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,
                    ),
                )
                logger.info(
                    "Created collection: %s (dim=%d, distance=%s)",
                    self.collection_name,
                    self.embedding_dim,
                    self.distance,
                )

        await asyncio.to_thread(_check_and_create)
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for filtered search. Idempotent."""

        def _create_indexes():
            for field_name, field_schema in _PAYLOAD_INDEXES:
                try:
                    self._client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
                except Exception as e:
                    logger.debug("Index for %s: %s", field_name, e)

        await asyncio.to_thread(_create_indexes)

    async def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
        sparse_vectors: Sequence[dict[str, Any]] | None = None,
        batch_size: int = 128,
    ) -> tuple[int, StoreOperationMetrics]:
        """
        Upsert chunks with embeddings. Returns (count, metrics).

        Idempotent: same chunk_id → same point_id → overwrite.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise ValueError(
                f"Chunks ({len(chunks)}) and sparse_vectors ({len(sparse_vectors)}) count mismatch"
            )

        if not chunks:
            metrics = StoreOperationMetrics(
                operation="upsert",
                collection=self.collection_name,
                duration_ms=0,
                num_items=0,
                success=True,
            )
            return 0, metrics

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings, strict=True)):
            if len(vec) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dim mismatch at index {i} (chunk={chunk.id}): "
                    f"got {len(vec)}, expected {self.embedding_dim}"
                )

        points: list[models.PointStruct] = []
        for i, (chunk, dense_vec) in enumerate(zip(chunks, embeddings, strict=True)):
            point_id = _chunk_id_to_point_id(chunk.id)
            payload = _chunk_to_payload(chunk)

            if self._sparse_name and sparse_vectors is not None:
                sv = sparse_vectors[i]
                vector: dict[str, models.Vector | models.SparseVector] = {
                    (self._dense_name or "dense"): dense_vec,
                    self._sparse_name: models.SparseVector(
                        indices=sv["indices"],
                        values=sv["values"],
                    ),
                }
                points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
            elif self._sparse_name:
                named_vector: dict[str, models.Vector] = {
                    (self._dense_name or "dense"): dense_vec,
                }
                points.append(models.PointStruct(id=point_id, vector=named_vector, payload=payload))
            else:
                points.append(models.PointStruct(id=point_id, vector=dense_vec, payload=payload))

        t0 = time.monotonic()
        error_msg = None
        success = True

        try:

            def _upsert_batches():
                for i in range(0, len(points), batch_size):
                    batch = points[i : i + batch_size]
                    self._client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True,
                    )

            await _retry_async(
                _upsert_batches,
                operation_name=f"upsert({len(points)} points)",
            )

        except Exception as e:
            success = False
            error_msg = str(e)
            raise

        finally:
            duration = (time.monotonic() - t0) * 1000
            op_metrics = StoreOperationMetrics(
                operation="upsert",
                collection=self.collection_name,
                duration_ms=duration,
                num_items=len(points) if success else 0,
                success=success,
                error=error_msg,
            )
            self._metrics_log.append(op_metrics)
            logger.info(
                "Upsert complete",
                extra=op_metrics.to_dict(),
            )

        return len(points), op_metrics

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[Chunk, float]], SearchMetrics]:
        """
        Vector similarity search. Returns (results, metrics).

        Args:
            query_embedding: Dense vector from embedding model.
            top_k: Maximum results to return.
            score_threshold: Minimum similarity score (None = no threshold).
            filters: Payload field filters (see _build_filter).
        """
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dim mismatch: "
                f"got {len(query_embedding)}, expected {self.embedding_dim}"
            )

        qdrant_filter = _build_filter(filters) if filters else None

        t0 = time.monotonic()
        error_msg = None
        success = True
        results: list[tuple[Chunk, float]] = []

        using = (self._dense_name or "dense") if self._sparse_name else None
        try:

            def _search():
                kwargs: dict[str, Any] = {
                    "collection_name": self.collection_name,
                    "query": query_embedding,
                    "limit": top_k,
                    "score_threshold": score_threshold,
                    "query_filter": qdrant_filter,
                    "with_payload": True,
                }
                if using:
                    kwargs["using"] = using
                return self._client.query_points(**kwargs)

            response = await _retry_async(
                _search,
                operation_name=f"search(top_k={top_k})",
            )

            for point in response.points:
                payload = point.payload or {}
                score = float(point.score) if point.score is not None else 0.0
                chunk = _payload_to_chunk(payload)
                results.append((chunk, score))

        except Exception as e:
            success = False
            error_msg = str(e)
            raise

        finally:
            duration = (time.monotonic() - t0) * 1000
            scores = [s for _, s in results]

            search_metrics = SearchMetrics(
                operation="search",
                collection=self.collection_name,
                duration_ms=duration,
                num_items=len(results),
                success=success,
                error=error_msg,
                top_k_requested=top_k,
                results_returned=len(results),
                min_score=min(scores) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                above_threshold_count=(
                    sum(1 for s in scores if s >= score_threshold)
                    if score_threshold is not None
                    else len(scores)
                ),
            )
            self._metrics_log.append(search_metrics)

        return results, search_metrics

    async def search_sparse(
        self,
        sparse_vector: dict[str, Any],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[Chunk, float]], SearchMetrics]:
        """
        Sparse vector search.

        Args:
            sparse_vector: {"indices": list[int], "values": list[float]}
        """
        if not self._sparse_name:
            raise RuntimeError("search_sparse requires a hybrid collection (sparse_vector_name)")

        qdrant_filter = _build_filter(filters) if filters else None

        t0 = time.monotonic()
        error_msg = None
        success = True
        results: list[tuple[Chunk, float]] = []

        try:
            query_vec = models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            )

            def _search():
                return self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    using=self._sparse_name,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )

            response = await _retry_async(
                _search,
                operation_name=f"search_sparse(top_k={top_k})",
            )

            for point in response.points:
                payload = point.payload or {}
                score = float(point.score) if point.score is not None else 0.0
                results.append((_payload_to_chunk(payload), score))

        except Exception as e:
            success = False
            error_msg = str(e)
            raise

        finally:
            duration = (time.monotonic() - t0) * 1000
            scores = [s for _, s in results]
            search_metrics = SearchMetrics(
                operation="search_sparse",
                collection=self.collection_name,
                duration_ms=duration,
                num_items=len(results),
                success=success,
                error=error_msg,
                top_k_requested=top_k,
                results_returned=len(results),
                min_score=min(scores) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                above_threshold_count=len(scores),
            )
            self._metrics_log.append(search_metrics)

        return results, search_metrics

    async def search_hybrid(
        self,
        query_embedding: list[float],
        sparse_vector: dict[str, Any],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        prefetch_limit: int = 50,
    ) -> tuple[list[tuple[Chunk, float]], SearchMetrics]:
        """
        Hybrid search via Qdrant prefetch + RRF fusion.
        """
        if not self._sparse_name:
            raise RuntimeError("search_hybrid requires a hybrid collection")

        qdrant_filter = _build_filter(filters) if filters else None
        dense_name = self._dense_name or "dense"

        t0 = time.monotonic()
        error_msg = None
        success = True
        results: list[tuple[Chunk, float]] = []

        try:
            sparse_qvec = models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            )

            def _search():
                return self._client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=query_embedding,
                            using=dense_name,
                            limit=prefetch_limit,
                            filter=qdrant_filter,
                        ),
                        models.Prefetch(
                            query=sparse_qvec,
                            using=self._sparse_name,
                            limit=prefetch_limit,
                            filter=qdrant_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )

            response = await _retry_async(
                _search,
                operation_name=f"search_hybrid(top_k={top_k})",
            )

            for point in response.points:
                payload = point.payload or {}
                score = float(point.score) if point.score is not None else 0.0
                results.append((_payload_to_chunk(payload), score))

        except Exception as e:
            success = False
            error_msg = str(e)
            raise

        finally:
            duration = (time.monotonic() - t0) * 1000
            scores = [s for _, s in results]
            search_metrics = SearchMetrics(
                operation="search_hybrid",
                collection=self.collection_name,
                duration_ms=duration,
                num_items=len(results),
                success=success,
                error=error_msg,
                top_k_requested=top_k,
                results_returned=len(results),
                min_score=min(scores) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                above_threshold_count=len(scores),
            )
            self._metrics_log.append(search_metrics)

        return results, search_metrics

    async def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by their IDs."""
        if not chunk_ids:
            return []

        point_ids = [_chunk_id_to_point_id(cid) for cid in chunk_ids]

        def _retrieve():
            return self._client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=True,
            )

        t0 = time.monotonic()
        try:
            points = await _retry_async(
                _retrieve,
                operation_name=f"get_by_ids({len(chunk_ids)})",
            )
            chunks = [_payload_to_chunk(p.payload or {}) for p in points]

            self._metrics_log.append(
                StoreOperationMetrics(
                    operation="get_by_ids",
                    collection=self.collection_name,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    num_items=len(chunks),
                    success=True,
                )
            )
            return chunks

        except Exception as e:
            self._metrics_log.append(
                StoreOperationMetrics(
                    operation="get_by_ids",
                    collection=self.collection_name,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    num_items=0,
                    success=False,
                    error=str(e),
                )
            )
            raise

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        doc_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

        def _delete():
            count_result = self._client.count(
                collection_name=self.collection_name,
                count_filter=doc_filter,
            )
            count_before = count_result.count

            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=doc_filter),
                wait=True,
            )
            return count_before

        t0 = time.monotonic()
        try:
            count = await _retry_async(
                _delete,
                operation_name=f"delete(doc={document_id})",
            )
            self._metrics_log.append(
                StoreOperationMetrics(
                    operation="delete",
                    collection=self.collection_name,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    num_items=count,
                    success=True,
                )
            )
            logger.info("Deleted %d chunks for document %s", count, document_id)
            return count

        except Exception as e:
            self._metrics_log.append(
                StoreOperationMetrics(
                    operation="delete",
                    collection=self.collection_name,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    num_items=0,
                    success=False,
                    error=str(e),
                )
            )
            raise

    async def get_stats(self) -> dict[str, Any]:
        """Collection statistics."""

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
            logger.info("Deleted collection: %s", self.collection_name)
        return result

    async def close(self) -> None:
        self._client.close()

    def get_metrics_log(self) -> list[StoreOperationMetrics]:
        """Return accumulated operation metrics."""
        return list(self._metrics_log)

    def clear_metrics_log(self) -> None:
        """Clear accumulated metrics (e.g. between experiment runs)."""
        self._metrics_log.clear()

    def get_metrics_summary(self) -> dict[str, Any]:
        """Aggregate summary of all operations since last clear."""
        if not self._metrics_log:
            return {"total_operations": 0}

        by_op: dict[str, list[StoreOperationMetrics]] = {}
        for m in self._metrics_log:
            by_op.setdefault(m.operation, []).append(m)

        summary: dict[str, Any] = {"total_operations": len(self._metrics_log)}
        for op, entries in by_op.items():
            durations = [e.duration_ms for e in entries]
            summary[op] = {
                "count": len(entries),
                "success_count": sum(1 for e in entries if e.success),
                "error_count": sum(1 for e in entries if not e.success),
                "mean_duration_ms": round(sum(durations) / len(durations), 2),
                "max_duration_ms": round(max(durations), 2),
                "total_items": sum(e.num_items for e in entries),
            }
        return summary


def _build_filter(filters: dict[str, Any]) -> models.Filter | None:
    """
    Build Qdrant filter from a flat dict.

    Supported value types:
      - scalar (str, int, bool) → MatchValue
      - list/tuple/set → MatchAny
      - dict with gte/lte/gt/lt keys → Range filter
    """
    conditions: list[models.Condition] = []

    for key, value in filters.items():
        if value is None:
            continue

        if isinstance(value, bool):
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
        elif isinstance(value, (list, tuple, set)):
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=list(value)),
                )
            )
        elif isinstance(value, dict):
            range_keys = {"gte", "lte", "gt", "lt"}
            if range_keys & value.keys():
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


async def create_qdrant_store(  # noqa: PLR0913
    collection_name: str = "documents",
    embedding_dim: int = 1024,
    distance: str = "cosine",
    on_disk: bool = False,
    sparse_vector_name: str | None = None,
    dense_vector_name: str | None = None,
) -> QdrantStore:
    return await QdrantStore.create(
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        distance=distance,
        on_disk=on_disk,
        sparse_vector_name=sparse_vector_name,
        dense_vector_name=dense_vector_name,
    )
