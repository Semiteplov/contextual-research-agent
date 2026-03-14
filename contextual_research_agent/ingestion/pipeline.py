from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from uuid import uuid4

from contextual_research_agent.common import logging
from contextual_research_agent.ingestion.domain.entities import Document
from contextual_research_agent.ingestion.domain.types import DocumentStatus
from contextual_research_agent.ingestion.embeddings.hf_embedder import HuggingFaceEmbedder
from contextual_research_agent.ingestion.parsers.docling import DoclingParser, ParseResult
from contextual_research_agent.ingestion.result import (
    BatchResult,
    IngestionMetrics,
    IngestionResult,
)
from contextual_research_agent.ingestion.vectorstores.qdrant_store import QdrantStore

logger = logging.get_logger(__name__)


class IngestionPipeline:
    """
    Async ingestion pipeline: PDF → parse → chunk → embed → index.

    Each stage returns typed results with metrics.
    Failures at any stage produce a failed IngestionResult
    without propagating exceptions (unless continue_on_error=False).
    """

    def __init__(
        self,
        parser: DoclingParser,
        embedder: HuggingFaceEmbedder,
        vector_store: QdrantStore,
    ):
        self._parser = parser
        self._embedder = embedder
        self._vector_store = vector_store

        logger.info("IngestionPipeline initialized")

    async def ingest(self, file_path: str) -> IngestionResult:  # noqa: PLR0911
        """
        Ingest a single document through the full pipeline.

        Returns IngestionResult with status "indexed" or "failed".
        Does not raise on pipeline errors — captures them in result.error.
        """
        run_id = f"ingest_{uuid4().hex[:12]}"
        metrics = IngestionMetrics()

        logger.info("Starting ingestion", extra={"run_id": run_id, "file_path": file_path})

        # --- Stage 1: Parse ---
        t0 = time.perf_counter()
        try:
            parse_result: ParseResult = await self._parser.parse(file_path)
        except Exception as e:
            logger.exception("Parse stage failed", extra={"file_path": file_path})
            return self._failed_result(run_id, file_path, f"parse_error: {e}", metrics)

        metrics.latency.parse_ms = (time.perf_counter() - t0) * 1000
        document = parse_result.document

        if document.is_failed:
            return self._failed_result(
                run_id,
                file_path,
                document.metadata.get("error", "Parse failed"),
                metrics,
                document,
            )

        logger.info(
            "Parse complete",
            extra={"doc_id": document.id, "ms": round(metrics.latency.parse_ms)},
        )

        # --- Stage 2: Chunk ---
        t0 = time.perf_counter()
        try:
            chunks, chunking_metrics = await self._parser.extract_chunks(
                document=document,
                docling_doc=parse_result.docling_doc,
            )
        except Exception as e:
            logger.exception("Chunk stage failed", extra={"doc_id": document.id})
            return self._failed_result(run_id, file_path, f"chunk_error: {e}", metrics, document)

        metrics.latency.chunk_ms = (time.perf_counter() - t0) * 1000
        metrics.chunking = chunking_metrics

        if not chunks:
            return self._failed_result(
                run_id,
                file_path,
                "No chunks extracted",
                metrics,
                document,
            )

        logger.info(
            "Chunking complete",
            extra={
                "doc_id": document.id,
                "num_chunks": len(chunks),
                "ms": round(metrics.latency.chunk_ms),
            },
        )

        # --- Stage 3: Embed ---
        t0 = time.perf_counter()
        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = await self._embedder.embed_texts(texts)
        except Exception as e:
            logger.exception("Embed stage failed", extra={"doc_id": document.id})
            return self._failed_result(run_id, file_path, f"embed_error: {e}", metrics, document)

        metrics.latency.embed_ms = (time.perf_counter() - t0) * 1000

        # Retrieve embedding metrics from embedder's log (last entry)
        emb_log = self._embedder.get_metrics_log()
        if emb_log:
            metrics.embedding = emb_log[-1]

        if len(embeddings) != len(chunks):
            return self._failed_result(
                run_id,
                file_path,
                f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}",
                metrics,
                document,
            )

        logger.info(
            "Embedding complete",
            extra={
                "doc_id": document.id,
                "num_embeddings": len(embeddings),
                "ms": round(metrics.latency.embed_ms),
            },
        )

        # --- Stage 4: Index ---
        t0 = time.perf_counter()
        try:
            count, store_metrics = await self._vector_store.add_chunks(chunks, embeddings)
        except Exception as e:
            logger.exception("Index stage failed", extra={"doc_id": document.id})
            return self._failed_result(run_id, file_path, f"index_error: {e}", metrics, document)

        metrics.latency.index_ms = (time.perf_counter() - t0) * 1000
        metrics.indexing = store_metrics

        # Mark document as indexed
        document.status = DocumentStatus.INDEXED
        document.indexed_at = datetime.now(UTC)

        logger.info(
            "Ingestion complete",
            extra={
                "doc_id": document.id,
                "chunks": count,
                "total_ms": round(metrics.latency.total_ms),
            },
        )

        return IngestionResult(
            run_id=run_id,
            file_path=file_path,
            status="indexed",
            document=document,
            chunk_count=count,
            metrics=metrics,
            completed_at=datetime.now(UTC).isoformat(),
        )

    async def ingest_batch(
        self,
        file_paths: list[str],
        continue_on_error: bool = True,
        max_concurrent: int = 1,
    ) -> BatchResult:
        """
        Ingest multiple documents.

        Args:
            file_paths: List of S3 paths to PDF files.
            continue_on_error: If False, stop on first failure.
            max_concurrent: Max parallel ingestions.
                            Default 1 (sequential) — safe for GPU embedding.
        """
        batch_start = time.perf_counter()
        results: list[IngestionResult] = []

        if max_concurrent <= 1:
            for i, path in enumerate(file_paths):
                logger.info("Batch progress: %d/%d", i + 1, len(file_paths))
                try:
                    result = await self.ingest(path)
                    results.append(result)

                    if not result.success and not continue_on_error:
                        logger.error("Stopping batch: %s", result.error)
                        break

                except Exception as e:
                    logger.exception("Batch item failed: %s", path)
                    if not continue_on_error:
                        raise
                    results.append(
                        self._failed_result(
                            f"ingest_{uuid4().hex[:12]}",
                            path,
                            str(e),
                            IngestionMetrics(),
                        )
                    )
        else:
            sem = asyncio.Semaphore(max_concurrent)

            async def _bounded_ingest(path: str) -> IngestionResult:
                async with sem:
                    try:
                        return await self.ingest(path)
                    except Exception as e:
                        logger.exception("Concurrent ingest failed: %s", path)
                        return self._failed_result(
                            f"ingest_{uuid4().hex[:12]}",
                            path,
                            str(e),
                            IngestionMetrics(),
                        )

            results = await asyncio.gather(
                *[_bounded_ingest(p) for p in file_paths],
            )
            results = list(results)

        batch_duration = (time.perf_counter() - batch_start) * 1000
        batch_result = BatchResult(results=results, total_duration_ms=batch_duration)

        logger.info(
            "Batch complete",
            extra={
                "total": batch_result.total,
                "succeeded": batch_result.succeeded,
                "failed": batch_result.failed,
                "total_chunks": batch_result.total_chunks,
                "duration_ms": round(batch_duration),
            },
        )

        return batch_result

    async def reindex_document(self, document_id: str, file_path: str) -> IngestionResult:
        """Delete existing chunks and re-ingest."""
        deleted = await self._vector_store.delete_by_document(document_id)
        logger.info("Deleted %d existing chunks for %s", deleted, document_id)
        return await self.ingest(file_path)

    @staticmethod
    def _failed_result(
        run_id: str,
        file_path: str,
        error: str,
        metrics: IngestionMetrics,
        document: Document | None = None,
    ) -> IngestionResult:
        return IngestionResult(
            run_id=run_id,
            file_path=file_path,
            status="failed",
            document=document,
            chunk_count=0,
            metrics=metrics,
            error=error,
            completed_at=datetime.now(UTC).isoformat(),
        )


def create_ingestion_pipeline(
    parser: DoclingParser,
    embedder: HuggingFaceEmbedder,
    vector_store: QdrantStore,
) -> IngestionPipeline:
    return IngestionPipeline(
        parser=parser,
        embedder=embedder,
        vector_store=vector_store,
    )
