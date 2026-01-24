from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from contextual_research_agent.common import logging
from contextual_research_agent.ingestion.domain.entities import Chunk, Document
from contextual_research_agent.ingestion.domain.types import DocumentStatus
from contextual_research_agent.ingestion.embeddings.base import Embedder
from contextual_research_agent.ingestion.parsers.base import DocumentParser
from contextual_research_agent.ingestion.result import IngestionResult
from contextual_research_agent.ingestion.state import IngestionPatch, IngestionState
from contextual_research_agent.ingestion.vectorstores.qdrant_store import QdrantStore

logger = logging.get_logger(__name__)


def create_initial_state(file_path: str) -> dict[str, Any]:
    return {
        "file_path": file_path,
        "run_id": f"ingest_{uuid4().hex[:12]}",
        "started_at": datetime.now(UTC).isoformat(),
        "document": None,
        "chunks": [],
        "embeddings": [],
        "latency": {},
        "status": "pending",
        "error": None,
    }


class IngestionPipeline:
    def __init__(
        self,
        parser: DocumentParser,
        embedder: Embedder,
        vector_store: QdrantStore,
    ):
        self._parser = parser
        self._embedder = embedder
        self._vector_store = vector_store

        self._graph = self._build_ingestion_graph()

        logger.info("IngestionPipeline initialized")

    async def parse_document(self, state: IngestionState) -> IngestionPatch:
        start = time.perf_counter()
        document = await self._parser.parse(state["file_path"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        latency = state.get("latency", {})
        latency["parse_ms"] = elapsed_ms

        if document.status == DocumentStatus.FAILED:
            return {
                "document": document,
                "latency": latency,
                "status": "failed",
                "error": document.metadata.get("error", "Parse failed"),
            }

        logger.info(f"Parsed document: {document.id} ({elapsed_ms:.0f}ms)")

        return {
            "document": document,
            "latency": latency,
            "status": "parsed",
        }

    async def extract_chunks(self, state: IngestionState) -> IngestionPatch:
        document: Document | None = state.get("document")
        if document is None:
            return {"status": "failed", "error": "No document to chunk"}

        start = time.perf_counter()
        chunks: list[Chunk] = await self._parser.extract_chunks(document)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latency = state.get("latency", {})
        latency["chunk_ms"] = elapsed_ms

        if not chunks:
            return {
                "chunks": [],
                "latency": latency,
                "status": "failed",
                "error": "No chunks extracted",
            }

        logger.info(f"Extracted {len(chunks)} chunks ({elapsed_ms:.0f}ms)")

        return {
            "chunks": chunks,
            "latency": latency,
            "status": "chunked",
        }

    async def generate_embeddings(self, state: IngestionState) -> IngestionPatch:
        chunks: list[Chunk] = state.get("chunks", [])
        if not chunks:
            return {"status": "failed", "error": "No chunks to embed"}

        texts = [chunk.text for chunk in chunks]

        start = time.perf_counter()
        embeddings = await self._embedder.embed_texts(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latency = state.get("latency", {})
        latency["embed_ms"] = elapsed_ms

        if len(embeddings) != len(chunks):
            return {
                "embeddings": [],
                "latency": latency,
                "status": "failed",
                "error": f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}",
            }

        logger.info(f"Generated {len(embeddings)} embeddings ({elapsed_ms:.0f}ms)")

        return {
            "embeddings": embeddings,
            "latency": latency,
            "status": "embedded",
        }

    async def index_chunks(self, state: IngestionState) -> IngestionPatch:
        chunks: list[Chunk] = state.get("chunks", [])
        embeddings: list[list[float]] = state.get("embeddings", [])
        document: Document | None = state.get("document")

        if not chunks or not embeddings:
            return {"status": "failed", "error": "No chunks or embeddings to index"}

        start = time.perf_counter()
        count = await self._vector_store.add_chunks(chunks, embeddings)
        elapsed_ms = (time.perf_counter() - start) * 1000

        latency = dict(state.get("latency", {}))
        latency["index_ms"] = elapsed_ms
        latency["total_ms"] = float(
            sum(latency.get(k, 0.0) for k in ("parse_ms", "chunk_ms", "embed_ms", "index_ms"))
        )

        if document is not None:
            document.status = DocumentStatus.INDEXED
            document.indexed_at = datetime.now(UTC)

        logger.info(
            "Indexed %d chunks (%.0fms, total=%.0fms)", count, elapsed_ms, latency["total_ms"]
        )

        return {
            "document": document,
            "latency": latency,
            "status": "indexed",
        }

    def _should_continue(self, state: IngestionState) -> str:
        if state.get("status") == "failed":
            return "end"
        return "continue"

    def _build_ingestion_graph(self) -> CompiledStateGraph:
        graph = StateGraph(IngestionState)

        graph.add_node("parse", self.parse_document)
        graph.add_node("chunk", self.extract_chunks)
        graph.add_node("embed", self.generate_embeddings)
        graph.add_node("index", self.index_chunks)

        graph.set_entry_point("parse")

        graph.add_conditional_edges(
            "parse",
            self._should_continue,
            {"continue": "chunk", "end": END},
        )
        graph.add_conditional_edges(
            "chunk",
            self._should_continue,
            {"continue": "embed", "end": END},
        )
        graph.add_conditional_edges(
            "embed",
            self._should_continue,
            {"continue": "index", "end": END},
        )

        graph.add_edge("index", END)

        return graph.compile()

    async def ingest(self, file_path: str) -> IngestionResult:
        initial_state = create_initial_state(file_path)
        logger.info(f"Starting ingestion: {file_path}")

        final_state = await self._graph.ainvoke(initial_state)

        return IngestionResult(
            run_id=final_state.get("run_id", ""),
            file_path=file_path,
            document=final_state.get("document"),
            chunk_count=len(final_state.get("chunks", [])),
            latency=final_state.get("latency", {}),
            status=final_state.get("status", "unknown"),
            error=final_state.get("error"),
        )

    async def ingest_batch(
        self,
        file_paths: list[str],
        continue_on_error: bool = True,
    ) -> list[IngestionResult]:
        results: list[IngestionResult] = []

        for path in file_paths:
            try:
                result = await self.ingest(path)
                results.append(result)

                if result.status == "failed" and not continue_on_error:
                    logger.error(f"Stopping batch due to failure: {result.error}")
                    break

            except Exception as e:
                logger.exception(f"Ingestion failed for {path}")

                if not continue_on_error:
                    raise

                results.append(
                    IngestionResult(
                        run_id="",
                        file_path=path,
                        document=None,
                        chunk_count=0,
                        latency={},
                        status="failed",
                        error=str(e),
                    )
                )

        success = sum(1 for r in results if r.status == "indexed")
        failed = len(results) - success
        logger.info(f"Batch complete: {success} succeeded, {failed} failed")

        return results

    async def reindex_document(self, document_id: str, file_path: str) -> IngestionResult:
        deleted = await self._vector_store.delete_by_document(document_id)
        logger.info(f"Deleted {deleted} existing chunks for {document_id}")

        return await self.ingest(file_path)


def create_ingestion_pipeline(
    parser: DocumentParser,
    embedder: Embedder,
    vector_store: QdrantStore,
) -> IngestionPipeline:
    return IngestionPipeline(
        parser=parser,
        embedder=embedder,
        vector_store=vector_store,
    )
