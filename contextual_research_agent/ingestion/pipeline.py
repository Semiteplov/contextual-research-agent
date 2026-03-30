from __future__ import annotations

import asyncio
import contextlib
import re
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from contextual_research_agent.common import logging
from contextual_research_agent.ingestion.domain.entities import Chunk, Document
from contextual_research_agent.ingestion.domain.types import DocumentStatus
from contextual_research_agent.ingestion.embeddings.hf_embedder import HuggingFaceEmbedder
from contextual_research_agent.ingestion.embeddings.sparse import SparseEncoder
from contextual_research_agent.ingestion.extraction.citation_extractor import (
    CitationExtractionResult,
    CitationExtractor,
)
from contextual_research_agent.ingestion.extraction.entity_extractor import (
    EntityExtractionResult,
    EntityExtractor,
    store_entity_results,
)
from contextual_research_agent.ingestion.extraction.section_classifier import (
    SectionClassifier,
    SectionType,
)
from contextual_research_agent.ingestion.parsers.docling import DoclingParser, ParseResult
from contextual_research_agent.ingestion.result import (
    BatchResult,
    ExtractionMetrics,
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

    def __init__(  # noqa: PLR0913
        self,
        parser: DoclingParser,
        embedder: HuggingFaceEmbedder,
        vector_store: QdrantStore,
        paper_store: QdrantStore | None = None,
        graph_repo=None,
        section_classifier: SectionClassifier | None = None,
        citation_extractor: CitationExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
        sparse_encoder: SparseEncoder | None = None,
        print_summary: bool = True,
    ):
        self._parser = parser
        self._embedder = embedder
        self._store = vector_store
        self._paper_store = paper_store
        self._graph_repo = graph_repo
        self._section_classifier = section_classifier or SectionClassifier()
        self._citation_extractor = citation_extractor or CitationExtractor()
        self._entity_extractor = entity_extractor
        self._sparse_encoder = sparse_encoder
        self._print_summary = print_summary

        logger.info("IngestionPipeline initialized")

    async def ingest(self, file_path: str) -> IngestionResult:  # noqa: C901, PLR0911, PLR0912, PLR0915
        """
        Ingest a single document through the full pipeline.

        Stages:
          1. Parse (PDF → DoclingDocument + Document)
          2. Chunk (DoclingDocument → list[Chunk] + ChunkingMetrics)
          3. Enrich (section classification + citation extraction)
          4. Embed (chunks → embeddings)
          5. Index (embeddings → Qdrant)
          6. Store graph (citation edges → PostgreSQL)
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

        formula_not_decoded_count = sum(
            c.text.count("<!-- formula-not-decoded -->") for c in chunks
        )
        equation_chunks_count = sum(1 for c in chunks if c.metadata.get("chunk_type") == "equation")
        chunks_with_undecoded = sum(
            1
            for c in chunks
            if "<!-- formula-not-decoded -->" in c.text
            and c.metadata.get("chunk_type") == "equation"
        )

        for chunk in chunks:
            chunk.text = chunk.text.replace(
                "<!-- formula-not-decoded -->",
                "[mathematical formula — see original PDF]",
            )

        # --- Stage 3: Enrich ---
        t0 = time.perf_counter()
        extraction_metrics = ExtractionMetrics()
        citation_result: CitationExtractionResult | None = None

        try:
            t_classify = time.perf_counter()
            chunks = self._section_classifier.enrich_chunks(chunks)
            extraction_metrics.section_classify_ms = (time.perf_counter() - t_classify) * 1000

            section_dist = self._section_classifier.get_classification_stats(chunks)
            extraction_metrics.section_type_distribution = section_dist
            unknown_count = section_dist.get(SectionType.UNKNOWN.value, 0)
            extraction_metrics.unknown_section_rate = unknown_count / len(chunks) if chunks else 0.0

            extraction_metrics.formula_not_decoded_count = formula_not_decoded_count
            extraction_metrics.formula_total_count = equation_chunks_count
            extraction_metrics.formula_decode_rate = (
                1.0 - (chunks_with_undecoded / max(equation_chunks_count, 1))
                if equation_chunks_count > 0
                else 1.0
            )

            logger.info(
                "Formula quality: %d not decoded / %d equation chunks (%.1f%% decode rate)",
                formula_not_decoded_count,
                equation_chunks_count,
                extraction_metrics.formula_decode_rate * 100,
            )

            logger.info(
                "Section classification complete",
                extra={
                    "doc_id": document.id,
                    "distribution": section_dist,
                    "unknown_rate": f"{extraction_metrics.unknown_section_rate:.1%}",
                    "formula_decode_rate": f"{extraction_metrics.formula_decode_rate:.1%}",
                },
            )

            t_cite = time.perf_counter()
            arxiv_id = document.arxiv_id or self._guess_arxiv_id(file_path)

            if arxiv_id:
                citation_result = self._citation_extractor.extract(
                    docling_doc=parse_result.docling_doc,
                    document_id=document.id,
                    citing_arxiv_id=arxiv_id,
                    chunks=chunks,
                )
                extraction_metrics.citation_extract_ms = (time.perf_counter() - t_cite) * 1000
                extraction_metrics.total_references = citation_result.total_references
                extraction_metrics.resolved_references = citation_result.resolved_count
                extraction_metrics.resolution_rate = citation_result.resolution_rate
            else:
                logger.warning("No arxiv_id for %s, skipping citation extraction", document.id)

        except Exception as _:
            logger.exception("Enrichment stage failed (non-fatal)", extra={"doc_id": document.id})
            extraction_metrics = ExtractionMetrics()

        metrics.latency.enrich_ms = (time.perf_counter() - t0) * 1000
        metrics.extraction = extraction_metrics

        entity_result: EntityExtractionResult | None = None
        if self._entity_extractor:
            t0 = time.perf_counter()
            try:
                arxiv_id_for_entities = document.arxiv_id or self._guess_arxiv_id(file_path)
                if arxiv_id_for_entities:
                    entity_result = await self._entity_extractor.extract(
                        document_id=document.id,
                        arxiv_id=arxiv_id_for_entities,
                        chunks=chunks,
                    )
                    metrics.entity_extraction = entity_result

                    logger.info(
                        "Entity extraction complete",
                        extra={
                            "doc_id": document.id,
                            "entities": len(entity_result.entities),
                            "edges": len(entity_result.edges),
                            "llm_calls": entity_result.llm_calls,
                        },
                    )
            except Exception as e:
                logger.exception(
                    "Entity extraction failed (non-fatal)", extra={"doc_id": document.id}
                )

            metrics.latency.entity_extract_ms = (time.perf_counter() - t0) * 1000

        if self._print_summary:
            summary = format_document_summary(
                document,
                chunks,
                extraction_metrics,
                citation_result,
                entity_result,
            )
            print(summary)

        # --- Stage 4: Embed ---
        t0 = time.perf_counter()
        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = await self._embedder.embed_texts(texts)
        except Exception as e:
            logger.exception("Embed stage failed", extra={"doc_id": document.id})
            return self._failed_result(run_id, file_path, f"embed_error: {e}", metrics, document)

        metrics.latency.embed_ms = (time.perf_counter() - t0) * 1000

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

        sparse_vectors: list[dict[str, Any]] | None = None
        if self._sparse_encoder:
            t0 = time.perf_counter()
            try:
                sparse_vectors = await self._sparse_encoder.encode_texts_async(texts)
                metrics.latency.sparse_embed_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    "Sparse encoding complete",
                    extra={
                        "doc_id": document.id,
                        "num_vectors": len(sparse_vectors),
                        "ms": round(metrics.latency.sparse_embed_ms),
                    },
                )
            except Exception as e:
                metrics.latency.sparse_embed_ms = (time.perf_counter() - t0) * 1000
                logger.warning(
                    "Sparse encoding failed (non-fatal): %s",
                    e,
                    extra={"doc_id": document.id},
                )
                sparse_vectors = None

        if self._paper_store and document.abstract:
            t0 = time.perf_counter()
            try:
                paper_text = f"{document.title}\n\n{document.abstract}"
                paper_embedding = await self._embedder.embed_query(paper_text)

                paper_chunk = Chunk(
                    id=f"{document.id}_paper",
                    document_id=document.id,
                    text=paper_text,
                    token_count=0,
                    chunk_index=-1,
                    section="",
                    metadata={
                        "index_level": "paper",
                        "title": document.title,
                        "arxiv_id": document.arxiv_id or "",
                        "chunk_type": "text",
                        "section_type": "abstract",
                    },
                )

                await self._paper_store.add_chunks([paper_chunk], [paper_embedding])

                logger.info("Paper-level embedding indexed", extra={"doc_id": document.id})

            except Exception as e:
                logger.exception(
                    "Paper-level embedding failed (non-fatal)", extra={"doc_id": document.id}
                )

            metrics.latency.paper_embed_ms = (time.perf_counter() - t0) * 1000

        # --- Stage 5: Index ---
        t0 = time.perf_counter()
        try:
            count, store_metrics = await self._store.add_chunks(
                chunks, embeddings, sparse_vectors=sparse_vectors
            )
        except Exception as e:
            logger.exception("Index stage failed", extra={"doc_id": document.id})
            return self._failed_result(run_id, file_path, f"index_error: {e}", metrics, document)

        metrics.latency.index_ms = (time.perf_counter() - t0) * 1000
        metrics.indexing = store_metrics

        # --- Stage 6: Store graph edges ---
        t0 = time.perf_counter()
        if self._graph_repo:
            if citation_result and citation_result.edges:
                try:
                    self._graph_repo.store_extraction_result(citation_result)
                    logger.info(
                        "Citation edges stored",
                        extra={"doc_id": document.id, "citation_edges": len(citation_result.edges)},
                    )
                except Exception as e:
                    logger.exception("Citation storage failed (non-fatal)")
                    if self._graph_repo and hasattr(self._graph_repo, "_conn"):
                        with contextlib.suppress(Exception):
                            self._graph_repo._conn.rollback()

            if entity_result and entity_result.edges:
                try:
                    stored = store_entity_results(self._graph_repo, entity_result)
                    logger.info(
                        "Entity edges stored",
                        extra={"doc_id": document.id, "entity_edges": stored},
                    )
                except Exception as e:
                    logger.exception(
                        "Entity storage failed (non-fatal)", extra={"doc_id": document.id}
                    )

            with contextlib.suppress(Exception):
                self._graph_repo._conn.commit()

        metrics.latency.graph_store_ms = (time.perf_counter() - t0) * 1000

        document.status = DocumentStatus.INDEXED
        document.indexed_at = datetime.now(UTC)

        logger.info(
            "Ingestion complete",
            extra={
                "doc_id": document.id,
                "chunks": count,
                "citations": citation_result.resolved_count if citation_result else 0,
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
            citation_result=citation_result,
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
        deleted = await self._store.delete_by_document(document_id)
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

    @staticmethod
    def _guess_arxiv_id(file_path: str) -> str | None:
        """
        Try to extract arxiv_id from file path.
        e.g., "arxiv/papers/pdf/2106.09685.pdf" → "2106.09685"
        """
        match = re.search(r"(\d{4}\.\d{4,5})", file_path)
        return match.group(1) if match else None


def format_document_summary(  # noqa: C901, PLR0912, PLR0915
    document: Document,
    chunks: list[Chunk],
    extraction_metrics: ExtractionMetrics | None,
    citation_result: CitationExtractionResult | None,
    entity_result: EntityExtractionResult | None = None,
) -> str:
    """
    Shows structure, section types, citation stats, chunk breakdown.
    """
    lines: list[str] = []
    sep = "-" * 60

    # Header
    lines.append(sep)
    lines.append(f"Document: {document.title or document.id}")
    lines.append(sep)

    # Metadata
    if document.abstract:
        abstract_preview = document.abstract[:200] + ("..." if len(document.abstract) > 200 else "")
        lines.append(f"  Abstract:  {abstract_preview}")
    lines.append(f"  Sections:  {len(document.sections)}")
    lines.append(f"  Pages:     {document.num_pages}")
    lines.append(f"  Chunks:    {len(chunks)}")

    # Section type breakdown
    if extraction_metrics and extraction_metrics.section_type_distribution:
        lines.append("")
        lines.append("  Section type distribution:")
        dist = extraction_metrics.section_type_distribution
        for stype, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = count / len(chunks) * 100 if chunks else 0
            lines.append(f"    {stype:<20s} {count:>4d} chunks ({pct:.1f}%)")

    # Chunk type breakdown
    type_counts: dict[str, int] = {}
    for c in chunks:
        ct = c.metadata.get(
            "chunk_type", c.chunk_type.value if hasattr(c, "chunk_type") else "text"
        )
        type_counts[ct] = type_counts.get(ct, 0) + 1
    if type_counts:
        lines.append("")
        lines.append("  Chunk type distribution:")
        for ctype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {ctype:<20s} {count:>4d}")

    # Citations
    if citation_result:
        lines.append("")
        lines.append("  Citations:")
        lines.append(f"    Total references:  {citation_result.total_references}")
        lines.append(
            f"    Resolved:          {citation_result.resolved_count}"
            f" ({citation_result.resolution_rate:.0%})"
        )
        lines.append(f"    Unresolved:        {citation_result.unresolved_count}")

        # Show a few resolved citations
        if citation_result.edges:
            lines.append("")
            lines.append("  Resolved citations (first 5):")
            for edge in citation_result.edges[:5]:
                title_preview = edge.cited_title[:60] if edge.cited_title else "?"
                lines.append(f"    -> {edge.cited_paper_id:<20s} {title_preview}")

    if entity_result:
        lines.append("")
        lines.append("  Extracted entities:")
        # Group by type
        by_type: dict[str, list[str]] = {}
        for e in entity_result.entities:
            if e.entity_type not in by_type:
                by_type[e.entity_type] = []
            by_type[e.entity_type].append(e.name)

        for etype in ["method", "dataset", "task", "metric", "model"]:
            names = by_type.get(etype, [])
            if names:
                names_str = ", ".join(names[:8])
                if len(names) > 8:
                    names_str += f", ... (+{len(names) - 8})"
                lines.append(f"    {etype:<10s} {names_str}")

    # Sample chunks
    lines.append("")
    lines.append("  Sample chunks:")
    for chunk in chunks[:3]:
        st = chunk.metadata.get("section_type", "?")
        text_preview = chunk.text[:100].replace("\n", " ") + "..."
        lines.append(f"    [{chunk.chunk_index:>3d}] [{st:<15s}] {chunk.section}")
        lines.append(f"          {text_preview}")

    lines.append(sep)
    return "\n".join(lines)


def create_ingestion_pipeline(  # noqa: PLR0913
    parser: DoclingParser,
    embedder: HuggingFaceEmbedder,
    vector_store: QdrantStore,
    paper_store: QdrantStore | None = None,
    graph_repo=None,
    entity_extractor=None,
    print_summary: bool = True,
) -> IngestionPipeline:
    return IngestionPipeline(
        parser=parser,
        embedder=embedder,
        vector_store=vector_store,
        paper_store=paper_store,
        graph_repo=graph_repo,
        entity_extractor=entity_extractor,
        print_summary=print_summary,
    )
