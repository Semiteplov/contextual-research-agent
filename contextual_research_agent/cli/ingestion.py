from __future__ import annotations

import asyncio
import json
from pathlib import Path

from contextual_research_agent.agent.llm import LlamaCppProvider, OllamaProvider
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.common.settings import get_settings
from contextual_research_agent.db.connection import get_connection
from contextual_research_agent.db.repositories.datasets import DatasetsRepository
from contextual_research_agent.db.repositories.knowledge_graph import KnowledgeGraphRepository
from contextual_research_agent.ingestion.analytics import IngestionAnalytics
from contextual_research_agent.ingestion.domain.entities import DoclingParserConfig
from contextual_research_agent.ingestion.embeddings.hf_embedder import create_hf_embedder
from contextual_research_agent.ingestion.extraction.entity_extractor import (
    EntityExtractor,
    LlamaCppProviderAdapter,
    OllamaProviderAdapter,
)
from contextual_research_agent.ingestion.parsers.docling import create_docling_parser
from contextual_research_agent.ingestion.pipeline import IngestionPipeline
from contextual_research_agent.ingestion.result import BatchResult, IngestionResult
from contextual_research_agent.ingestion.tracking import IngestionTracker
from contextual_research_agent.ingestion.vectorstores.qdrant_store import create_qdrant_store

logger = get_logger(__name__)


async def _create_pipeline(  # noqa: PLR0913
    embedding_model: str = "BAAI/bge-m3",
    collection_name: str = "documents",
    max_tokens: int = 512,
    merge_peers: bool = True,
    include_context: bool = True,
    filter_empty_chunks: bool = False,
    min_chunk_tokens: int = 20,
    distance: str = "cosine",
    on_disk: bool = False,
    device: str | None = None,
    batch_size: int = 32,
    enable_graph: bool = True,
    print_summary: bool = True,
    enable_entities: bool = True,
    enable_paper_index: bool = True,
):
    """
    Build ingestion pipeline with all components.

    Returns (pipeline, config) — config is passed to MLflow.
    """
    config = DoclingParserConfig(
        max_tokens=max_tokens,
        embedding_model=embedding_model,
        merge_peers=merge_peers,
        include_context=include_context,
        filter_empty_chunks=filter_empty_chunks,
        min_chunk_tokens=min_chunk_tokens,
    )

    parser = create_docling_parser(
        embedding_model=embedding_model,
        max_tokens=max_tokens,
        include_context=include_context,
        merge_peers=merge_peers,
        filter_empty_chunks=filter_empty_chunks,
    )

    embedder = create_hf_embedder(
        model=embedding_model,
        device=device,
        batch_size=batch_size,
    )

    store = await create_qdrant_store(
        collection_name=collection_name,
        embedding_dim=embedder.dimension,
        distance=distance,
        on_disk=on_disk,
    )

    entity_extractor = None
    if enable_entities:
        # provider = OllamaProvider(model="qwen3:4b")
        # llm_client = OllamaProviderAdapter(provider)
        # entity_extractor = EntityExtractor(llm_client=llm_client)
        provider = LlamaCppProvider(model="qwen3:4b")
        llm_client = LlamaCppProviderAdapter(provider)
        entity_extractor = EntityExtractor(llm_client=llm_client)

    paper_store = None
    if enable_paper_index:
        paper_store = await create_qdrant_store(
            collection_name=f"{collection_name}_papers",
            embedding_dim=embedder.dimension,
        )

    graph_repo = None
    conn = None
    if enable_graph:
        try:
            conn = get_connection("arxiv")
            graph_repo = KnowledgeGraphRepository(conn)
        except Exception as e:
            logger.warning("Could not connect to PostgreSQL for graph storage: %s", e)
            conn = None
            graph_repo = None

    pipeline = IngestionPipeline(
        parser=parser,
        embedder=embedder,
        vector_store=store,
        paper_store=paper_store,
        graph_repo=graph_repo,
        entity_extractor=entity_extractor,
    )

    return pipeline, config, conn


def ingest_file(  # noqa: PLR0913
    file_path: str,
    embedding_model: str = "BAAI/bge-m3",
    collection: str = "documents",
    max_tokens: int = 512,
    no_merge_peers: bool = False,
    no_context: bool = False,
    filter_empty: bool = False,
    min_chunk_tokens: int = 20,
    distance: str = "cosine",
    device: str | None = None,
    batch_size: int = 32,
    save_report: bool = True,
    no_tracking: bool = False,
    no_graph: bool = False,
    enable_entities: bool = True,
    enable_paper_index: bool = True,
) -> None:
    """
    Ingest a single PDF file into the vector store.

    Args:
        file_path: S3 path (s3://bucket/key) or key within default bucket.

    Embedding params:
        embedding_model: HuggingFace model ID (must match chunking tokenizer).
        device: Torch device (None=auto, 'cpu', 'cuda', 'mps').
        batch_size: Embedding batch size.

    Chunking params:
        max_tokens: Target chunk size in tokens (256/512/1024).
        no_merge_peers: Disable merging adjacent small chunks.
        no_context: Disable prepending section headings to chunks.
        filter_empty: Drop chunks below min_chunk_tokens.
        min_chunk_tokens: Degenerate chunk threshold.

    Index params:
        collection: Qdrant collection name.
        distance: Distance metric (cosine/euclid/dot).

    Pipeline params:
        save_report: Save JSON report to reports/.
        no_tracking: Disable MLflow tracking.

    Example:
        python main.py ingest-file s3://rag-storage/arxiv/papers/2401.12345.pdf
        python main.py ingest-file s3://... --max-tokens=256 --no-context
    """

    async def _run() -> None:
        pipeline, config, conn = await _create_pipeline(
            embedding_model=embedding_model,
            collection_name=collection,
            max_tokens=max_tokens,
            merge_peers=not no_merge_peers,
            include_context=not no_context,
            filter_empty_chunks=filter_empty,
            min_chunk_tokens=min_chunk_tokens,
            distance=distance,
            device=device,
            batch_size=batch_size,
            enable_graph=not no_graph,
            enable_entities=enable_entities,
            enable_paper_index=enable_paper_index,
        )

        _print_config(config, collection, distance)

        try:
            result = await pipeline.ingest(file_path)
        finally:
            if conn:
                conn.close()

        _print_result(result)

        if save_report:
            _save_single_report(result)

        if not no_tracking:
            tracker = IngestionTracker(
                tracking_uri=get_settings().mlflow.tracking_uri,
            )
            tracker.log_single(
                result=result,
                config=config,
                embedding_model=embedding_model,
            )

    asyncio.run(_run())


def ingest_dataset(  # noqa: PLR0913
    name: str,
    split: str | None = None,
    embedding_model: str = "BAAI/bge-m3",
    collection: str = "documents",
    max_tokens: int = 512,
    no_merge_peers: bool = False,
    no_context: bool = False,
    filter_empty: bool = False,
    min_chunk_tokens: int = 20,
    distance: str = "cosine",
    on_disk: bool = False,
    device: str | None = None,
    batch_size: int = 32,
    continue_on_error: bool = True,
    max_concurrent: int = 1,
    limit: int | None = None,
    no_tracking: bool = False,
    no_graph: bool = False,
) -> None:
    """
    Ingest all papers from a dataset into the vector store.

    Reads paper paths from PostgreSQL (dataset_papers + arxiv_papers),
    filters by split if specified, and runs the ingestion pipeline.

    Args:
        name: Dataset name (e.g., 'baseline-v1').
        split: Split filter ('train', 'val', 'test', or None for all).

    Embedding params:
        embedding_model: HuggingFace model ID.
        device: Torch device.
        batch_size: Embedding batch size.

    Chunking params:
        max_tokens: Target chunk size in tokens.
        no_merge_peers: Disable merging adjacent small chunks.
        no_context: Disable section heading context in chunks.
        filter_empty: Drop degenerate chunks.
        min_chunk_tokens: Degenerate chunk threshold.

    Index params:
        collection: Qdrant collection name (use unique names for ablation).
        distance: Distance metric (cosine/euclid/dot).
        on_disk: Store vectors on disk (saves RAM, slower search).

    Pipeline params:
        continue_on_error: Continue on individual file failures.
        max_concurrent: Max parallel ingestions (1=sequential, safe for GPU).
        limit: Max papers to ingest (None=all).
        no_tracking: Disable MLflow tracking.

    Examples:
        python main.py ingest-dataset baseline-v1
        python main.py ingest-dataset baseline-v1 --max-tokens=1024 --collection=exp_1024
        python main.py ingest-dataset baseline-v1 --no-context --collection=exp_no_ctx
        python main.py ingest-dataset baseline-v1 --split=test --limit=5
    """

    async def _run() -> None:
        conn = get_connection("arxiv")
        try:
            repo = DatasetsRepository(conn)
            papers = repo.get_papers_with_paths(
                dataset_name=name,
                split=split,
                only_downloaded=True,
            )
        finally:
            conn.close()

        if not papers:
            print(
                f"No downloaded papers found for dataset '{name}'"
                f"{f' (split={split})' if split else ''}"
            )
            return

        paths = [p.storage_path for p in papers if p.storage_path]

        if limit:
            paths = paths[:limit]

        merge_peers = not no_merge_peers
        include_context = not no_context

        print(
            f"Ingesting {len(paths)} papers from dataset '{name}'"
            f"{f' (split={split})' if split else ''}"
        )
        print(f"  embedding_model: {embedding_model}")
        print(f"  collection:      {collection}")
        print(f"  max_tokens:      {max_tokens}")
        print(f"  max_concurrent:  {max_concurrent}")
        print()

        pipeline, config, conn = await _create_pipeline(
            embedding_model=embedding_model,
            collection_name=collection,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            include_context=include_context,
            filter_empty_chunks=filter_empty,
            min_chunk_tokens=min_chunk_tokens,
            distance=distance,
            on_disk=on_disk,
            device=device,
            batch_size=batch_size,
            enable_graph=not no_graph,
        )

        try:
            batch_result = await pipeline.ingest_batch(
                file_paths=paths,
                continue_on_error=continue_on_error,
                max_concurrent=max_concurrent,
            )
        finally:
            if conn:
                conn.close()

        _print_batch_result(batch_result, name)
        _save_batch_report(batch_result, name, split)

        if not no_tracking:
            tracker = IngestionTracker(
                tracking_uri=get_settings().mlflow.tracking_uri,
            )
            tracker.log_batch(
                batch=batch_result,
                config=config,
                dataset_name=name,
                split=split,
                embedding_model=embedding_model,
                tags={
                    "dataset": name,
                    "collection": collection,
                    "distance": distance,
                },
            )

    asyncio.run(_run())


def ingest_status(
    name: str,
    collection: str = "documents",
) -> None:
    """
    Show ingestion status for a dataset: how many papers are indexed.

    Args:
        name: Dataset name.
        collection: Qdrant collection name.

    Example:
        python main.py ingest-status baseline-v1
    """

    async def _run() -> None:
        conn = get_connection("arxiv")
        try:
            repo = DatasetsRepository(conn)
            stats = repo.get_stats(name)
        finally:
            conn.close()

        if stats is None:
            print(f"Dataset '{name}' not found")
            return

        store = await create_qdrant_store(collection_name=collection)
        store_stats = await store.get_stats()
        await store.close()

        print(f"\nDataset: {name}")
        print("=" * 50)
        print(f"  Total papers:     {stats.total}")
        print(f"  Downloaded:       {stats.downloaded}")
        print(f"  Train/Val/Test:   {stats.train}/{stats.val}/{stats.test}")
        print()
        print(f"Vector store: {collection}")
        print(f"  Total points:     {store_stats.get('points_count', '?')}")
        print(f"  Indexed vectors:  {store_stats.get('indexed_vectors_count', '?')}")
        print(f"  Status:           {store_stats.get('status', '?')}")

    asyncio.run(_run())


def reingest_failed(  # noqa: PLR0913
    name: str,
    report_path: str | None = None,
    embedding_model: str = "BAAI/bge-m3",
    collection: str = "documents",
    max_tokens: int = 512,
    no_merge_peers: bool = False,
    no_context: bool = False,
    filter_empty: bool = False,
    distance: str = "cosine",
    device: str | None = None,
    no_tracking: bool = False,
    no_graph: bool = False,
) -> None:
    """
    Re-ingest papers that failed in a previous batch run.

    Reads the batch report JSON to find failed file_paths, then re-runs them.

    Args:
        name: Dataset name.
        report_path: Path to batch report JSON. Default: reports/ingestion_{name}.json.

    All chunking/embedding/index params same as ingest-dataset.

    Example:
        python main.py reingest-failed baseline-v1
        python main.py reingest-failed baseline-v1 --max-tokens=256 --collection=exp_256
    """

    async def _run() -> None:
        path = report_path or f"reports/ingestion_{name}.json"
        report_file = Path(path)

        if not report_file.exists():
            print(f"Report not found: {path}")
            print("Run ingest-dataset first to generate a report.")
            return

        with Path.open(report_file) as f:
            report = json.load(f)

        failed_paths = [
            r["file_path"]
            for r in report.get("results", [])
            if r.get("status") == "failed" and r.get("file_path")
        ]

        if not failed_paths:
            print("No failed ingestions found in report.")
            return

        print(f"Re-ingesting {len(failed_paths)} failed papers...")

        pipeline, config, conn = await _create_pipeline(
            embedding_model=embedding_model,
            collection_name=collection,
            max_tokens=max_tokens,
            merge_peers=not no_merge_peers,
            include_context=not no_context,
            filter_empty_chunks=filter_empty,
            distance=distance,
            device=device,
        )

        _print_config(config, collection, distance)

        try:
            batch_result = await pipeline.ingest_batch(
                file_paths=failed_paths,
                continue_on_error=True,
            )
        finally:
            if conn:
                conn.close()

        _print_batch_result(batch_result, f"{name} (retry)")
        _save_batch_report(batch_result, name, split_label="retry")

        if not no_tracking:
            tracker = IngestionTracker(
                tracking_uri=get_settings().mlflow.tracking_uri,
            )
            tracker.log_batch(
                batch=batch_result,
                config=config,
                dataset_name=name,
                split="retry",
                embedding_model=embedding_model,
                tags={"dataset": name, "retry": "true"},
            )

    asyncio.run(_run())


def _print_config(
    config: DoclingParserConfig,
    collection: str,
    distance: str,
) -> None:
    """Print pipeline configuration before run."""
    print("\nConfiguration:")
    print(f"  embedding_model:   {config.embedding_model}")
    print(f"  max_tokens:        {config.max_tokens}")
    print(f"  merge_peers:       {config.merge_peers}")
    print(f"  include_context:   {config.include_context}")
    print(f"  filter_empty:      {config.filter_empty_chunks}")
    print(f"  min_chunk_tokens:  {config.min_chunk_tokens}")
    print(f"  collection:        {collection}")
    print(f"  distance:          {distance}")


def _print_result(result: IngestionResult) -> None:
    """Print single ingestion result."""
    status_icon = "✓" if result.success else "✗"
    print(f"\n{status_icon} Ingestion {'complete' if result.success else 'FAILED'}")
    print(f"  Run ID:      {result.run_id}")
    print(f"  File:        {result.file_path}")
    print(f"  Document ID: {result.document_id or 'N/A'}")
    print(f"  Chunks:      {result.chunk_count}")
    print(f"  Status:      {result.status}")

    latency = result.metrics.latency
    print(
        f"  Timing:      parse={latency.parse_ms:.0f}ms "
        f"chunk={latency.chunk_ms:.0f}ms "
        f"embed={latency.embed_ms:.0f}ms "
        f"index={latency.index_ms:.0f}ms "
        f"total={latency.total_ms:.0f}ms"
    )

    if result.error:
        print(f"  Error:       {result.error}")


def _print_batch_result(batch: BatchResult, dataset_name: str) -> None:
    """Print batch ingestion summary."""
    print()
    print("=" * 60)
    print(f"Batch ingestion: {dataset_name}")
    print("=" * 60)
    print(f"  Total:        {batch.total}")
    print(f"  Succeeded:    {batch.succeeded}")
    print(f"  Failed:       {batch.failed}")
    print(f"  Total chunks: {batch.total_chunks}")
    print(f"  Duration:     {batch.total_duration_ms / 1000:.1f}s")

    if batch.total > 0:
        avg = batch.total_duration_ms / batch.total
        print(f"  Avg per doc:  {avg:.0f}ms")

    if batch.failed > 0:
        print("\nFailed files:")
        for r in batch.results:
            if not r.success:
                print(f"  ✗ {r.file_path}: {r.error}")


def _save_batch_report(
    batch: BatchResult,
    dataset_name: str,
    split_label: str | None = None,
) -> None:
    """Save batch report as JSON for later analysis / re-ingestion."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    suffix = f"_{split_label}" if split_label else ""
    report_path = reports_dir / f"ingestion_{dataset_name}{suffix}.json"

    report = batch.to_dict()
    report["dataset_name"] = dataset_name
    report["split"] = split_label

    with Path.open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved: {report_path}")


def _save_single_report(result: IngestionResult) -> None:
    """Save single file ingestion report."""
    reports_dir = Path("reports/ingestion")
    reports_dir.mkdir(parents=True, exist_ok=True)

    doc_id = result.document_id or result.run_id
    report_path = reports_dir / f"{doc_id}.json"

    with Path.open(report_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"Report saved: {report_path}")


def print_ingestion_analytics(
    dataset_name: str,
    collection: str = "documents",
    paper_collection: str | None = None,
    log_mlflow: bool = False,
) -> None:
    """
    CLI command: compute and print corpus analytics.

    Usage:
        python main.py ingestion-analytics baseline-v1
        python main.py ingestion-analytics baseline-v1 --log-mlflow
    """

    async def _run():
        conn = get_connection()
        try:
            graph_repo = KnowledgeGraphRepository(conn)

            chunk_store = await create_qdrant_store(collection_name=collection)
            paper_store = None
            if paper_collection:
                paper_store = await create_qdrant_store(collection_name=paper_collection)

            analytics = IngestionAnalytics(
                graph_repo=graph_repo,
                chunk_store=chunk_store,
                paper_store=paper_store,
            )

            report = analytics.compute(dataset_name=dataset_name)
            print(report.format())

            if log_mlflow:
                report.log_to_mlflow()
                print("\nLogged to MLflow.")

            await chunk_store.close()
            if paper_store:
                await paper_store.close()

        finally:
            conn.close()

    asyncio.run(_run())
