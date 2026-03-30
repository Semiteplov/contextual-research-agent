from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import mlflow
import psycopg2

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.common.settings import get_settings
from contextual_research_agent.db.connection import get_connection
from contextual_research_agent.db.repositories.knowledge_graph import (
    KnowledgeGraphRepository,
)
from contextual_research_agent.ingestion.embeddings.hf_embedder import create_hf_embedder
from contextual_research_agent.ingestion.vectorstores.qdrant_store import (
    QdrantStore,
    _payload_to_chunk,
    create_qdrant_store,
)
from contextual_research_agent.retrieval.config import (
    DenseChannelConfig,
    FusionConfig,
    GraphChannelConfig,
    PaperLevelConfig,
    QueryAnalysisConfig,
    RerankConfig,
    RetrievalConfig,
    SparseChannelConfig,
)
from contextual_research_agent.retrieval.metrics import (
    IRQualityMetrics,
    aggregate_ir_metrics,
    compute_ir_metrics,
    compute_operational_metrics,
)
from contextual_research_agent.retrieval.pipeline import create_retrieval_pipeline
from contextual_research_agent.retrieval.tracking import RetrievalTracker
from contextual_research_agent.retrieval.types import RetrievalResult

logger = get_logger(__name__)


def retrieve(  # noqa: PLR0913
    question: str,
    collection: str = "documents",
    top_k: int = 10,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    rerank: bool = True,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    device: str | None = None,
    document: str | None = None,
    channels: str = "dense",
    verbose: bool = False,
    log_mlflow: bool = False,
    experiment_name: str = "retrieval",
) -> None:
    """
    Retrieve chunks for a query using the multi-channel pipeline.

    Args:
        question: Query text.
        collection: Qdrant collection name.
        top_k: Number of final results.
        embedding_model: Dense embedding model.
        rerank: Enable cross-encoder reranking.
        rerank_model: Reranker model name.
        device: Device for models (cuda/cpu/None=auto).
        document: Filter by document_id.
        channels: Comma-separated channel names (dense,sparse,graph_citation,graph_entity,paper_level).
        verbose: Show full chunk text.
    """

    async def _run() -> None:
        channel_list = (
            channels
            if isinstance(channels, list)
            else (list(channels) if isinstance(channels, tuple) else channels.split(","))
        )

        pipeline, config = await _build_pipeline(
            collection=collection,
            embedding_model=embedding_model,
            rerank_enabled=rerank,
            rerank_model=rerank_model,
            device=device,
            enabled_channels=channel_list,
        )

        document_ids = [document] if document else None

        result = await pipeline.retrieve(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
        )

        _print_retrieval_result(result, verbose=verbose)

        if log_mlflow:
            tracker = RetrievalTracker(
                experiment_name=experiment_name,
                tracking_uri=get_settings().mlflow.tracking_uri,
            )
            operational = compute_operational_metrics(result)

            tracker.log_single_query(
                config=config,
                ir_metrics=IRQualityMetrics(query=question),
                operational=operational,
                run_name=f"query_{question[:30]}",
            )

    asyncio.run(_run())


def evaluate(  # noqa: PLR0913
    eval_set: str,
    collection: str = "documents",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    rerank: bool = True,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    device: str | None = None,
    channels: str = "dense",
    output: str | None = None,
    experiment_name: str = "retrieval",
    run_name: str | None = None,
    k_values: str = "1,3,5,10",
) -> None:
    """
    Evaluate retrieval pipeline on a labeled query set.

    Args:
        eval_set: Path to evaluation JSON file.
            Format: [{"query": "...", "relevant_ids": ["chunk_id1", ...]}]
        collection: Qdrant collection name.
        embedding_model: Dense embedding model.
        rerank: Enable reranking.
        rerank_model: Reranker model name.
        device: Device for models.
        channels: Comma-separated channel names.
        output: Path to save evaluation results JSON.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name (auto-generated if None).
        k_values: Comma-separated K values for metrics.
    """

    async def _run() -> None:
        eval_data = _load_eval_set(eval_set)
        k_list = [
            int(k)
            for k in (k_values if isinstance(k_values, str) else ",".join(k_values)).split(",")
        ]

        channel_list = (
            channels
            if isinstance(channels, list)
            else (list(channels) if isinstance(channels, tuple) else channels.split(","))
        )

        pipeline, config = await _build_pipeline(
            collection=collection,
            embedding_model=embedding_model,
            rerank_enabled=rerank,
            rerank_model=rerank_model,
            device=device,
            enabled_channels=channel_list,
        )

        per_query_ir: list[IRQualityMetrics] = []
        latencies: list[float] = []

        all_results: list[RetrievalResult] = []
        for i, item in enumerate(eval_data):
            query_text = item["query"]
            relevant_ids = set(item["relevant_ids"])

            result = await pipeline.retrieve(query=query_text)
            all_results.append(result)

            retrieved_ids = [c.chunk_id for c in result.candidates]

            ir_metrics = compute_ir_metrics(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k_values=k_list,
                query=query_text,
            )
            per_query_ir.append(ir_metrics)
            latencies.append(result.total_latency_ms)

            print(
                f"  [{i + 1}/{len(eval_data)}] "
                f"MRR={ir_metrics.mrr:.3f} "
                f"R@5={ir_metrics.recall_at_k.get(5, 0):.3f} "
                f"NDCG@10={ir_metrics.ndcg_at_k.get(10, 0):.3f} "
                f"({result.total_latency_ms:.0f}ms)"
            )

        agg = aggregate_ir_metrics(per_query_ir, latencies)

        print("\n" + "=" * 60)
        print("Aggregated Results")
        print("=" * 60)
        for key, value in agg.to_dict().items():
            print(f"  {key}: {value}")

        tracker = RetrievalTracker(
            experiment_name=experiment_name,
            tracking_uri=get_settings().mlflow.tracking_uri,
        )

        operational_metrics = [compute_operational_metrics(r) for r in all_results]

        channels_str = ",".join(channel_list)
        tracker.log_evaluation(
            config=config,
            agg_metrics=agg,
            per_query=per_query_ir,
            operational=operational_metrics,
            run_name=run_name,
            tags={"channels": channels_str},
            eval_set_path=eval_set,
        )

        if output:
            report = {
                "config": config.to_mlflow_params(),
                "aggregated": agg.to_dict(),
                "per_query": [m.to_dict() for m in per_query_ir],
            }
            Path(output).write_text(json.dumps(report, indent=2))
            print(f"\nResults saved to: {output}")

    asyncio.run(_run())


def generate_eval_set(  # noqa: PLR0913
    collection: str = "documents",
    num_queries: int = 50,
    chunks_per_query: int = 1,
    output: str = "eval_set.json",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    llm_model: str = "Qwen/Qwen3-4B",
    seed: int = 42,
) -> None:
    async def _run() -> None:
        store = await QdrantStore.create(collection_name=collection)
        client = store._client

        random.seed(seed)
        info = client.get_collection(collection)
        total_points = info.points_count or 0

        if total_points == 0:
            print(f"Collection '{collection}' is empty.")
            return

        all_points, _ = client.scroll(
            collection_name=collection,
            limit=min(total_points, num_queries * 5),
            with_payload=True,
        )

        candidates = []
        for p in all_points:
            payload = p.payload or {}
            if payload.get("is_degenerate") or payload.get("is_fallback"):
                continue
            text = payload.get("text", "")
            if len(text) < 100:  # noqa: PLR2004
                continue
            section_type = (payload.get("metadata") or {}).get("section_type", "unknown")
            if section_type in ("references", "title", "unknown"):
                continue
            candidates.append(payload)

        if len(candidates) < num_queries:
            print(
                f"Only {len(candidates)} suitable chunks found "
                f"(requested {num_queries} queries). Using all."
            )

        sampled = random.sample(candidates, min(len(candidates), num_queries))

        eval_items: list[dict[str, Any]] = []
        for i, payload in enumerate(sampled):
            chunk_id = payload.get("chunk_id", "")
            text = payload.get("text", "")[:1500]
            section_type = (payload.get("metadata") or {}).get("section_type", "")

            question = _generate_question_template(text, section_type)

            eval_items.append(
                {
                    "query": question,
                    "relevant_ids": [chunk_id],
                    "metadata": {
                        "source_section_type": section_type,
                        "source_document_id": payload.get("document_id", ""),
                        "generation_method": "template",
                    },
                }
            )

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(sampled)} queries")

        Path(output).write_text(json.dumps(eval_items, indent=2))
        print(f"\nGenerated {len(eval_items)} evaluation queries → {output}")

    asyncio.run(_run())


def _generate_question_template(text: str, section_type: str) -> str:
    first_sentence = text.split(".")[0].strip() + "."

    templates = {
        "method": f"What method or approach is described: {first_sentence[:100]}",
        "experiments": f"What experimental setup or results are reported: {first_sentence[:100]}",
        "results": f"What are the main findings: {first_sentence[:100]}",
        "introduction": f"What problem or motivation is discussed: {first_sentence[:100]}",
        "related_work": f"What prior work is referenced: {first_sentence[:100]}",
        "abstract": f"What is the main contribution of this paper: {first_sentence[:100]}",
        "conclusion": f"What are the conclusions: {first_sentence[:100]}",
        "background": f"What background concepts are explained: {first_sentence[:100]}",
    }

    return templates.get(section_type, f"What does this passage describe: {first_sentence[:100]}")


async def _build_pipeline(  # noqa: PLR0913
    collection: str,
    embedding_model: str,
    rerank_enabled: bool,
    rerank_model: str,
    device: str | None,
    enabled_channels: list[str],
) -> tuple[Any, Any]:
    embedder = create_hf_embedder(model=embedding_model, device=device)
    is_hybrid = "sparse" in enabled_channels

    vector_store = await create_qdrant_store(
        collection_name=collection,
        embedding_dim=embedder.dimension,
        sparse_vector_name="sparse" if is_hybrid else None,
        dense_vector_name="dense" if is_hybrid else None,
    )

    arxiv_to_doc_id: dict[str, str] = {}
    try:
        client = vector_store._client
        points, _ = client.scroll(
            collection_name=collection,
            limit=10000,
            with_payload=["document_id"],
        )
        for p in points:
            doc_id = p.payload.get("document_id", "")
            arxiv_id = doc_id.split("_")[0] if "_" in doc_id else doc_id
            arxiv_to_doc_id[arxiv_id] = doc_id
        logger.info("Built arxiv→doc_id mapping: %d entries", len(arxiv_to_doc_id))
    except Exception as e:
        logger.warning("Failed to build arxiv→doc_id mapping: %s", e)

    paper_collection = f"{collection}_papers"
    paper_store = None
    try:
        paper_store = await create_qdrant_store(
            collection_name=paper_collection,
            embedding_dim=embedder.dimension,
        )
        if not await paper_store.collection_exists():
            paper_store = None
    except Exception:
        paper_store = None

    graph_repo = None
    try:
        conn = get_connection("arxiv")
        graph_repo = KnowledgeGraphRepository(conn)
    except Exception as e:
        logger.warning("Knowledge graph not available: %s", e)

    config = RetrievalConfig(
        embedding_model=embedding_model,
        dense=DenseChannelConfig(enabled="dense" in enabled_channels),
        sparse=SparseChannelConfig(enabled="sparse" in enabled_channels),
        graph=GraphChannelConfig(
            citation_enabled="graph_citation" in enabled_channels,
            entity_enabled="graph_entity" in enabled_channels,
            seed_top_k=10,
            citation_depth=2,
            max_papers=15,
            chunks_per_paper=3,
        ),
        paper_level=PaperLevelConfig(enabled="paper_level" in enabled_channels),
        fusion=FusionConfig(strategy="rrf"),
        rerank=RerankConfig(
            enabled=rerank_enabled,
            model=rerank_model,
            device=device,
        ),
        query_analysis=QueryAnalysisConfig(method="rule"),
    )

    pipeline = create_retrieval_pipeline(
        embedder=embedder,
        vector_store=vector_store,
        config=config,
        paper_store=paper_store,
        graph_repo=graph_repo,
        arxiv_to_doc_id=arxiv_to_doc_id,
    )

    return pipeline, config


def _load_eval_set(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    for i, item in enumerate(data):
        if "query" not in item or "relevant_ids" not in item:
            raise ValueError(f"Item {i} missing 'query' or 'relevant_ids' keys")
    return data


def _print_retrieval_result(result: Any, verbose: bool = False) -> None:
    print(f"\nQuery: {result.query}")
    print(f"Intent: {result.intent}")
    print(f"Channels: {', '.join(result.active_channels)}")
    print(f"Results: {result.num_results} ({result.total_latency_ms:.0f}ms)")
    print("=" * 60)

    for cr in result.channel_results:
        print(f"  {cr.channel.value}: {cr.num_candidates} candidates ({cr.latency_ms:.0f}ms)")

    if result.fusion_result:
        print(f"\nFusion: {len(result.fusion_result.candidates)} candidates")
        for ch, cnt in result.fusion_result.channel_contributions.items():
            print(f"  {ch}: {cnt} in final set")

    if result.rerank_result and result.rerank_result.model_name != "noop":
        print(
            f"\nRerank: {result.rerank_result.model_name} "
            f"(mean rank change: {result.rerank_result.rank_changes:.1f})"
        )

    print("\n" + "-" * 60)
    for i, c in enumerate(result.candidates, 1):
        section_type = c.chunk.metadata.get("section_type", "?")
        print(f"\n[{i}] {c.chunk_id} (score: {c.score:.4f}) [{section_type}]")
        print(f"    Document: {c.document_id}")
        if c.chunk.section:
            print(f"    Section: {c.chunk.section}")
        print(f"    Channel: {c.channel.value}")

        if verbose:
            print("-" * 40)
            print(c.chunk.text)
        else:
            text = c.chunk.text
            if len(text) > 200:  # noqa: PLR2004
                text = text[:200] + "..."
            print(f"    {text}")

    print("\n" + "-" * 60)
    print("Operational Metrics")
    print("=" * 60)
    print(f"  Total latency:     {result.total_latency_ms:.0f}ms")
    print(f"  Query analysis:    {result.query_analysis_ms:.1f}ms")
    for cr in result.channel_results:
        print(f"  {cr.channel.value}:  {cr.latency_ms:.0f}ms ({cr.num_candidates} candidates)")
    if result.fusion_result:
        print(f"  Fusion:            {result.fusion_result.latency_ms:.1f}ms")
    if result.rerank_result:
        print(f"  Rerank:            {result.rerank_result.latency_ms:.0f}ms")
    print(f"  Unique documents:  {len(result.document_ids)}")

    print()


def map_eval_queries_to_chunks(  # noqa: PLR0913
    eval_path: str = "eval/peft_gold_v1.json",
    collection: str = "peft_hybrid",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    top_k_per_paper: int = 10,
    score_threshold: float = 0.35,
    output: str = "eval/peft_gold_v1_mapped.json",
) -> None:
    """
    Map evaluation queries to relevant chunk_ids.

    For each query:
      1. Embed query
      2. Search within source_papers (document_id filter)
      3. Filter by relevant_sections (section_type filter)
      4. Take top-K chunks above threshold as relevant_ids
    """

    async def _run() -> None:
        eval_data = json.loads(Path(eval_path).read_text())
        embedder = create_hf_embedder(model=embedding_model)
        store = await create_qdrant_store(
            collection_name=collection,
            embedding_dim=embedder.dimension,
        )

        mapped = 0
        empty = 0

        for i, item in enumerate(eval_data):
            query = item["query"]

            query_embedding = await embedder.embed_query(query)

            results, _ = await store.search(
                query_embedding=query_embedding,
                top_k=top_k_per_paper,
                score_threshold=score_threshold,
            )

            all_chunks = []
            for chunk, score in results:
                all_chunks.append(
                    {
                        "chunk_id": chunk.id,
                        "score": score,
                        "section_type": chunk.metadata.get("section_type", ""),
                        "document_id": chunk.document_id,
                        "text_preview": chunk.text[:150],
                    }
                )

            all_chunks.sort(key=lambda x: x["score"], reverse=True)
            top_chunks = all_chunks[:top_k_per_paper]

            item["relevant_ids"] = [c["chunk_id"] for c in top_chunks]
            item["mapped_chunks"] = top_chunks

            if top_chunks:
                mapped += 1
            else:
                empty += 1

            if (i + 1) % 20 == 0:
                print(f"  Mapped {i + 1}/{len(eval_data)} ({mapped} with chunks, {empty} empty)")

        Path(output).write_text(json.dumps(eval_data, indent=2))
        print(f"\nDone: {mapped} mapped, {empty} empty → {output}")

    asyncio.run(_run())
