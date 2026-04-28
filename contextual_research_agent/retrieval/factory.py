from __future__ import annotations

from typing import Any

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.db.connection import get_connection
from contextual_research_agent.db.repositories.knowledge_graph import (
    KnowledgeGraphRepository,
)
from contextual_research_agent.ingestion.embeddings.hf_embedder import create_hf_embedder
from contextual_research_agent.ingestion.vectorstores.qdrant_store import (
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
from contextual_research_agent.retrieval.pipeline import create_retrieval_pipeline

logger = get_logger(__name__)


async def build_pipeline(
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
