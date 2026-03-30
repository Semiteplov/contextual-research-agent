from __future__ import annotations

import asyncio
import time
from typing import Any

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.ingestion.embeddings.base import Embedder
from contextual_research_agent.ingestion.embeddings.sparse import SparseEncoder
from contextual_research_agent.retrieval.channels import RetrievalChannel
from contextual_research_agent.retrieval.channels.context import ContextAssembler
from contextual_research_agent.retrieval.channels.dense import DenseChannel
from contextual_research_agent.retrieval.channels.graph import (
    CitationGraphChannel,
    EntityGraphChannel,
)
from contextual_research_agent.retrieval.channels.paper_level import PaperLevelChannel
from contextual_research_agent.retrieval.channels.sparse import QdrantSparseBackend, SparseChannel
from contextual_research_agent.retrieval.config import FusionConfig, RetrievalConfig
from contextual_research_agent.retrieval.fusion import FusionStrategy, create_fusion_strategy
from contextual_research_agent.retrieval.metrics import (
    RetrievalOperationalMetrics,
    compute_operational_metrics,
)
from contextual_research_agent.retrieval.query import QueryAnalyzer, QueryIntent, QueryPlan
from contextual_research_agent.retrieval.reranking import Reranker, create_reranker
from contextual_research_agent.retrieval.types import (
    ChannelResult,
    RetrievalResult,
)

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    Multi-channel retrieval pipeline with fusion and reranking.

    Usage:
        pipeline = RetrievalPipeline(
            channels=[dense_channel, sparse_channel, graph_channel],
            embedder=embedder,
            config=config,
        )
        result = await pipeline.retrieve("How does LoRA work?")
    """

    def __init__(  # noqa: PLR0913
        self,
        channels: list[RetrievalChannel],
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        fusion: FusionStrategy | None = None,
        reranker: Reranker | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        context_assembler: ContextAssembler | None = None,
    ):
        self._channels = {ch.name: ch for ch in channels}
        self._embedder = embedder
        self._config = config or RetrievalConfig()

        self._fusion = fusion or create_fusion_strategy(self._config.fusion)
        self._reranker = reranker or create_reranker(self._config.rerank)
        self._query_analyzer = query_analyzer or QueryAnalyzer(
            method=self._config.query_analysis.method
        )
        self._context_assembler = context_assembler or ContextAssembler(self._config.context)

        logger.info(
            "RetrievalPipeline initialized: channels=%s, fusion=%s, reranker=%s",
            list(self._channels.keys()),
            self._config.fusion.strategy,
            self._config.rerank.model if self._config.rerank.enabled else "disabled",
        )

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        intent_override: str | None = None,
    ) -> RetrievalResult:
        """
        Run retrieval pipeline.

        Args:
            query: User query text.
            top_k: Override final number of results.
            document_ids: Restrict retrieval to specific documents.
            filters: Additional payload filters for all channels.
            intent_override: Force a specific intent (skip analysis).

        Returns:
            RetrievalResult with candidates, context, and full metrics.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        if intent_override:
            plan = QueryPlan(
                intent=QueryIntent(intent_override),
                original_query=query,
                document_ids=document_ids,
            )
        else:
            plan = self._query_analyzer.analyze(query, document_ids=document_ids)
        query_analysis_ms = (time.perf_counter() - t0) * 1000

        query_embedding = await self._embedder.embed_query(query)

        base_filters = dict(filters) if filters else {}
        if plan.document_ids:
            base_filters["document_id"] = plan.document_ids

        channel_tasks = []
        active_channel_names: list[str] = []

        for ch_name, channel in self._channels.items():
            active_channel_names.append(ch_name.value)
            channel_tasks.append(
                channel.retrieve(
                    query=query,
                    query_embedding=query_embedding,
                    filters=base_filters if base_filters else None,
                    section_types=plan.section_types,
                )
            )

        channel_results: list[ChannelResult] = await asyncio.gather(*channel_tasks)

        fusion_config = self._build_fusion_config(plan)
        fusion_strategy = create_fusion_strategy(fusion_config)

        fusion_result = fusion_strategy.fuse(
            channel_results=channel_results,
            top_n=self._config.fusion.top_n,
        )

        # rerank_top_k = top_k or self._config.rerank.top_k
        rerank_result = await self._reranker.rerank(
            query=query,
            candidates=fusion_result.candidates,
            top_k=self._config.fusion.top_n,
        )

        context, final_candidates = self._context_assembler.assemble(
            candidates=rerank_result.candidates,
        )

        total_latency_ms = (time.perf_counter() - t_total) * 1000

        result = RetrievalResult(
            query=query,
            candidates=final_candidates,
            context=context,
            channel_results=channel_results,
            fusion_result=fusion_result,
            rerank_result=rerank_result,
            total_latency_ms=total_latency_ms,
            query_analysis_ms=query_analysis_ms,
            intent=plan.intent.value,
            active_channels=active_channel_names,
            filters_applied=base_filters,
        )

        logger.info(
            "Retrieval complete: intent=%s, channels=%d, "
            "pre_fusion=%d, post_fusion=%d, final=%d (%.0fms)",
            plan.intent.value,
            len(channel_results),
            sum(cr.num_candidates for cr in channel_results),
            len(fusion_result.candidates),
            len(final_candidates),
            total_latency_ms,
        )

        return result

    async def retrieve_simple(
        self,
        query: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
    ) -> RetrievalResult:
        return await self.retrieve(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )

    def _build_fusion_config(self, plan: QueryPlan) -> FusionConfig:
        if not plan.channel_weight_overrides:
            return self._config.fusion

        merged_weights = dict(self._config.fusion.channel_weights)
        for ch, weight in plan.channel_weight_overrides.items():
            if ch in merged_weights:
                merged_weights[ch] = merged_weights[ch] * weight
            else:
                merged_weights[ch] = weight

        return FusionConfig(
            strategy=self._config.fusion.strategy,
            rrf_k=self._config.fusion.rrf_k,
            top_n=self._config.fusion.top_n,
            channel_weights=merged_weights,
        )

    def get_config(self) -> RetrievalConfig:
        return self._config


def create_retrieval_pipeline(
    embedder: Embedder,
    vector_store: Any,
    config: RetrievalConfig | None = None,
    paper_store: Any | None = None,
    graph_repo: Any | None = None,
    arxiv_to_doc_id: dict[str, str] | None = None,
) -> RetrievalPipeline:
    config = config or RetrievalConfig()
    channels: list[RetrievalChannel] = []

    if config.dense.enabled:
        channels.append(
            DenseChannel(
                vector_store=vector_store,
                embedder=embedder,
                config=config.dense,
            )
        )

    if config.sparse.enabled:
        sparse_encoder = SparseEncoder(model_name=config.sparse.sparse_model)

        sparse_backend = QdrantSparseBackend(
            client=vector_store._client,
            collection_name=vector_store.collection_name,
            sparse_vector_name="sparse",
            sparse_encoder=sparse_encoder,
        )

        channels.append(
            SparseChannel(
                backend=sparse_backend,
                config=config.sparse,
            )
        )

    # Paper-level channel
    if config.paper_level.enabled and paper_store is not None:
        channels.append(
            PaperLevelChannel(
                paper_store=paper_store,
                chunk_store=vector_store,
                embedder=embedder,
                config=config.paper_level,
            )
        )

    # Graph channels
    if graph_repo is not None:
        if config.graph.citation_enabled:
            channels.append(
                CitationGraphChannel(
                    graph_repo=graph_repo,
                    vector_store=vector_store,
                    embedder=embedder,
                    config=config.graph,
                    arxiv_to_doc_id=arxiv_to_doc_id,
                )
            )

        if config.graph.entity_enabled:
            channels.append(
                EntityGraphChannel(
                    graph_repo=graph_repo,
                    vector_store=vector_store,
                    embedder=embedder,
                    config=config.graph,
                    arxiv_to_doc_id=arxiv_to_doc_id,
                )
            )

    if not channels:
        raise ValueError("No retrieval channels enabled. At least one channel must be active.")

    return RetrievalPipeline(
        channels=channels,
        embedder=embedder,
        config=config,
    )
