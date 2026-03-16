from __future__ import annotations

import time
from typing import Any

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.ingestion.embeddings.base import Embedder
from contextual_research_agent.ingestion.vectorstores.qdrant_store import QdrantStore
from contextual_research_agent.retrieval.channels import RetrievalChannel
from contextual_research_agent.retrieval.config import GraphChannelConfig
from contextual_research_agent.retrieval.types import (
    ChannelName,
    ChannelResult,
    ScoredCandidate,
)

logger = get_logger(__name__)


class CitationGraphChannel(RetrievalChannel):
    """
    Citation graph traversal channel.

    Strategy:
      1. Dense search → top seed paper IDs.
      2. PostgreSQL citation_edges → papers that cite or are cited by seeds.
      3. Fetch top chunks from discovered papers via Qdrant filter.
      4. Score: seed_score x edge_confidence x depth_decay.

    This channel finds related work that may not match the query embedding
    but is structurally connected to highly relevant papers.
    """

    def __init__(
        self,
        graph_repo: Any,
        vector_store: QdrantStore,
        embedder: Embedder,
        config: GraphChannelConfig | None = None,
    ):
        self._graph = graph_repo
        self._store = vector_store
        self._embedder = embedder
        self._config = config or GraphChannelConfig()

    @property
    def name(self) -> ChannelName:
        return ChannelName.GRAPH_CITATION

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChannelResult:
        t0 = time.perf_counter()

        try:
            if query_embedding is None:
                query_embedding = await self._embedder.embed_query(query)

            seed_results, _ = await self._store.search(
                query_embedding=query_embedding,
                top_k=self._config.seed_top_k,
                filters=filters,
            )

            if not seed_results:
                return self._empty_result(time.perf_counter() - t0)

            seed_doc_ids: list[str] = []
            seed_scores: dict[str, float] = {}
            for chunk, score in seed_results:
                doc_id = chunk.document_id
                if doc_id not in seed_scores:
                    seed_doc_ids.append(doc_id)
                    seed_scores[doc_id] = score

            connected_paper_ids = self._walk_citations(
                seed_paper_ids=seed_doc_ids,
                depth=self._config.citation_depth,
                max_papers=self._config.max_papers,
            )

            if not connected_paper_ids:
                return self._empty_result(time.perf_counter() - t0)

            candidates = await self._fetch_chunks_for_papers(
                paper_ids=connected_paper_ids,
                query_embedding=query_embedding,
                seed_scores=seed_scores,
                chunks_per_paper=self._config.chunks_per_paper,
            )

            latency_ms = (time.perf_counter() - t0) * 1000

            logger.debug(
                "Citation graph: %d seed papers → %d connected → %d chunks (%.0fms)",
                len(seed_doc_ids),
                len(connected_paper_ids),
                len(candidates),
                latency_ms,
            )

            return ChannelResult(
                channel=ChannelName.GRAPH_CITATION,
                candidates=candidates,
                latency_ms=latency_ms,
                metadata={
                    "seed_papers": len(seed_doc_ids),
                    "connected_papers": len(connected_paper_ids),
                    "depth": self._config.citation_depth,
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Citation graph channel failed: %s", e)
            return ChannelResult(
                channel=ChannelName.GRAPH_CITATION,
                candidates=[],
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    def _walk_citations(
        self,
        seed_paper_ids: list[str],
        depth: int,
        max_papers: int,
    ) -> list[str]:
        if not seed_paper_ids:
            return []

        try:
            conn = self._graph._conn
            with conn.cursor() as cur:
                placeholders = ",".join(["%s"] * len(seed_paper_ids))
                cur.execute(
                    f"""
                    WITH RECURSIVE citation_walk AS (
                        -- Depth 0: direct connections
                        SELECT cited_paper_id AS paper_id, 1 AS depth
                        FROM citation_edges
                        WHERE citing_paper_id IN ({placeholders})

                        UNION

                        SELECT citing_paper_id AS paper_id, 1 AS depth
                        FROM citation_edges
                        WHERE cited_paper_id IN ({placeholders})

                        UNION ALL

                        -- Depth > 1: recursive step
                        SELECT ce.cited_paper_id, cw.depth + 1
                        FROM citation_walk cw
                        JOIN citation_edges ce ON ce.citing_paper_id = cw.paper_id
                        WHERE cw.depth < %s
                    )
                    SELECT DISTINCT paper_id, MIN(depth) as min_depth
                    FROM citation_walk
                    WHERE paper_id NOT IN ({placeholders})
                    GROUP BY paper_id
                    ORDER BY min_depth, paper_id
                    LIMIT %s
                    """,
                    seed_paper_ids + seed_paper_ids + [depth] + seed_paper_ids + [max_papers],
                )
                rows = cur.fetchall()
                return [row[0] for row in rows]

        except Exception as e:
            logger.warning("Citation walk failed: %s", e)
            return []

    async def _fetch_chunks_for_papers(
        self,
        paper_ids: list[str],
        query_embedding: list[float],
        seed_scores: dict[str, float],
        chunks_per_paper: int,
    ) -> list[ScoredCandidate]:
        """Fetch top chunks from discovered papers, scored by query relevance."""
        candidates: list[ScoredCandidate] = []

        for paper_id in paper_ids:
            results, _ = await self._store.search(
                query_embedding=query_embedding,
                top_k=chunks_per_paper,
                filters={"document_id": paper_id},
            )

            decay = 0.8

            for chunk, score in results:
                candidates.append(
                    ScoredCandidate(
                        chunk=chunk,
                        score=score * decay,
                        rank=len(candidates),
                        channel=ChannelName.GRAPH_CITATION,
                        metadata={"source_paper": paper_id, "discovery": "citation"},
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return [
            ScoredCandidate(
                chunk=c.chunk,
                score=c.score,
                rank=i,
                channel=c.channel,
                metadata=c.metadata,
            )
            for i, c in enumerate(candidates)
        ]

    def _empty_result(self, elapsed: float) -> ChannelResult:
        return ChannelResult(
            channel=ChannelName.GRAPH_CITATION,
            candidates=[],
            latency_ms=elapsed * 1000,
        )


class EntityGraphChannel(RetrievalChannel):
    """
    Entity-based graph retrieval channel.

    Strategy:
      1. Extract entity mentions from query (keyword match or LLM).
      2. Look up entities in PostgreSQL entities table.
      3. Find papers connected to those entities via paper_entity_edges.
      4. Fetch top chunks from those papers.
    """

    def __init__(
        self,
        graph_repo: Any,
        vector_store: QdrantStore,
        embedder: Embedder,
        config: GraphChannelConfig | None = None,
    ):
        self._graph = graph_repo
        self._store = vector_store
        self._embedder = embedder
        self._config = config or GraphChannelConfig()

    @property
    def name(self) -> ChannelName:
        return ChannelName.GRAPH_ENTITY

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChannelResult:
        t0 = time.perf_counter()

        try:
            entity_matches = self._match_entities(query)

            if not entity_matches:
                return ChannelResult(
                    channel=ChannelName.GRAPH_ENTITY,
                    candidates=[],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    metadata={"matched_entities": 0},
                )

            paper_ids = self._find_papers_by_entities(
                entity_ids=[e["id"] for e in entity_matches],
                max_papers=self._config.max_papers,
            )

            if not paper_ids:
                return ChannelResult(
                    channel=ChannelName.GRAPH_ENTITY,
                    candidates=[],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    metadata={
                        "matched_entities": len(entity_matches),
                        "discovered_papers": 0,
                    },
                )

            if query_embedding is None:
                query_embedding = await self._embedder.embed_query(query)

            candidates: list[ScoredCandidate] = []
            for paper_id in paper_ids:
                results, _ = await self._store.search(
                    query_embedding=query_embedding,
                    top_k=self._config.chunks_per_paper,
                    filters={"document_id": paper_id},
                )

                for chunk, score in results:
                    candidates.append(
                        ScoredCandidate(
                            chunk=chunk,
                            score=score * 0.85,
                            rank=len(candidates),
                            channel=ChannelName.GRAPH_ENTITY,
                            metadata={
                                "source_paper": paper_id,
                                "discovery": "entity",
                                "matched_entities": [e["name"] for e in entity_matches],
                            },
                        )
                    )

            candidates.sort(key=lambda c: c.score, reverse=True)
            candidates = [
                ScoredCandidate(
                    chunk=c.chunk,
                    score=c.score,
                    rank=i,
                    channel=c.channel,
                    metadata=c.metadata,
                )
                for i, c in enumerate(candidates)
            ]

            latency_ms = (time.perf_counter() - t0) * 1000

            logger.debug(
                "Entity graph: %d entities → %d papers → %d chunks (%.0fms)",
                len(entity_matches),
                len(paper_ids),
                len(candidates),
                latency_ms,
            )

            return ChannelResult(
                channel=ChannelName.GRAPH_ENTITY,
                candidates=candidates,
                latency_ms=latency_ms,
                metadata={
                    "matched_entities": [e["name"] for e in entity_matches],
                    "discovered_papers": len(paper_ids),
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Entity graph channel failed: %s", e)
            return ChannelResult(
                channel=ChannelName.GRAPH_ENTITY,
                candidates=[],
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    def _match_entities(self, query: str) -> list[dict[str, Any]]:
        try:
            conn = self._graph._conn
            query_lower = query.lower()

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, entity_type, normalized_name
                    FROM entities
                    WHERE LOWER(name) = ANY(
                        SELECT unnest(string_to_array(%s, ' '))
                    )
                    OR %s ILIKE '%%' || name || '%%'
                    ORDER BY LENGTH(name) DESC
                    LIMIT 10
                    """,
                    [query_lower, query_lower],
                )
                rows = cur.fetchall()
                return [{"id": r[0], "name": r[1], "type": r[2], "normalized": r[3]} for r in rows]

        except Exception as e:
            logger.warning("Entity matching failed: %s", e)
            return []

    def _find_papers_by_entities(
        self,
        entity_ids: list[int],
        max_papers: int,
    ) -> list[str]:
        """Find papers connected to given entities, ranked by connection count."""
        if not entity_ids:
            return []

        try:
            conn = self._graph._conn
            placeholders = ",".join(["%s"] * len(entity_ids))
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT paper_id,
                           COUNT(DISTINCT entity_id) as entity_count,
                           AVG(confidence) as avg_confidence
                    FROM paper_entity_edges
                    WHERE entity_id IN ({placeholders})
                    GROUP BY paper_id
                    ORDER BY entity_count DESC, avg_confidence DESC
                    LIMIT %s
                    """,
                    [*entity_ids, max_papers],
                )
                return [row[0] for row in cur.fetchall()]

        except Exception as e:
            logger.warning("Paper-entity lookup failed: %s", e)
            return []
