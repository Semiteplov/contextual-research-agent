from __future__ import annotations

import asyncio
import concurrent.futures
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow

from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EntityQualityMetrics:
    """Quality metrics for entity extraction on a single document."""

    total_entities: int = 0
    entity_type_distribution: dict[str, int] = field(default_factory=dict)
    entity_density: float = 0.0
    mean_confidence: float = 0.0
    low_confidence_count: int = 0
    empty_calls: int = 0
    total_calls: int = 0
    empty_call_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_entities": self.total_entities,
            "entity_type_distribution": self.entity_type_distribution,
            "entity_density": round(self.entity_density, 3),
            "mean_confidence": round(self.mean_confidence, 3),
            "low_confidence_count": self.low_confidence_count,
            "empty_call_rate": round(self.empty_call_rate, 3),
        }


def compute_entity_quality(
    entities: list,
    num_chunks: int,
    llm_calls: int,
    empty_calls: int = 0,
) -> EntityQualityMetrics:
    """Compute quality metrics from entity extraction results."""
    if not entities:
        return EntityQualityMetrics(total_calls=llm_calls, empty_calls=empty_calls)

    type_dist: dict[str, int] = {}
    confidences: list[float] = []
    low_conf = 0

    for e in entities:
        etype = getattr(e, "entity_type", "unknown")
        type_dist[etype] = type_dist.get(etype, 0) + 1
        conf = getattr(e, "confidence", 1.0)
        confidences.append(conf)
        if conf < 0.5:
            low_conf += 1

    return EntityQualityMetrics(
        total_entities=len(entities),
        entity_type_distribution=type_dist,
        entity_density=len(entities) / num_chunks if num_chunks > 0 else 0,
        mean_confidence=sum(confidences) / len(confidences) if confidences else 0,
        low_confidence_count=low_conf,
        empty_calls=empty_calls,
        total_calls=llm_calls,
        empty_call_rate=empty_calls / llm_calls if llm_calls > 0 else 0,
    )


@dataclass
class CorpusAnalyticsReport:
    """Comprehensive analytics for an ingested corpus."""

    dataset_name: str = ""

    # Corpus size
    total_documents: int = 0
    total_chunks: int = 0

    # Knowledge graph
    total_entities: int = 0
    entity_type_counts: dict[str, int] = field(default_factory=dict)
    total_citation_edges: int = 0
    total_entity_edges: int = 0
    papers_with_citations: int = 0
    papers_with_entities: int = 0

    # Citation graph analysis
    mean_citations_per_paper: float = 0.0
    max_citations: int = 0
    in_corpus_citation_rate: float = 0.0
    top_cited_papers: list[dict[str, Any]] = field(default_factory=list)

    # Entity analysis
    mean_entities_per_paper: float = 0.0
    top_entities: list[dict[str, Any]] = field(default_factory=list)
    entity_cooccurrence_pairs: int = 0

    # Paper-level index
    paper_index_count: int = 0
    paper_index_coverage: float = 0.0

    # Vector store
    chunk_index_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "knowledge_graph": {
                "total_entities": self.total_entities,
                "entity_type_counts": self.entity_type_counts,
                "total_citation_edges": self.total_citation_edges,
                "total_entity_edges": self.total_entity_edges,
                "papers_with_citations": self.papers_with_citations,
                "papers_with_entities": self.papers_with_entities,
            },
            "citation_graph": {
                "mean_citations_per_paper": round(self.mean_citations_per_paper, 2),
                "max_citations": self.max_citations,
                "in_corpus_citation_rate": round(self.in_corpus_citation_rate, 3),
                "top_cited_papers": self.top_cited_papers[:10],
            },
            "entity_analysis": {
                "mean_entities_per_paper": round(self.mean_entities_per_paper, 2),
                "top_entities": self.top_entities[:20],
                "entity_cooccurrence_pairs": self.entity_cooccurrence_pairs,
            },
            "indexes": {
                "chunk_index_count": self.chunk_index_count,
                "paper_index_count": self.paper_index_count,
                "paper_index_coverage": round(self.paper_index_coverage, 3),
            },
        }

    def format(self) -> str:
        """Human-readable report."""
        lines = [
            "",
            "=" * 70,
            f"  Corpus Analytics: {self.dataset_name}",
            "=" * 70,
            "",
            "  Corpus size:",
            f"    Documents:           {self.total_documents}",
            f"    Chunks:              {self.total_chunks}",
            "",
            "  Knowledge graph:",
            f"    Entities:            {self.total_entities}",
            f"    Citation edges:      {self.total_citation_edges}",
            f"    Entity edges:        {self.total_entity_edges}",
            f"    Papers w/ citations: {self.papers_with_citations}",
            f"    Papers w/ entities:  {self.papers_with_entities}",
        ]

        if self.entity_type_counts:
            lines.append("")
            lines.append("  Entity types:")
            for etype, count in sorted(self.entity_type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {etype:<15s} {count:>6d}")

        lines.extend(
            [
                "",
                "  Citation graph:",
                f"    Mean citations/paper: {self.mean_citations_per_paper:.1f}",
                f"    Max citations:        {self.max_citations}",
                f"    In-corpus rate:       {self.in_corpus_citation_rate:.1%}",
            ]
        )

        if self.top_cited_papers:
            lines.append("")
            lines.append("  Top cited papers (in corpus):")
            for p in self.top_cited_papers[:5]:
                lines.append(
                    f"    [{p.get('cited_by', 0):>3d}x] {p.get('paper_id', '?'):<20s} "
                    f"{p.get('title', '?')[:50]}"
                )

        if self.top_entities:
            lines.append("")
            lines.append("  Top entities:")
            for e in self.top_entities[:10]:
                lines.append(
                    f"    [{e.get('paper_count', 0):>3d} papers] "
                    f"{e.get('type', '?'):<10s} {e.get('name', '?')}"
                )

        lines.extend(
            [
                "",
                "  Indexes:",
                f"    Chunk vectors:       {self.chunk_index_count}",
                f"    Paper vectors:       {self.paper_index_count}",
                f"    Paper coverage:      {self.paper_index_coverage:.1%}",
                "=" * 70,
            ]
        )

        return "\n".join(lines)

    def log_to_mlflow(self, run_name: str | None = None) -> None:
        """Log corpus analytics as MLflow metrics."""
        try:
            name = run_name or f"analytics_{self.dataset_name}"
            with mlflow.start_run(run_name=name):
                mlflow.log_param("dataset_name", self.dataset_name)
                mlflow.log_param("analysis_type", "corpus_analytics")

                mlflow.log_metrics(
                    {
                        "corpus/documents": float(self.total_documents),
                        "corpus/chunks": float(self.total_chunks),
                        "graph/entities": float(self.total_entities),
                        "graph/citation_edges": float(self.total_citation_edges),
                        "graph/entity_edges": float(self.total_entity_edges),
                        "graph/papers_with_citations": float(self.papers_with_citations),
                        "graph/papers_with_entities": float(self.papers_with_entities),
                        "citations/mean_per_paper": self.mean_citations_per_paper,
                        "citations/in_corpus_rate": self.in_corpus_citation_rate,
                        "entities/mean_per_paper": self.mean_entities_per_paper,
                        "entities/cooccurrence_pairs": float(self.entity_cooccurrence_pairs),
                        "index/chunk_count": float(self.chunk_index_count),
                        "index/paper_count": float(self.paper_index_count),
                        "index/paper_coverage": self.paper_index_coverage,
                    }
                )

                # Entity type breakdown
                for etype, count in self.entity_type_counts.items():
                    mlflow.log_metric(f"entity_type/{etype}", float(count))

                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "corpus_analytics.json"
                    with Path.open(path, "w") as f:
                        json.dump(self.to_dict(), f, indent=2, default=str)
                    mlflow.log_artifact(str(path))

        except ImportError:
            logger.warning("MLflow not available, skipping analytics logging")
        except Exception as e:
            logger.exception("Failed to log analytics to MLflow: %s", e)


class IngestionAnalytics:
    """
    Compute corpus-level analytics from PostgreSQL + Qdrant.

    Args:
        graph_repo: KnowledgeGraphRepository instance.
        chunk_store: QdrantStore for chunk-level vectors.
        paper_store: QdrantStore for paper-level vectors (optional).
    """

    def __init__(self, graph_repo, chunk_store=None, paper_store=None):
        self._graph = graph_repo
        self._chunk_store = chunk_store
        self._paper_store = paper_store

    def compute(self, dataset_name: str = "") -> CorpusAnalyticsReport:
        """Compute full corpus analytics report."""
        report = CorpusAnalyticsReport(dataset_name=dataset_name)

        # Knowledge graph stats
        try:
            stats = self._graph.get_graph_stats()
            report.total_entities = stats.get("entities", 0)
            report.entity_type_counts = stats.get("entity_types", {})
            report.total_citation_edges = stats.get("citation_edges", 0)
            report.total_entity_edges = stats.get("paper_entity_edges", 0)
            report.papers_with_citations = stats.get("papers_with_citations", 0)
        except Exception as e:
            logger.warning("Failed to get graph stats: %s", e)

        # Citation analysis
        try:
            self._compute_citation_stats(report)
        except Exception as e:
            logger.warning("Failed to compute citation stats: %s", e)

        # Entity analysis
        try:
            self._compute_entity_stats(report)
        except Exception as e:
            logger.warning("Failed to compute entity stats: %s", e)

        # Vector store stats
        try:
            self._compute_index_stats(report)
        except Exception as e:
            logger.warning("Failed to compute index stats: %s", e)

        return report

    def _compute_citation_stats(self, report: CorpusAnalyticsReport) -> None:
        """Compute citation graph statistics."""
        conn = self._graph._conn
        with conn.cursor() as cur:
            cur.execute("""
                SELECT citing_paper_id, COUNT(*) as cnt
                FROM citation_edges
                GROUP BY citing_paper_id
            """)
            rows = cur.fetchall()
            if rows:
                counts = [r[1] for r in rows]
                report.mean_citations_per_paper = sum(counts) / len(counts)
                report.max_citations = max(counts)

            cur.execute("""
                SELECT ce.cited_paper_id,
                       COUNT(*) as cited_by,
                       ce.cited_title
                FROM citation_edges ce
                GROUP BY ce.cited_paper_id, ce.cited_title
                ORDER BY cited_by DESC
                LIMIT 20
            """)
            report.top_cited_papers = [
                {"paper_id": r[0], "cited_by": r[1], "title": r[2] or ""} for r in cur.fetchall()
            ]

            cur.execute("""
                SELECT
                    COUNT(DISTINCT ce.cited_paper_id) as total_cited,
                    COUNT(DISTINCT am.arxiv_id) as in_corpus
                FROM citation_edges ce
                LEFT JOIN arxiv_papers_metadata am
                    ON ce.cited_paper_id = am.arxiv_id
                WHERE ce.cited_id_type = 'arxiv'
            """)
            row = cur.fetchone()
            if row and row[0] > 0:
                report.in_corpus_citation_rate = row[1] / row[0]

    def _compute_entity_stats(self, report: CorpusAnalyticsReport) -> None:
        """Compute entity statistics."""
        conn = self._graph._conn
        with conn.cursor() as cur:
            # Papers with entities
            cur.execute("SELECT COUNT(DISTINCT paper_id) FROM paper_entity_edges")
            row = cur.fetchone()
            report.papers_with_entities = row[0] if row else 0

            # Mean entities per paper
            cur.execute("""
                SELECT paper_id, COUNT(DISTINCT entity_id) as cnt
                FROM paper_entity_edges
                GROUP BY paper_id
            """)
            rows = cur.fetchall()
            if rows:
                counts = [r[1] for r in rows]
                report.mean_entities_per_paper = sum(counts) / len(counts)

            # Top entities (by paper count)
            cur.execute("""
                SELECT e.name, e.entity_type,
                       COUNT(DISTINCT pe.paper_id) as paper_count
                FROM entities e
                JOIN paper_entity_edges pe ON e.id = pe.entity_id
                GROUP BY e.name, e.entity_type
                ORDER BY paper_count DESC
                LIMIT 30
            """)
            report.top_entities = [
                {"name": r[0], "type": r[1], "paper_count": r[2]} for r in cur.fetchall()
            ]

            # Entity co-occurrence pairs
            cur.execute("""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT pe1.paper_id, pe2.paper_id
                    FROM paper_entity_edges pe1
                    JOIN paper_entity_edges pe2
                        ON pe1.entity_id = pe2.entity_id
                        AND pe1.paper_id < pe2.paper_id
                ) sub
            """)
            row = cur.fetchone()
            report.entity_cooccurrence_pairs = row[0] if row else 0

    def _compute_index_stats(self, report: CorpusAnalyticsReport) -> None:
        """Compute vector index statistics (sync wrappers)."""

        async def _get_stats():
            if self._chunk_store:
                stats = await self._chunk_store.get_stats()
                report.chunk_index_count = stats.get("points_count", 0)
                report.total_chunks = report.chunk_index_count

            if self._paper_store:
                stats = await self._paper_store.get_stats()
                report.paper_index_count = stats.get("points_count", 0)

            if report.total_documents > 0:
                report.paper_index_coverage = report.paper_index_count / report.total_documents
            elif report.papers_with_citations > 0:
                report.paper_index_coverage = (
                    report.paper_index_count / report.papers_with_citations
                )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, _get_stats()).result()
            else:
                asyncio.run(_get_stats())
        except Exception:
            pass
