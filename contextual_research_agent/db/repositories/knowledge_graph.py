from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psycopg2.extras import execute_values

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.db.repositories.base import BaseRepository
from contextual_research_agent.ingestion.extraction.citation_extractor import (
    CitationEdge,
    CitationExtractionResult,
)

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

logger = get_logger(__name__)


class KnowledgeGraphRepository(BaseRepository):
    """PostgreSQL repository for the knowledge graph."""

    def __init__(self, conn: PGConnection) -> None:
        super().__init__(conn)

    def store_citation_edges(self, edges: list[CitationEdge]) -> int:
        """
        Upsert citation edges. Returns count of inserted/updated rows.

        ON CONFLICT: updates context and metadata.
        """
        if not edges:
            return 0

        records = [
            (
                e.citing_paper_id,
                e.cited_paper_id,
                e.cited_id_type,
                e.context[:2000] if e.context else None,
                e.section or None,
                e.section_type or None,
                e.ref_key or None,
                e.cited_title[:1000] if e.cited_title else None,
                e.cited_authors[:1000] if e.cited_authors else None,
                e.cited_year or None,
            )
            for e in edges
        ]

        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO citation_edges (
                    citing_paper_id, cited_paper_id, cited_id_type,
                    context, section, section_type,
                    ref_key, cited_title, cited_authors, cited_year
                )
                VALUES %s
                ON CONFLICT (citing_paper_id, cited_paper_id)
                DO UPDATE SET
                    context = COALESCE(EXCLUDED.context, citation_edges.context),
                    section = COALESCE(EXCLUDED.section, citation_edges.section),
                    section_type = COALESCE(EXCLUDED.section_type, citation_edges.section_type),
                    cited_title = COALESCE(EXCLUDED.cited_title, citation_edges.cited_title),
                    cited_authors = COALESCE(EXCLUDED.cited_authors, citation_edges.cited_authors),
                    cited_year = COALESCE(EXCLUDED.cited_year, citation_edges.cited_year)
                """,
                records,
            )

        count = len(records)
        logger.info("Stored %d citation edges", count)
        return count

    def store_extraction_result(self, result: CitationExtractionResult) -> int:
        """Convenience: store all edges from a CitationExtractionResult."""
        return self.store_citation_edges(result.edges)

    def get_citing_papers(
        self,
        paper_id: str,
        section_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get papers that cite the given paper.

        Args:
            paper_id: arxiv_id of the cited paper.
            section_type: Optional filter by section where citation occurs.
        """
        query = """
            SELECT citing_paper_id, context, section, section_type,
                   cited_title, cited_year
            FROM citation_edges
            WHERE cited_paper_id = %s
        """
        params: list = [paper_id]

        if section_type:
            query += " AND section_type = %s"
            params.append(section_type)

        query += " ORDER BY cited_year DESC NULLS LAST"

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            {
                "citing_paper_id": r[0],
                "context": r[1],
                "section": r[2],
                "section_type": r[3],
                "cited_title": r[4],
                "cited_year": r[5],
            }
            for r in rows
        ]

    def get_cited_papers(self, paper_id: str) -> list[dict[str, Any]]:
        """Get papers cited by the given paper."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT cited_paper_id, cited_id_type, context,
                       section_type, cited_title, cited_year
                FROM citation_edges
                WHERE citing_paper_id = %s
                ORDER BY ref_key
                """,
                (paper_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "cited_paper_id": r[0],
                "cited_id_type": r[1],
                "context": r[2],
                "section_type": r[3],
                "cited_title": r[4],
                "cited_year": r[5],
            }
            for r in rows
        ]

    def get_citation_neighborhood(
        self,
        paper_id: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Get N-hop citation neighborhood using recursive CTE.

        Returns papers within max_depth hops (both citing and cited).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE neighborhood AS (
                    -- Seed: direct citations (both directions)
                    SELECT citing_paper_id AS paper_id, 1 AS depth, 'cites' AS direction
                    FROM citation_edges WHERE cited_paper_id = %s
                    UNION
                    SELECT cited_paper_id, 1, 'cited_by'
                    FROM citation_edges WHERE citing_paper_id = %s
                    UNION ALL
                    -- Expand: follow citation edges
                    SELECT
                        CASE WHEN n.direction = 'cites'
                            THEN ce.citing_paper_id
                            ELSE ce.cited_paper_id
                        END,
                        n.depth + 1,
                        n.direction
                    FROM neighborhood n
                    JOIN citation_edges ce ON (
                        (n.direction = 'cites' AND ce.cited_paper_id = n.paper_id)
                        OR
                        (n.direction = 'cited_by' AND ce.citing_paper_id = n.paper_id)
                    )
                    WHERE n.depth < %s
                )
                SELECT DISTINCT paper_id, MIN(depth) AS min_depth
                FROM neighborhood
                WHERE paper_id != %s
                GROUP BY paper_id
                ORDER BY min_depth, paper_id
                """,
                (paper_id, paper_id, max_depth, paper_id),
            )
            rows = cur.fetchall()

        return [{"paper_id": r[0], "depth": r[1]} for r in rows]

    def get_citation_stats(self, paper_id: str) -> dict[str, int]:
        """Get citation counts for a paper."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM citation_edges WHERE citing_paper_id = %s",
                (paper_id,),
            )
            row = cur.fetchone()
            cites_count = row[0] if row else 0

            cur.execute(
                "SELECT COUNT(*) FROM citation_edges WHERE cited_paper_id = %s",
                (paper_id,),
            )
            row = cur.fetchone()
            cited_by_count = row[0] if row else 0

        return {
            "cites": cites_count,
            "cited_by": cited_by_count,
        }

    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        description: str | None = None,
        source: str = "extracted",
        aliases: list[str] | None = None,
    ) -> int:
        """
        Insert or get entity ID. Returns entity ID.

        Deduplication by (normalized_name, entity_type).
        """
        normalized = name.strip().lower()

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entities (name, entity_type, normalized_name, description, source, aliases)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (normalized_name, entity_type)
                DO UPDATE SET
                    updated_at = NOW(),
                    description = COALESCE(EXCLUDED.description, entities.description)
                RETURNING id
                """,  # noqa: E501
                (name, entity_type, normalized, description, source, aliases),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def upsert_entities_batch(
        self,
        entities: list[tuple[str, str]],
        source: str = "extracted",
    ) -> dict[tuple[str, str], int]:
        """
        Batch upsert entities. Returns mapping (name, type) → entity_id.
        """
        result: dict[tuple[str, str], int] = {}
        for name, entity_type in entities:
            eid = self.upsert_entity(name, entity_type, source=source)
            result[(name, entity_type)] = eid
        return result

    def get_entity_id(self, name: str, entity_type: str) -> int | None:
        """Get entity ID by name and type."""
        normalized = name.strip().lower()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE normalized_name = %s AND entity_type = %s",
                (normalized, entity_type),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def store_paper_entity_edges(
        self,
        paper_id: str,
        edges: list[dict[str, Any]],
    ) -> int:
        """
        Store paper → entity edges.

        Args:
            paper_id: arxiv_id.
            edges: List of dicts with keys:
                entity_id, relation, confidence, evidence, section_type, extraction_method
        """
        if not edges:
            return 0

        records = [
            (
                paper_id,
                e["entity_id"],
                e["relation"],
                e.get("confidence", 1.0),
                e.get("evidence"),
                e.get("section_type"),
                e.get("extraction_method", "rule"),
            )
            for e in edges
        ]

        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO paper_entity_edges (
                    paper_id, entity_id, relation,
                    confidence, evidence, section_type, extraction_method
                )
                VALUES %s
                ON CONFLICT (paper_id, entity_id, relation) DO UPDATE SET
                    confidence = GREATEST(EXCLUDED.confidence, paper_entity_edges.confidence),
                    evidence = COALESCE(EXCLUDED.evidence, paper_entity_edges.evidence)
                """,
                records,
            )

        count = len(records)
        logger.info("Stored %d paper-entity edges for %s", count, paper_id)
        return count

    def get_papers_by_entity(
        self,
        entity_name: str,
        entity_type: str | None = None,
        relation: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find papers related to a given entity."""
        query = """
            SELECT pe.paper_id, pe.relation, pe.confidence, pe.evidence,
                   e.name, e.entity_type
            FROM paper_entity_edges pe
            JOIN entities e ON pe.entity_id = e.id
            WHERE e.normalized_name = %s
        """
        params: list = [entity_name.strip().lower()]

        if entity_type:
            query += " AND e.entity_type = %s"
            params.append(entity_type)

        if relation:
            query += " AND pe.relation = %s"
            params.append(relation)

        query += " ORDER BY pe.confidence DESC"

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            {
                "paper_id": r[0],
                "relation": r[1],
                "confidence": r[2],
                "evidence": r[3],
                "entity_name": r[4],
                "entity_type": r[5],
            }
            for r in rows
        ]

    def get_papers_sharing_entities(
        self,
        paper_id: str,
        min_shared: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Find papers that share entities with the given paper.
        Ranked by number of shared entities (entity-based similarity).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    pe2.paper_id,
                    COUNT(DISTINCT pe2.entity_id) AS shared_count,
                    ARRAY_AGG(DISTINCT e.name) AS shared_entities
                FROM paper_entity_edges pe1
                JOIN paper_entity_edges pe2
                    ON pe1.entity_id = pe2.entity_id
                    AND pe1.paper_id != pe2.paper_id
                JOIN entities e ON pe1.entity_id = e.id
                WHERE pe1.paper_id = %s
                GROUP BY pe2.paper_id
                HAVING COUNT(DISTINCT pe2.entity_id) >= %s
                ORDER BY shared_count DESC
                """,
                (paper_id, min_shared),
            )
            rows = cur.fetchall()

        return [
            {
                "paper_id": r[0],
                "shared_count": r[1],
                "shared_entities": r[2],
            }
            for r in rows
        ]

    def get_graph_stats(self) -> dict[str, Any]:
        """Get overall knowledge graph statistics."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM entities")
            row = cur.fetchone()
            entity_count = row[0] if row else 0

            cur.execute("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")
            entity_types = dict(cur.fetchall())

            cur.execute("SELECT COUNT(*) FROM citation_edges")
            row = cur.fetchone()
            citation_count = row[0] if row else 0

            cur.execute("SELECT COUNT(*) FROM paper_entity_edges")
            row = cur.fetchone()
            paper_entity_count = row[0] if row else 0

            cur.execute("SELECT COUNT(DISTINCT citing_paper_id) FROM citation_edges")
            row = cur.fetchone()
            papers_with_citations = row[0] if row else 0

        return {
            "entities": entity_count,
            "entity_types": entity_types,
            "citation_edges": citation_count,
            "paper_entity_edges": paper_entity_count,
            "papers_with_citations": papers_with_citations,
        }
