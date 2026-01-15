from collections.abc import Sequence
from typing import TYPE_CHECKING

import psycopg2.extras

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.arxiv.parser import PaperRow
from contextual_research_agent.db.errors import QueryError
from contextual_research_agent.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

logger = get_logger(__name__)


class PapersMetadataRepository(BaseRepository):
    """Repository for arXiv papers metadata operations."""

    TABLE_NAME = "arxiv_papers_metadata"
    DEFAULT_PAGE_SIZE = 2000

    _UPSERT_SQL = """
        INSERT INTO arxiv_papers_metadata (
            arxiv_id, title, abstract, authors, categories, primary_category,
            doi, journal_ref, update_date, latest_version, latest_version_created
        )
        VALUES %s
        ON CONFLICT (arxiv_id) DO UPDATE SET
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            authors = EXCLUDED.authors,
            categories = EXCLUDED.categories,
            primary_category = EXCLUDED.primary_category,
            doi = EXCLUDED.doi,
            journal_ref = EXCLUDED.journal_ref,
            update_date = EXCLUDED.update_date,
            latest_version = EXCLUDED.latest_version,
            latest_version_created = EXCLUDED.latest_version_created,
            ingested_at = NOW()
    """

    def __init__(self, conn: "PGConnection") -> None:
        super().__init__(conn)

    def upsert(
        self,
        papers: Sequence[PaperRow],
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> int:
        """
        Insert or update papers metadata.

        Args:
            papers: Sequence of paper records to upsert.
            page_size: Batch size for bulk insert.

        Returns:
            Number of papers processed.

        Raises:
            QueryError: If database operation fails.
        """
        if not papers:
            return 0

        values = [self._paper_to_tuple(p) for p in papers]

        try:
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    self._UPSERT_SQL,
                    values,
                    page_size=page_size,
                )
            logger.debug("Upserted %d papers", len(papers))
            return len(papers)
        except psycopg2.Error as e:
            logger.exception("Failed to upsert papers batch")
            raise QueryError(f"Failed to upsert papers: {e}") from e

    def count(self) -> int:
        """Get total number of papers in the database."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}")
            result = cur.fetchone()
            return result[0] if result else 0

    def get_by_id(self, arxiv_id: str) -> PaperRow | None:
        """
        Get paper by arXiv ID.

        Args:
            arxiv_id: The arXiv identifier.

        Returns:
            PaperRow if found, None otherwise.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT arxiv_id, title, abstract, authors, categories,
                       primary_category, doi, journal_ref, update_date,
                       latest_version, latest_version_created
                FROM {self.TABLE_NAME}
                WHERE arxiv_id = %s
                """,
                (arxiv_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            return PaperRow(
                arxiv_id=row[0],
                title=row[1],
                abstract=row[2],
                authors=row[3],
                categories=row[4],
                primary_category=row[5],
                doi=row[6],
                journal_ref=row[7],
                update_date=row[8],
                latest_version=row[9],
                latest_version_created=row[10],
            )

    @staticmethod
    def _paper_to_tuple(paper: PaperRow) -> tuple:
        """Convert PaperRow to tuple for bulk insert."""
        return (
            paper.arxiv_id,
            paper.title,
            paper.abstract,
            paper.authors,
            paper.categories,
            paper.primary_category,
            paper.doi,
            paper.journal_ref,
            paper.update_date,
            paper.latest_version,
            paper.latest_version_created,
        )
