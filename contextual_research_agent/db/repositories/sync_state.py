from datetime import datetime
from typing import TYPE_CHECKING

import psycopg2

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.db.errors import QueryError
from contextual_research_agent.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

logger = get_logger(__name__)


class SyncStateRepository(BaseRepository):
    TABLE_NAME = "arxiv_category_sync_state"

    def __init__(self, conn: "PGConnection") -> None:
        super().__init__(conn)

    def upsert(self, category: str, last_synced_at: datetime) -> None:
        """
        Update sync state for a category.

        Args:
            category: arXiv category identifier (e.g., "cs.AI").
            last_synced_at: Timestamp of last successful sync.

        Raises:
            QueryError: If database operation fails.
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO arxiv_category_sync_state (category, last_synced_at)
                    VALUES (%s, %s)
                    ON CONFLICT (category)
                    DO UPDATE SET last_synced_at = EXCLUDED.last_synced_at
                    """,
                    (category, last_synced_at),
                )
            logger.debug("Updated sync state for category %s", category)
        except psycopg2.Error as e:
            logger.exception("Failed to upsert sync state for %s", category)
            raise QueryError(f"Failed to upsert sync state: {e}") from e

    def get(self, category: str) -> datetime | None:
        """
        Get last sync timestamp for a category.

        Args:
            category: arXiv category identifier.

        Returns:
            Last sync timestamp or None if never synced.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT last_synced_at FROM arxiv_category_sync_state WHERE category = %s",
                (category,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def get_all(self) -> dict[str, datetime]:
        """
        Get sync state for all categories.

        Returns:
            Dictionary mapping category to last sync timestamp.
        """
        with self._conn.cursor() as cur:
            cur.execute("SELECT category, last_synced_at FROM arxiv_category_sync_state")
            return {row[0]: row[1] for row in cur.fetchall()}
