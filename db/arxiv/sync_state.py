import logging
from datetime import datetime

from db.errors import DBError

logger = logging.getLogger(__name__)


def upsert_sync_state(conn, category: str, last_synced_at: datetime) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO arxiv_category_sync_state(category, last_synced_at)
                VALUES (%s, %s)
                ON CONFLICT (category) DO UPDATE SET last_synced_at=EXCLUDED.last_synced_at
                """,
                (category, last_synced_at),
            )
    except Exception as e:
        logger.exception("Failed to upsert batch into DB")
        raise DBError("Failed to upsert batch into DB") from e


def get_sync_state(conn, category: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT last_synced_at FROM arxiv_category_sync_state WHERE category=%s",
            (category,),
        )
        row = cur.fetchone()
        return row[0] if row else None
