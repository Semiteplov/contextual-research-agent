import logging
from collections.abc import Sequence

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PGConnection

from data_pipeline.arxiv.parser import PaperRow
from db.errors import DBError

logger = logging.getLogger(__name__)


UPSERT_SQL = """
INSERT INTO arxiv_papers_metadata(
  arxiv_id, title, abstract, authors, categories, primary_category,
  doi, journal_ref, update_date, latest_version, latest_version_created
)
VALUES %s
ON CONFLICT (arxiv_id) DO UPDATE SET
  title=EXCLUDED.title,
  abstract=EXCLUDED.abstract,
  authors=EXCLUDED.authors,
  categories=EXCLUDED.categories,
  primary_category=EXCLUDED.primary_category,
  doi=EXCLUDED.doi,
  journal_ref=EXCLUDED.journal_ref,
  update_date=EXCLUDED.update_date,
  latest_version=EXCLUDED.latest_version,
  latest_version_created=EXCLUDED.latest_version_created,
  ingested_at=NOW()
"""


def upsert_papers(conn: PGConnection, rows: Sequence[PaperRow]) -> int:
    if not rows:
        return 0

    values = [
        (
            r.arxiv_id,
            r.title,
            r.abstract,
            r.authors,
            r.categories,
            r.primary_category,
            r.doi,
            r.journal_ref,
            r.update_date,
            r.latest_version,
            r.latest_version_created,
        )
        for r in rows
    ]

    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, UPSERT_SQL, values, page_size=2000)
        return len(rows)
    except Exception as e:
        logger.exception("Failed to upsert batch into DB")
        raise DBError("Failed to upsert batch into DB") from e
