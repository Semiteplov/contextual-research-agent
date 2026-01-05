import logging
from pathlib import Path

import psycopg2

from data_pipeline.arxiv.constants import ML_CATEGORIES
from data_pipeline.arxiv.parser import PaperRow, is_ml_paper, iter_papers_jsonl
from db import get_pg_dsn
from db.arxiv.metadata import upsert_papers

logger = logging.getLogger(__name__)

METADATA_PATH = ".cache/kaggle/extracted/arxiv-metadata-oai-snapshot.json"
DB_BATCH_SIZE = 5_000


def _process(*, metadata_path: Path, conn) -> tuple[int, int]:
    """
    Core ingestion loop:
    - stream JSONL
    - filter ML categories
    - upsert to Postgres in batches

    Returns:
        (total_rows, kept_rows)
    """
    total = 0
    kept = 0

    db_buf: list[PaperRow] = []

    for row in iter_papers_jsonl(metadata_path):
        total += 1
        if not is_ml_paper(row.categories, ML_CATEGORIES):
            continue

        kept += 1
        db_buf.append(row)

        if len(db_buf) >= DB_BATCH_SIZE:
            _flush_db(conn, db_buf)
            db_buf.clear()

    if db_buf:
        _flush_db(conn, db_buf)
        db_buf.clear()

    return total, kept


def _flush_db(conn, buf: list[PaperRow]) -> int:
    upserted = upsert_papers(conn, buf)
    conn.commit()
    logger.info("DB upserted: %d rows", upserted)
    return upserted


def ingest_arxiv_metadata() -> None:
    metadata_path = Path(METADATA_PATH).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    logger.info("Starting ingestion (no snapshots)")
    logger.info("Metadata input: %s", metadata_path)
    logger.info("Categories: %s", sorted(ML_CATEGORIES))
    logger.info("DB batch size: %d", DB_BATCH_SIZE)

    conn = psycopg2.connect(get_pg_dsn("arxiv"))
    conn.autocommit = False

    try:
        total, kept = _process(metadata_path=metadata_path, conn=conn)
        logger.info("Done. total=%d kept=%d", total, kept)

    except Exception:
        conn.rollback()
        logger.exception("Ingestion failed; rolled back")
        raise
    finally:
        conn.close()
