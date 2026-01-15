import time
from datetime import UTC, datetime, timedelta

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.arxiv.client import ArxivClient, ArxivClientConfig
from contextual_research_agent.data.arxiv.constants import ML_CATEGORIES
from contextual_research_agent.data.arxiv.converter import entries_to_paper_rows
from contextual_research_agent.db import PapersMetadataRepository, SyncStateRepository

logger = get_logger(__name__)

CATEGORY_DELAY_SECONDS = 3.0


def _sync_category(
    conn,
    client: ArxivClient,
    category: str,
) -> int:
    """
    Sync single category: fetch from API and upsert to DB.

    Returns:
        Number of papers upserted.
    """
    papers_repo = PapersMetadataRepository(conn)
    sync_repo = SyncStateRepository(conn)

    # Get last sync timestamp
    last_synced_at = sync_repo.get(category)
    logger.info("Category=%s last_synced_at=%s", category, last_synced_at)

    # Fetch new entries from arXiv
    entries, newest_published = client.fetch_category(category, last_synced_at)

    if not entries:
        logger.info("Category=%s: no new entries", category)
        return 0

    # Convert and upsert
    paper_rows = entries_to_paper_rows(entries, fallback_category=category)
    upserted = papers_repo.upsert(paper_rows)

    # Update sync state
    if newest_published is not None:
        sync_repo.upsert(category, newest_published)

    logger.info(
        "Category=%s finished: upserted=%d newest_published=%s",
        category,
        upserted,
        newest_published,
    )
    return upserted


def fetch_and_upsert_metadata() -> None:
    """
    Main task: fetch metadata for all ML categories.

    Processes each category in a separate transaction.
    Raises RuntimeError if any category fails.
    """
    hook = PostgresHook(postgres_conn_id="arxiv_postgres")
    conn = hook.get_conn()
    conn.autocommit = False

    client = ArxivClient(ArxivClientConfig())

    total_upserted = 0
    failed_categories: list[str] = []

    try:
        for category in sorted(ML_CATEGORIES):
            try:
                upserted = _sync_category(conn, client, category)
                conn.commit()
                total_upserted += upserted

            except Exception:
                conn.rollback()
                logger.exception("Category=%s failed; transaction rolled back", category)
                failed_categories.append(category)

            time.sleep(CATEGORY_DELAY_SECONDS)

    finally:
        conn.close()

    logger.info(
        "Weekly ingestion complete: total_upserted=%d failed=%s",
        total_upserted,
        failed_categories or "none",
    )

    if failed_categories:
        raise RuntimeError(f"Failed categories: {failed_categories}")


with DAG(
    dag_id="arxiv_weekly_metadata",
    description="Weekly fetch of arXiv metadata for ML categories",
    start_date=datetime(2026, 1, 11, tzinfo=UTC),
    schedule="0 3 * * 1",  # Mondays 03:00 UTC
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["arxiv", "metadata", "weekly"],
) as dag:
    fetch_task = PythonOperator(
        task_id="fetch_and_upsert_metadata",
        python_callable=fetch_and_upsert_metadata,
    )
