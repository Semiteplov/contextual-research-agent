from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import psycopg2
from psycopg2.extensions import connection as PGConnection

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.common.settings import get_settings
from contextual_research_agent.db.errors import DBConnectionError

if TYPE_CHECKING:
    from psycopg2.extensions import cursor as PGCursor

logger = get_logger(__name__)


def get_connection(
    db_name: str | None = None,
    autocommit: bool = False,
) -> PGConnection:
    """
    Create a new database connection.

    Args:
        db_name: Database name override. If None, uses settings.postgres_db.
        autocommit: If True, connection operates in autocommit mode.

    Returns:
        PostgreSQL connection object.

    Raises:
        DBConnectionError: If connection cannot be established.
    """
    settings = get_settings()

    try:
        conn = psycopg2.connect(
            host=settings.postgres.host,
            port=settings.postgres.port,
            user=settings.postgres.user,
            password=settings.postgres.password.get_secret_value(),
            dbname=db_name or settings.postgres.db,
        )
        conn.autocommit = autocommit
        return conn
    except psycopg2.Error as e:
        logger.exception("Failed to connect to database")
        raise DBConnectionError(f"Failed to connect to database: {e}") from e


@contextmanager
def get_connection_context(
    db_name: str | None = None,
    autocommit: bool = False,
) -> Generator[PGConnection, None, None]:
    """
    Context manager for database connections with automatic cleanup.

    Commits on successful exit, rolls back on exception.

    Example:
        with get_connection_context() as conn:
            repo = PapersRepository(conn)
            repo.upsert(papers)
    """
    conn = get_connection(db_name=db_name or "arxiv", autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor(
    conn: PGConnection,
) -> Generator["PGCursor", None, None]:
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
