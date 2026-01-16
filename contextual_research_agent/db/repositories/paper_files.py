import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import psycopg2

from contextual_research_agent.data.arxiv.downloader import FileType
from contextual_research_agent.db.errors import QueryError
from contextual_research_agent.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PaperFileRecord:
    id: int
    arxiv_id: str
    storage_path: str
    file_type: str
    file_size_bytes: int | None
    checksum_sha256: str | None
    downloaded_at: datetime


class PaperFilesRepository(BaseRepository):
    TABLE_NAME = "arxiv_papers"

    def __init__(self, conn: "PGConnection") -> None:
        super().__init__(conn)

    def insert(
        self,
        arxiv_id: str,
        storage_path: str,
        file_type: FileType,
        file_size_bytes: int | None = None,
        checksum_sha256: str | None = None,
    ) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO arxiv_papers
                        (arxiv_id, storage_path, file_type, file_size_bytes, checksum_sha256)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (arxiv_id, storage_path, file_type.value, file_size_bytes, checksum_sha256),
                )
                result = cur.fetchone()
                return result[0] if result else 0

        except psycopg2.Error as e:
            logger.exception("Failed to insert paper file: %s", arxiv_id)
            raise QueryError(f"Insert failed: {e}") from e

    def get_by_arxiv_id(
        self,
        arxiv_id: str,
        file_type: FileType | None = None,
    ) -> PaperFileRecord | None:
        query = """
            SELECT id, arxiv_id, storage_path, file_type,
                   file_size_bytes, checksum_sha256, downloaded_at
            FROM arxiv_papers
            WHERE arxiv_id = %s
        """
        params: list = [arxiv_id]

        if file_type is not None:
            query += " AND file_type = %s"
            params.append(file_type.value)

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()

            if row is None:
                return None

            return PaperFileRecord(
                id=row[0],
                arxiv_id=row[1],
                storage_path=row[2],
                file_type=row[3],
                file_size_bytes=row[4],
                checksum_sha256=row[5],
                downloaded_at=row[6],
            )

    def exists(self, arxiv_id: str, file_type: FileType | None = None) -> bool:
        query = "SELECT 1 FROM arxiv_papers WHERE arxiv_id = %s"
        params: list = [arxiv_id]

        if file_type is not None:
            query += " AND file_type = %s"
            params.append(file_type.value)

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone() is not None

    def get_missing_arxiv_ids(
        self,
        file_type: FileType,
        limit: int = 1000,
    ) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.arxiv_id
                FROM arxiv_papers_metadata m
                LEFT JOIN arxiv_papers p
                    ON m.arxiv_id = p.arxiv_id AND p.file_type = %s
                WHERE p.id IS NULL
                ORDER BY m.update_date DESC
                LIMIT %s
                """,
                (file_type.value, limit),
            )
            return [row[0] for row in cur.fetchall()]

    def count(self, file_type: FileType | None = None) -> int:
        query = "SELECT COUNT(*) FROM arxiv_papers"
        params: list = []

        if file_type is not None:
            query += " WHERE file_type = %s"
            params.append(file_type.value)

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            result = cur.fetchone()
            return result[0] if result else 0
