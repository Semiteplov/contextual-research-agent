import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import psycopg2

from contextual_research_agent.data.extraction.pdf_extractor import ExtractionMethod
from contextual_research_agent.db.errors import QueryError
from contextual_research_agent.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractedTextRecord:
    id: int
    arxiv_id: str
    extraction_method: str
    storage_path: str
    num_pages: int | None
    num_characters: int | None
    num_words: int | None
    language: str | None
    status: str
    error_message: str | None
    created_at: datetime


class ExtractedTextsRepository(BaseRepository):
    TABLE_NAME = "extracted_texts"

    def __init__(self, conn: "PGConnection") -> None:
        super().__init__(conn)

    def insert(  # noqa: PLR0913
        self,
        arxiv_id: str,
        extraction_method: ExtractionMethod,
        storage_path: str,
        num_pages: int | None = None,
        num_characters: int | None = None,
        num_words: int | None = None,
        language: str | None = None,
        status: str = "completed",
        error_message: str | None = None,
    ) -> int:
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO extracted_texts (
                        arxiv_id, extraction_method, storage_path,
                        num_pages, num_characters, num_words,
                        language, status, error_message
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (arxiv_id, extraction_method) DO UPDATE SET
                        storage_path = EXCLUDED.storage_path,
                        num_pages = EXCLUDED.num_pages,
                        num_characters = EXCLUDED.num_characters,
                        num_words = EXCLUDED.num_words,
                        language = EXCLUDED.language,
                        status = EXCLUDED.status,
                        error_message = EXCLUDED.error_message,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        arxiv_id,
                        extraction_method.value,
                        storage_path,
                        num_pages,
                        num_characters,
                        num_words,
                        language,
                        status,
                        error_message,
                    ),
                )
                result = cur.fetchone()
                return result[0] if result else 0

        except psycopg2.Error as e:
            logger.exception("Failed to insert extracted text: %s", arxiv_id)
            raise QueryError(f"Insert failed: {e}") from e

    def get_by_arxiv_id(
        self,
        arxiv_id: str,
        method: ExtractionMethod | None = None,
    ) -> ExtractedTextRecord | None:
        with self._conn.cursor() as cur:
            if method:
                cur.execute(
                    """
                    SELECT id, arxiv_id, extraction_method, storage_path,
                           num_pages, num_characters, num_words, language,
                           status, error_message, created_at
                    FROM extracted_texts
                    WHERE arxiv_id = %s AND extraction_method = %s
                    """,
                    (arxiv_id, method.value),
                )
            else:
                cur.execute(
                    """
                    SELECT id, arxiv_id, extraction_method, storage_path,
                           num_pages, num_characters, num_words, language,
                           status, error_message, created_at
                    FROM extracted_texts
                    WHERE arxiv_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (arxiv_id,),
                )
            row = cur.fetchone()

        if row is None:
            return None

        return ExtractedTextRecord(
            id=row[0],
            arxiv_id=row[1],
            extraction_method=row[2],
            storage_path=row[3],
            num_pages=row[4],
            num_characters=row[5],
            num_words=row[6],
            language=row[7],
            status=row[8],
            error_message=row[9],
            created_at=row[10],
        )

    def exists(
        self,
        arxiv_id: str,
        method: ExtractionMethod | None = None,
    ) -> bool:
        with self._conn.cursor() as cur:
            if method:
                cur.execute(
                    "SELECT 1 FROM extracted_texts WHERE arxiv_id = %s AND extraction_method = %s",
                    (arxiv_id, method.value),
                )
            else:
                cur.execute(
                    "SELECT 1 FROM extracted_texts WHERE arxiv_id = %s",
                    (arxiv_id,),
                )
            return cur.fetchone() is not None

    def get_unextracted_arxiv_ids(
        self,
        dataset_name: str,
        method: ExtractionMethod,
        limit: int = 1000,
    ) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT dp.arxiv_id
                FROM dataset_papers dp
                JOIN datasets d ON dp.dataset_id = d.id
                JOIN arxiv_papers ap ON dp.arxiv_id = ap.arxiv_id AND ap.file_type = 'pdf'
                LEFT JOIN extracted_texts et
                    ON dp.arxiv_id = et.arxiv_id AND et.extraction_method = %s
                WHERE d.name = %s
                  AND d.is_active = TRUE
                  AND et.id IS NULL
                ORDER BY dp.arxiv_id
                LIMIT %s
                """,
                (method.value, dataset_name, limit),
            )
            return [row[0] for row in cur.fetchall()]

    def get_stats(self, dataset_name: str) -> dict:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(DISTINCT dp.arxiv_id) as total_in_dataset,
                    COUNT(DISTINCT ap.arxiv_id) as total_downloaded,
                    COUNT(DISTINCT et.arxiv_id) as total_extracted,
                    COUNT(DISTINCT et.arxiv_id) FILTER (WHERE et.status = 'completed') as completed,
                    COUNT(DISTINCT et.arxiv_id) FILTER (WHERE et.status = 'partial') as partial,
                    COUNT(DISTINCT et.arxiv_id) FILTER (WHERE et.status = 'failed') as failed,
                    AVG(et.num_pages) as avg_pages,
                    AVG(et.num_words) as avg_words
                FROM dataset_papers dp
                JOIN datasets d ON dp.dataset_id = d.id
                LEFT JOIN arxiv_papers ap ON dp.arxiv_id = ap.arxiv_id AND ap.file_type = 'pdf'
                LEFT JOIN extracted_texts et ON dp.arxiv_id = et.arxiv_id
                WHERE d.name = %s AND d.is_active = TRUE
                """,
                (dataset_name,),
            )
            row = cur.fetchone()

        return {
            "total_in_dataset": row[0] or 0,
            "total_downloaded": row[1] or 0,
            "total_extracted": row[2] or 0,
            "completed": row[3] or 0,
            "partial": row[4] or 0,
            "failed": row[5] or 0,
            "avg_pages": round(row[6], 1) if row[6] else 0,
            "avg_words": round(row[7], 0) if row[7] else 0,
        }
