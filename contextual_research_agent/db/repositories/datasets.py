import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import psycopg2
from psycopg2.extras import execute_values

from contextual_research_agent.db.errors import QueryError
from contextual_research_agent.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PGConnection

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    id: int
    name: str
    description: str | None
    selection_criteria: dict
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    total_papers: int
    purpose: str
    version: int
    created_at: datetime
    is_active: bool


@dataclass(frozen=True, slots=True)
class DatasetStats:
    total: int
    train: int
    val: int
    test: int
    downloaded: int


@dataclass(frozen=True, slots=True)
class DatasetPaper:
    arxiv_id: str
    split: str
    storage_path: str | None


class DatasetsRepository(BaseRepository):
    TABLE_NAME = "datasets"

    def __init__(self, conn: "PGConnection") -> None:
        super().__init__(conn)

    def create(  # noqa: PLR0913
        self,
        name: str,
        arxiv_ids: list[str],
        selection_criteria: dict,
        description: str | None = None,
        purpose: str = "training",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        overwrite: bool = False,
    ) -> int:
        eps = 0.001
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > eps:
            raise ValueError("Split ratios must sum to 1.0")

        existing = self.get_by_name(name)
        if existing:
            if overwrite:
                logger.info("Overwriting existing dataset '%s'", name)
                self._delete_hard(name)
            else:
                raise ValueError(f"Dataset '{name}' already exists. Use overwrite=True to replace.")

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO datasets (
                        name, description, selection_criteria,
                        train_ratio, val_ratio, test_ratio, random_seed,
                        total_papers, purpose
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        name,
                        description,
                        json.dumps(selection_criteria),
                        train_ratio,
                        val_ratio,
                        test_ratio,
                        random_seed,
                        len(arxiv_ids),
                        purpose,
                    ),
                )
                result = cur.fetchone()
                dataset_id = result[0] if result else 0

        except psycopg2.Error as e:
            raise QueryError(f"Failed to create dataset: {e}") from e

        random.seed(random_seed)
        shuffled = arxiv_ids.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        records = []
        for i, arxiv_id in enumerate(shuffled):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"
            records.append((dataset_id, arxiv_id, split))

        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO dataset_papers (dataset_id, arxiv_id, split)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                records,
            )

        logger.info(
            "Created dataset '%s' (id=%d): train=%d, val=%d, test=%d",
            name,
            dataset_id,
            train_end,
            val_end - train_end,
            n - val_end,
        )

        return dataset_id

    def get_by_name(self, name: str) -> DatasetRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, description, selection_criteria,
                       train_ratio, val_ratio, test_ratio, random_seed,
                       total_papers, purpose, version, created_at, is_active
                FROM datasets
                WHERE name = %s AND is_active = TRUE
                """,
                (name,),
            )
            row = cur.fetchone()

        if row is None:
            return None

        return DatasetRecord(
            id=row[0],
            name=row[1],
            description=row[2],
            selection_criteria=row[3] or {},
            train_ratio=row[4],
            val_ratio=row[5],
            test_ratio=row[6],
            random_seed=row[7],
            total_papers=row[8],
            purpose=row[9],
            version=row[10],
            created_at=row[11],
            is_active=row[12],
        )

    def get_arxiv_ids(
        self,
        dataset_name: str,
        split: str | None = None,
    ) -> list[str]:
        with self._conn.cursor() as cur:
            if split:
                cur.execute(
                    """
                    SELECT dp.arxiv_id
                    FROM dataset_papers dp
                    JOIN datasets d ON dp.dataset_id = d.id
                    WHERE d.name = %s AND d.is_active = TRUE AND dp.split = %s
                    ORDER BY dp.arxiv_id
                    """,
                    (dataset_name, split),
                )
            else:
                cur.execute(
                    """
                    SELECT dp.arxiv_id
                    FROM dataset_papers dp
                    JOIN datasets d ON dp.dataset_id = d.id
                    WHERE d.name = %s AND d.is_active = TRUE
                    ORDER BY dp.arxiv_id
                    """,
                    (dataset_name,),
                )

            return [row[0] for row in cur.fetchall()]

    def get_undownloaded_arxiv_ids(self, dataset_name: str) -> list[str]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT dp.arxiv_id
                FROM dataset_papers dp
                JOIN datasets d ON dp.dataset_id = d.id
                LEFT JOIN arxiv_papers ap
                    ON dp.arxiv_id = ap.arxiv_id AND ap.file_type = 'pdf'
                WHERE d.name = %s
                  AND d.is_active = TRUE
                  AND ap.id IS NULL
                ORDER BY dp.arxiv_id
                """,
                (dataset_name,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_stats(self, dataset_name: str) -> DatasetStats | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE dp.split = 'train') as train,
                    COUNT(*) FILTER (WHERE dp.split = 'val') as val,
                    COUNT(*) FILTER (WHERE dp.split = 'test') as test,
                    COUNT(ap.id) as downloaded
                FROM dataset_papers dp
                JOIN datasets d ON dp.dataset_id = d.id
                LEFT JOIN arxiv_papers ap
                    ON dp.arxiv_id = ap.arxiv_id AND ap.file_type = 'pdf'
                WHERE d.name = %s AND d.is_active = TRUE
                """,
                (dataset_name,),
            )
            row = cur.fetchone()

        if row is None or row[0] == 0:
            return None

        return DatasetStats(
            total=row[0],
            train=row[1],
            val=row[2],
            test=row[3],
            downloaded=row[4],
        )

    def get_papers_with_paths(
        self,
        dataset_name: str,
        split: str | None = None,
        only_downloaded: bool = False,
    ) -> list[DatasetPaper]:
        query = """
            SELECT dp.arxiv_id, dp.split, ap.storage_path
            FROM dataset_papers dp
            JOIN datasets d ON dp.dataset_id = d.id
            LEFT JOIN arxiv_papers ap
                ON dp.arxiv_id = ap.arxiv_id AND ap.file_type = 'pdf'
            WHERE d.name = %s AND d.is_active = TRUE
        """
        params: list = [dataset_name]

        if split:
            query += " AND dp.split = %s"
            params.append(split)

        if only_downloaded:
            query += " AND ap.id IS NOT NULL"

        query += " ORDER BY dp.split, dp.arxiv_id"

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            DatasetPaper(
                arxiv_id=row[0],
                split=row[1],
                storage_path=row[2],
            )
            for row in rows
        ]

    def list_datasets(self, purpose: str | None = None) -> list[DatasetRecord]:
        with self._conn.cursor() as cur:
            if purpose:
                cur.execute(
                    """
                    SELECT id, name, description, selection_criteria,
                           train_ratio, val_ratio, test_ratio, random_seed,
                           total_papers, purpose, version, created_at, is_active
                    FROM datasets
                    WHERE is_active = TRUE AND purpose = %s
                    ORDER BY created_at DESC
                    """,
                    (purpose,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, name, description, selection_criteria,
                           train_ratio, val_ratio, test_ratio, random_seed,
                           total_papers, purpose, version, created_at, is_active
                    FROM datasets
                    WHERE is_active = TRUE
                    ORDER BY created_at DESC
                    """
                )

            rows = cur.fetchall()

        return [
            DatasetRecord(
                id=row[0],
                name=row[1],
                description=row[2],
                selection_criteria=row[3] or {},
                train_ratio=row[4],
                val_ratio=row[5],
                test_ratio=row[6],
                random_seed=row[7],
                total_papers=row[8],
                purpose=row[9],
                version=row[10],
                created_at=row[11],
                is_active=row[12],
            )
            for row in rows
        ]

    def _delete_hard(self, name: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM dataset_papers
                WHERE dataset_id IN (SELECT id FROM datasets WHERE name = %s)
                """,
                (name,),
            )
            cur.execute("DELETE FROM datasets WHERE name = %s", (name,))
            return cur.rowcount > 0

    def delete(self, name: str, hard: bool = False) -> bool:
        if hard:
            return self._delete_hard(name)

        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE datasets SET is_active = FALSE, updated_at = NOW() WHERE name = %s",
                (name,),
            )
            return cur.rowcount > 0
