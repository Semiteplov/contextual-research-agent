import logging
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.arxiv.parser import PaperRow

logger = logging.getLogger(__name__)


class ParquetWriteError(RuntimeError):
    pass


def _to_schema() -> pa.Schema:
    return pa.schema(
        [
            ("arxiv_id", pa.string()),
            ("title", pa.string()),
            ("abstract", pa.string()),
            ("authors", pa.string()),
            ("categories", pa.list_(pa.string())),
            ("primary_category", pa.string()),
            ("doi", pa.string()),
            ("journal_ref", pa.string()),
            ("update_date", pa.date32()),
            ("latest_version", pa.int32()),
            ("latest_version_created", pa.timestamp("ms", tz="UTC")),
        ]
    )


def _to_row(r: PaperRow) -> dict:
    def dt(v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)

    return {
        "arxiv_id": r.arxiv_id,
        "title": r.title,
        "abstract": r.abstract,
        "authors": r.authors,
        "categories": r.categories,
        "primary_category": r.primary_category,
        "doi": r.doi,
        "journal_ref": r.journal_ref,
        "update_date": r.update_date,
        "latest_version": r.latest_version,
        "latest_version_created": dt(r.latest_version_created),
    }


class BatchedParquetWriter:
    def __init__(self, path: str | Path, schema: pa.Schema = None):
        self.path = Path(path)
        self.schema = schema or _to_schema()
        self._writer: pq.ParquetWriter | None = None
        self._count = 0

    def __enter__(self) -> "BatchedParquetWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(str(self.path), schema=self.schema)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def count(self) -> int:
        return self._count

    def write_batch(self, rows: list[PaperRow]) -> None:
        if not rows:
            return
        if self._writer is None:
            raise ParquetWriteError(
                "Writer not initialized. Use 'with BatchedParquetWriter(...) as w:'"
            )

        try:
            table = pa.Table.from_pylist([_to_row(r) for r in rows], schema=self.schema)
            self._writer.write_table(table)
            self._count += len(rows)
        except Exception as e:
            logger.exception("Failed to write parquet batch")
            raise ParquetWriteError("Failed to write parquet batch") from e
