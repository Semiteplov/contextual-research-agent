from pathlib import Path

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.arxiv.constants import ML_CATEGORIES
from contextual_research_agent.data.arxiv.parser import PaperRow, is_ml_paper, iter_papers_jsonl
from contextual_research_agent.db import PapersMetadataRepository, get_connection_context

logger = get_logger(__name__)

DEFAULT_METADATA_PATH = ".cache/kaggle/extracted/arxiv-metadata-oai-snapshot.json"
DB_BATCH_SIZE = 5_000


def ingest_arxiv_metadata(
    metadata_path: str = DEFAULT_METADATA_PATH,
    batch_size: int = DB_BATCH_SIZE,
) -> None:
    path = Path(metadata_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    logger.info("Starting arXiv metadata ingestion")
    logger.info("Input: %s", path)
    logger.info("Categories filter: %s", sorted(ML_CATEGORIES))
    logger.info("Batch size: %d", batch_size)

    with get_connection_context(db_name="arxiv") as conn:
        repo = PapersMetadataRepository(conn)
        total, kept = _process_jsonl(repo, path, batch_size)

    logger.info("Ingestion complete: total=%d kept=%d", total, kept)


def _process_jsonl(
    repo: PapersMetadataRepository,
    path: Path,
    batch_size: int,
) -> tuple[int, int]:
    total = 0
    kept = 0
    buffer: list[PaperRow] = []

    for row in iter_papers_jsonl(path):
        total += 1

        if not is_ml_paper(row.categories, ML_CATEGORIES):
            continue

        kept += 1
        buffer.append(row)

        if len(buffer) >= batch_size:
            _flush_batch(repo, buffer)
            buffer.clear()

    if buffer:
        _flush_batch(repo, buffer)

    return total, kept


def _flush_batch(repo: PapersMetadataRepository, buffer: list[PaperRow]) -> None:
    count = repo.upsert(buffer)
    logger.info("Upserted batch: %d rows", count)
