import json
from collections.abc import Iterable, Iterator, Set
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


class ArxivParseError(RuntimeError):
    """Error parsing arXiv metadata."""


@dataclass(frozen=True, slots=True)
class PaperRow:
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    categories: list[str]
    primary_category: str | None
    doi: str | None
    journal_ref: str | None
    update_date: date | None
    latest_version: int | None
    latest_version_created: datetime | None


def iter_papers_jsonl(
    path: str | Path,
    log_every: int = 200_000,
) -> Iterator[PaperRow]:
    """
    Stream-parse arXiv JSONL metadata file.

    Args:
        path: Path to JSONL file.
        log_every: Log progress every N lines (0 to disable).

    Yields:
        PaperRow for each valid record.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ArxivParseError: If JSON parsing fails.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    with file_path.open("rb") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj: dict = json.loads(line)
            except json.JSONDecodeError as e:
                raise ArxivParseError(f"Invalid JSON at line {line_no}: {e}") from e

            row = _parse_record(obj)
            if row is None:
                continue

            if log_every and line_no % log_every == 0:
                logger.info("Parsed %d lines; last arxiv_id=%s", line_no, row.arxiv_id)

            yield row


def is_ml_paper(categories: Iterable[str], allowed: Set[str]) -> bool:
    return any(cat in allowed for cat in categories)


def _parse_record(obj: dict) -> PaperRow | None:
    arxiv_id = str(obj.get("id") or "").strip()
    if not arxiv_id:
        return None

    categories = _parse_categories(obj.get("categories"))
    latest_version, latest_created = _parse_latest_version(obj.get("versions"))

    return PaperRow(
        arxiv_id=arxiv_id,
        title=(obj.get("title") or "").strip(),
        abstract=(obj.get("abstract") or "").strip(),
        authors=(obj.get("authors") or "").strip(),
        categories=categories,
        primary_category=categories[0] if categories else None,
        doi=obj.get("doi") or None,
        journal_ref=obj.get("journal-ref") or None,
        update_date=_parse_date(obj.get("update_date")),
        latest_version=latest_version,
        latest_version_created=latest_created,
    )


def _parse_categories(value: str | None) -> list[str]:
    if not value:
        return []

    seen: set[str] = set()
    result: list[str] = []

    for row_part in value.split():
        part = row_part.strip()
        if part and part not in seen:
            seen.add(part)
            result.append(part)

    return result


def _parse_latest_version(versions: list[dict] | None) -> tuple[int | None, datetime | None]:
    if not versions:
        return None, None

    try:
        last = versions[-1]
        version_str = str(last.get("version") or "").strip().lower()

        version_num = None
        if version_str.startswith("v") and version_str[1:].isdigit():
            version_num = int(version_str[1:])

        created_raw = last.get("created")
        created_dt = _parse_datetime(created_raw) if created_raw else None

        return version_num, created_dt
    except Exception:
        return None, None


def _parse_datetime(value: str) -> datetime | None:
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except Exception:
        return None
