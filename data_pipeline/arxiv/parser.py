import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ArxivParseError(RuntimeError):
    pass


def parse_categories(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split() if p.strip()]
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def parse_latest_version(versions: list[dict] | None) -> tuple[int | None, datetime | None]:
    if not versions:
        return None, None
    try:
        last = versions[-1]
        v = str(last.get("version") or "").strip().lower()
        vnum = int(v[1:]) if v.startswith("v") and v[1:].isdigit() else None
        created_raw = last.get("created")
        created_dt = parse_datetime(created_raw) if created_raw else None
        return vnum, created_dt
    except Exception:
        return None, None


def parse_datetime(value: str) -> datetime | None:
    # "Mon, 2 Apr 2007 19:18:42 GMT"
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def parse_update_date(value: str | None):
    # "2008-11-26"
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except Exception:
        return None


@dataclass(frozen=True)
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


def iter_papers_jsonl(path: str | Path, log_every: int = 200_000) -> Iterator[PaperRow]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {p}")

    with p.open("rb") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj: dict = json.loads(line)
            except json.JSONDecodeError as e:
                raise ArxivParseError(f"Invalid JSON at line {line_no}: {e}") from e

            arxiv_id = str(obj.get("id") or "").strip()
            if not arxiv_id:
                continue

            title = (obj.get("title") or "").strip()
            abstract = (obj.get("abstract") or "").strip()
            authors = (obj.get("authors") or "").strip()

            cats = parse_categories(obj.get("categories"))
            primary = cats[0] if cats else None

            latest_v, latest_created = parse_latest_version(obj.get("versions"))
            upd = parse_update_date(obj.get("update_date"))

            row = PaperRow(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=cats,
                primary_category=primary,
                doi=(obj.get("doi") or None),
                journal_ref=(obj.get("journal-ref") or None),
                update_date=upd,
                latest_version=latest_v,
                latest_version_created=latest_created,
            )

            if log_every and line_no % log_every == 0:
                logger.info("Parsed %d lines; last arxiv_id=%s", line_no, arxiv_id)

            yield row


def is_ml_paper(categories: Iterable[str], allowed: set[str]) -> bool:
    return any(c in allowed for c in categories)
