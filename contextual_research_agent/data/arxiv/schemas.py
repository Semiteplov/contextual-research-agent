from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ParsedEntry:
    """Parsed entry from arXiv Atom feed."""

    arxiv_id: str
    title: str
    abstract: str
    authors: str
    categories: list[str]
    primary_category: str | None
    doi: str | None
    journal_ref: str | None
    published: datetime
    updated: datetime
    latest_version: int | None
