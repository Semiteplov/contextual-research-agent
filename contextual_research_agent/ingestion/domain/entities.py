from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from contextual_research_agent.ingestion.domain.types import DocumentStatus


def _generate_id() -> str:
    return uuid4().hex[:12]


@dataclass
class Document:
    id: str = field(default_factory=_generate_id)
    source_path: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""

    status: DocumentStatus = DocumentStatus.PENDING
    content_hash: str = ""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    indexed_at: datetime | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str = field(default_factory=_generate_id)
    document_id: str = ""

    text: str = ""
    token_count: int = 0

    chunk_index: int = 0
    page_numbers: list[int] = field(default_factory=list)
    section: str = ""

    embedding: list[float] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class Citation:
    chunk_id: str
    text_excerpt: str
    relevance: str = ""
