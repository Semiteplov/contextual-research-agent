from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from docling.datamodel.accelerator_options import AcceleratorDevice

from contextual_research_agent.ingestion.domain.types import DocumentStatus


def _generate_id() -> str:
    return uuid4().hex[:12]


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    EQUATION = "equation"


class ParserType(str, Enum):
    DOCLING = "docling"
    GROBID = "grobid"
    NOUGAT = "nougat"
    MANUAL = "manual"


@dataclass
class DocumentConfig:
    """
    Immutable record of parser configuration used to produce this document.
    Stored for reproducibility — allows re-running with identical settings.
    """

    parser: ParserType = ParserType.DOCLING
    embedding_model: str = ""
    max_tokens: int = 512
    include_context: bool = True
    merge_peers: bool = True


@dataclass
class Document:
    """
    Represents a parsed scientific paper.

    Structured fields cover known, queryable attributes.
    `metadata` is reserved for parser-specific or experimental data
    that doesn't warrant a typed field.
    """

    id: str = field(default_factory=_generate_id)
    source_path: str = ""

    # --- Bibliographic ---
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    arxiv_id: str | None = None

    # --- Structure ---
    sections: list[str] = field(default_factory=list)
    num_pages: int = 0

    # --- Storage ---
    content_hash: str = ""
    markdown_s3_key: str | None = None
    markdown_s3_uri: str | None = None

    # --- Pipeline state ---
    status: DocumentStatus = DocumentStatus.PENDING
    parse_config: DocumentConfig = field(default_factory=DocumentConfig)

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    parsed_at: datetime | None = None
    indexed_at: datetime | None = None

    # --- Escape hatch for truly unstructured data ---
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_failed(self) -> bool:
        return self.status == DocumentStatus.FAILED

    @property
    def is_indexed(self) -> bool:
        return self.indexed_at is not None


@dataclass
class Chunk:
    """
    A retrieval unit derived from a Document.

    Each chunk has a typed classification, explicit provenance
    (document_id, section, pages), and quality flags.
    """

    id: str = field(default_factory=_generate_id)
    document_id: str = ""

    # --- Content ---
    text: str = ""
    token_count: int = 0
    chunk_type: ChunkType = ChunkType.TEXT

    # --- Provenance ---
    chunk_index: int = 0
    page_numbers: list[int] = field(default_factory=list)
    section: str = ""  # Heading path, e.g. "3 Method > 3.1 Architecture"

    # --- Embedding ---
    embedding: list[float] | None = None

    # --- Quality flags ---
    is_contextualized: bool = False  # Whether section headings were prepended
    is_fallback: bool = False  # Produced by fallback chunker
    is_degenerate: bool = False  # Below min_chunk_tokens threshold
    is_oversized: bool = False  # Above max_tokens threshold

    # --- Escape hatch ---
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_key(self) -> str:
        return self.metadata.get("source_key", "")


@dataclass
class RetrievedChunk:
    """
    A chunk returned by the retrieval pipeline with relevance scoring.
    """

    chunk: Chunk
    score: float
    rank: int

    # Retrieval method that produced this result (for ablation / logging)
    retrieval_method: str = ""  # e.g. "dense", "sparse", "hybrid"

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class Citation:
    """
    A grounded reference from an agent response to a specific chunk.

    Supports traceability: every claim the agent makes should be
    attributable to one or more Citations.
    """

    chunk_id: str
    document_id: str = ""

    # What was cited
    text_excerpt: str = ""  # The relevant passage from the chunk
    section: str = ""  # Section of the source document
    page_numbers: list[int] = field(default_factory=list)

    # How it was used
    relevance: str = ""  # Why this citation supports the claim
    confidence: float = 0.0  # Agent's confidence in the citation (0-1)

    # Provenance
    retrieval_score: float = 0.0  # Original retrieval score
    retrieval_rank: int = 0  # Original retrieval rank

    @classmethod
    def from_retrieved_chunk(
        cls,
        rc: RetrievedChunk,
        text_excerpt: str = "",
        relevance: str = "",
        confidence: float = 0.0,
    ):
        """Convenience constructor from a RetrievedChunk."""
        return cls(
            chunk_id=rc.chunk.id,
            document_id=rc.chunk.document_id,
            text_excerpt=text_excerpt or rc.chunk.text[:500],
            section=rc.chunk.section,
            page_numbers=rc.chunk.page_numbers,
            relevance=relevance,
            confidence=confidence,
            retrieval_score=rc.score,
            retrieval_rank=rc.rank,
        )


@dataclass(frozen=True)
class DoclingParserConfig:
    max_tokens: int = 512
    embedding_model: str = "Qwen/Qwen3-4B"
    merge_peers: bool = True
    include_context: bool = True
    do_ocr: bool = False
    do_table_structure: bool = True
    num_threads: int = 4
    device: AcceleratorDevice = AcceleratorDevice.AUTO
    min_chunk_tokens: int = 20
    image_placeholder: str = "[Figure]"
    filter_empty_chunks: bool = False
