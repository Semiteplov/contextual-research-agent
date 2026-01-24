from typing import TypedDict

from contextual_research_agent.ingestion.domain.entities import Chunk, Document


class IngestionState(TypedDict):
    file_path: str
    run_id: str
    started_at: str
    document: Document | None
    chunks: list[Chunk]
    embeddings: list[list[float]]
    latency: dict[str, float]
    status: str
    error: str | None


class IngestionPatch(TypedDict, total=False):
    document: Document | None
    chunks: list[Chunk]
    embeddings: list[list[float]]
    latency: dict[str, float]
    status: str
    error: str | None
