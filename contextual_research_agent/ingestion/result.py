from dataclasses import dataclass
from typing import Any

from contextual_research_agent.ingestion.domain.entities import Document


@dataclass
class IngestionResult:
    run_id: str
    file_path: str
    document: Document | None
    chunk_count: int
    latency: dict[str, float]
    status: str
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "indexed"

    @property
    def document_id(self) -> str | None:
        return self.document.id if self.document else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "file_path": self.file_path,
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "latency": self.latency,
            "status": self.status,
            "error": self.error,
            "success": self.success,
        }
