from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CognitiveModeAPI(StrEnum):
    FACTUAL_QA = "factual_qa"
    SUMMARIZATION = "summarization"
    CRITICAL_REVIEW = "critical_review"
    COMPARISON = "comparison"
    METHODOLOGICAL_AUDIT = "methodological_audit"
    IDEA_GENERATION = "idea_generation"


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query text",
    )
    mode: CognitiveModeAPI | None = Field(
        default=None,
        description=("Force cognitive mode. If None, Router Agent auto-detects from query intent."),
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional filter — restrict retrieval to specific documents",
    )
    include_trace: bool = Field(
        default=True,
        description="Include full execution trace in response (for debug)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How does LoRA reduce the number of trainable parameters?",
                "mode": None,
                "document_ids": None,
                "include_trace": True,
            },
        },
    }


class ChunkInfo(BaseModel):
    chunk_id: str
    text_preview: str
    section_type: str
    document_id: str
    score: float
    rank: int


class CriticInfo(BaseModel):
    verdict: str  # pass | fail | partial
    reasoning: str = ""
    issues: list[str] = Field(default_factory=list)
    faithfulness_score: float | None = None
    completeness_score: float | None = None


class TraceEventInfo(BaseModel):
    node: str
    status: str
    latency_ms: float
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class QueryResponse(BaseModel):
    answer: str
    query: str

    intent: str = ""
    complexity: str = ""
    resolved_mode: str = ""

    chunks: list[ChunkInfo] = Field(default_factory=list)
    chunks_count: int = 0

    critic: CriticInfo | None = None
    retry_count: int = 0

    latency_breakdown_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0

    tokens: dict[str, int] = Field(default_factory=dict)

    status: str = "completed"
    error: str | None = None

    events: list[TraceEventInfo] = Field(default_factory=list)


class StreamEventType(StrEnum):
    NODE_START = "node_start"  # Node beginning execution
    NODE_COMPLETE = "node_complete"  # Node finished
    NODE_ERROR = "node_error"  # Node failed
    PARTIAL_ANSWER = "partial"  # Streaming token chunks (future)
    FINAL = "final"  # Final QueryResponse
    ERROR = "error"  # Pipeline-level error
    DONE = "done"  # Stream finished


class StreamEvent(BaseModel):
    type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp_ms: float = 0.0
