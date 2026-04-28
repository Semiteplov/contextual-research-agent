from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict

from contextual_research_agent.generation.config import CognitiveMode


class QueryComplexity(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTI_ASPECT = "multi_aspect"


class AgentStatus(StrEnum):
    PENDING = "pending"
    ROUTING = "routing"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    CRITIQUING = "critiquing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class CriticVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass
class TraceEvent:
    node: str
    status: str
    latency_ms: float
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "node": self.node,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 1),
        }
        if self.data:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class SubQuery:
    text: str
    mode: CognitiveMode
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode.value,
            "rationale": self.rationale,
        }


@dataclass
class ChunkSnapshot:
    chunk_id: str
    text_preview: str
    section_type: str
    document_id: str
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text_preview": self.text_preview,
            "section_type": self.section_type,
            "document_id": self.document_id,
            "score": round(self.score, 4),
            "rank": self.rank,
        }


@dataclass
class CriticFeedback:
    verdict: CriticVerdict
    reasoning: str
    issues: list[str] = field(default_factory=list)
    faithfulness_score: float | None = None
    completeness_score: float | None = None
    citation_validity_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "verdict": self.verdict.value,
            "reasoning": self.reasoning,
        }
        if self.issues:
            d["issues"] = self.issues
        if self.faithfulness_score is not None:
            d["faithfulness_score"] = self.faithfulness_score
        if self.completeness_score is not None:
            d["completeness_score"] = self.completeness_score
        if self.citation_validity_score is not None:
            d["citation_validity_score"] = self.citation_validity_score
        return d


class AgentState(TypedDict, total=False):
    query: str
    mode_override: str | None
    document_ids: list[str] | None

    intent: str
    complexity: str
    resolved_mode: str

    sub_queries: list[dict]
    current_sub_query_idx: int

    retrieval_context: str
    retrieval_chunks: list[dict]
    retrieval_latency_ms: float
    retrieval_channel_stats: dict[str, Any]

    generated_answer: str
    generation_latency_ms: float
    generation_tokens: dict[str, int]
    system_prompt_used: str
    user_prompt_used: str

    critic_feedback: dict
    retry_count: int

    sub_answers: list[dict]
    final_answer: str

    status: str
    error: str | None
    trace_events: list[dict]
    total_latency_ms: float


def create_initial_state(
    query: str,
    mode_override: str | None = None,
    document_ids: list[str] | None = None,
) -> AgentState:
    return AgentState(
        query=query,
        mode_override=mode_override,
        document_ids=document_ids,
        intent="",
        complexity=QueryComplexity.SIMPLE.value,
        resolved_mode="",
        sub_queries=[],
        current_sub_query_idx=0,
        retrieval_context="",
        retrieval_chunks=[],
        retrieval_latency_ms=0.0,
        retrieval_channel_stats={},
        generated_answer="",
        generation_latency_ms=0.0,
        generation_tokens={},
        system_prompt_used="",
        user_prompt_used="",
        critic_feedback={},
        retry_count=0,
        sub_answers=[],
        final_answer="",
        status=AgentStatus.PENDING.value,
        error=None,
        trace_events=[],
        total_latency_ms=0.0,
    )
