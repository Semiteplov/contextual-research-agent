from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTrace:
    """Complete execution trace for one query."""

    # Input
    query: str
    mode_override: str | None = None

    # Router
    detected_intent: str = ""
    complexity: str = ""
    resolved_mode: str = ""

    # Planner
    sub_queries: list[dict] = field(default_factory=list)

    # Retrieval
    retrieved_chunks: list[dict] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    channel_stats: dict[str, Any] = field(default_factory=dict)

    # Generation
    generated_answer: str = ""
    generation_latency_ms: float = 0.0
    generation_tokens: dict[str, int] = field(default_factory=dict)
    system_prompt: str = ""
    user_prompt: str = ""

    # Critic
    critic_verdict: str = ""
    critic_feedback: dict = field(default_factory=dict)
    retry_count: int = 0

    # Final
    final_answer: str = ""
    total_latency_ms: float = 0.0
    status: str = ""
    error: str | None = None

    # Event log
    events: list[dict] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> AgentTrace:
        """Extract trace from completed AgentState."""
        critic_fb = state.get("critic_feedback", {})

        return cls(
            query=state.get("query", ""),
            mode_override=state.get("mode_override"),
            detected_intent=state.get("intent", ""),
            complexity=state.get("complexity", ""),
            resolved_mode=state.get("resolved_mode", ""),
            sub_queries=state.get("sub_queries", []),
            retrieved_chunks=state.get("retrieval_chunks", []),
            retrieval_latency_ms=state.get("retrieval_latency_ms", 0.0),
            channel_stats=state.get("retrieval_channel_stats", {}),
            generated_answer=state.get("generated_answer", ""),
            generation_latency_ms=state.get("generation_latency_ms", 0.0),
            generation_tokens=state.get("generation_tokens", {}),
            system_prompt=state.get("system_prompt_used", ""),
            user_prompt=state.get("user_prompt_used", ""),
            critic_verdict=critic_fb.get("verdict", ""),
            critic_feedback=critic_fb,
            retry_count=state.get("retry_count", 0),
            final_answer=state.get("final_answer", ""),
            total_latency_ms=state.get("total_latency_ms", 0.0),
            status=state.get("status", ""),
            error=state.get("error"),
            events=state.get("trace_events", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Full trace as dict (for JSON serialization / MLflow)."""
        return {
            "query": self.query,
            "mode_override": self.mode_override,
            "routing": {
                "intent": self.detected_intent,
                "complexity": self.complexity,
                "resolved_mode": self.resolved_mode,
                "sub_queries": self.sub_queries,
            },
            "retrieval": {
                "num_chunks": len(self.retrieved_chunks),
                "latency_ms": round(self.retrieval_latency_ms, 1),
                "channel_stats": self.channel_stats,
                "chunks": self.retrieved_chunks,
            },
            "generation": {
                "answer_length": len(self.generated_answer),
                "latency_ms": round(self.generation_latency_ms, 1),
                "tokens": self.generation_tokens,
                "mode": self.resolved_mode,
            },
            "critic": {
                "verdict": self.critic_verdict,
                "feedback": self.critic_feedback,
                "retry_count": self.retry_count,
            },
            "final_answer": self.final_answer,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "status": self.status,
            "error": self.error,
            "events": self.events,
        }

    def to_debug_summary(self) -> dict[str, Any]:
        """Compact summary for Gradio debug panel."""
        return {
            "intent": self.detected_intent,
            "complexity": self.complexity,
            "mode": self.resolved_mode,
            "chunks_retrieved": len(self.retrieved_chunks),
            "retrieval_ms": round(self.retrieval_latency_ms),
            "generation_ms": round(self.generation_latency_ms),
            "total_ms": round(self.total_latency_ms),
            "critic_verdict": self.critic_verdict,
            "retry_count": self.retry_count,
            "tokens": self.generation_tokens,
            "status": self.status,
        }

    def get_chunks_for_display(self) -> list[dict[str, Any]]:
        """Chunks formatted for Gradio Dataframe."""
        return [
            {
                "rank": c.get("rank", i),
                "chunk_id": c.get("chunk_id", ""),
                "section": c.get("section_type", ""),
                "document": c.get("document_id", "")[:20],
                "score": round(c.get("score", 0), 4),
                "preview": c.get("text_preview", "")[:100],
            }
            for i, c in enumerate(self.retrieved_chunks)
        ]

    def get_latency_breakdown(self) -> dict[str, float]:
        """Per-node latency breakdown."""
        breakdown = {}
        for event in self.events:
            node = event.get("node", "unknown")
            ms = event.get("latency_ms", 0)

            if node in breakdown:
                breakdown[node] += ms
            else:
                breakdown[node] = ms
        return breakdown
