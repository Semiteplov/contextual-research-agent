from __future__ import annotations

import time
from typing import Any

from contextual_research_agent.agent.state import (
    AgentState,
    AgentStatus,
    ChunkSnapshot,
    TraceEvent,
)
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.retrieval.pipeline import RetrievalPipeline

logger = get_logger(__name__)


class RetrieverNode:
    """LangGraph node: execute retrieval pipeline."""

    def __init__(self, pipeline: RetrievalPipeline):
        self._pipeline = pipeline

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        Reads: query, document_ids
        Writes: retrieval_context, retrieval_chunks, retrieval_latency_ms,
                retrieval_channel_stats, status, trace_events
        """
        t_start = time.perf_counter()

        query = state["query"]  # type: ignore
        document_ids = state.get("document_ids")

        try:
            result = await self._pipeline.retrieve(
                query=query,
                document_ids=document_ids,
            )

            # Build chunk snapshots for debug UI
            chunks: list[dict] = []
            for i, candidate in enumerate(result.candidates):
                snapshot = ChunkSnapshot(
                    chunk_id=candidate.chunk_id,
                    text_preview=candidate.chunk.text[:200],
                    section_type=candidate.chunk.metadata.get("section_type", "unknown"),
                    document_id=candidate.document_id,
                    score=candidate.score,
                    rank=i,
                )
                chunks.append(snapshot.to_dict())

            # Channel statistics
            channel_stats = {}
            if hasattr(result, "channel_results"):
                for ch_result in result.channel_results:
                    ch_name = ch_result.channel.value  # ChannelName enum → str
                    channel_stats[ch_name] = {
                        "candidates": ch_result.num_candidates,
                        "latency_ms": getattr(ch_result, "latency_ms", 0),
                    }

            latency_ms = (time.perf_counter() - t_start) * 1000

            trace_event = TraceEvent(
                node="retriever",
                status="completed",
                latency_ms=latency_ms,
                data={
                    "num_candidates": len(result.candidates),
                    "intent": result.intent,
                    "channels_used": list(channel_stats.keys()) if channel_stats else [],
                    "context_length": len(result.context),
                },
            )

            logger.info(
                "Retriever: %d candidates, intent=%s (%.0fms)",
                len(result.candidates),
                result.intent,
                latency_ms,
            )

            existing_events = list(state.get("trace_events", []))
            existing_events.append(trace_event.to_dict())

            return {
                "retrieval_context": result.context,
                "retrieval_chunks": chunks,
                "retrieval_latency_ms": latency_ms,
                "retrieval_channel_stats": channel_stats,
                "intent": result.intent or state.get("intent", ""),
                "status": AgentStatus.RETRIEVING.value,
                "trace_events": existing_events,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.error("Retrieval failed: %s", e)

            trace_event = TraceEvent(
                node="retriever",
                status="failed",
                latency_ms=latency_ms,
                error=str(e),
            )

            existing_events = list(state.get("trace_events", []))
            existing_events.append(trace_event.to_dict())

            return {
                "status": AgentStatus.FAILED.value,
                "error": f"Retrieval failed: {e}",
                "trace_events": existing_events,
            }
