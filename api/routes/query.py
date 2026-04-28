from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, AsyncIterator

from api.auth import verify_api_key
from api.lifespan import get_service
from api.schemas import (
    ChunkInfo,
    CriticInfo,
    QueryRequest,
    QueryResponse,
    StreamEvent,
    StreamEventType,
    TraceEventInfo,
)
from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

if TYPE_CHECKING:
    from contextual_research_agent.agent.service import ResearchAssistantService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


def _trace_to_response(trace, query: str, include_trace: bool) -> QueryResponse:
    """Convert AgentTrace → QueryResponse."""
    chunks = [
        ChunkInfo(
            chunk_id=c.get("chunk_id", ""),
            text_preview=c.get("text_preview", ""),
            section_type=c.get("section_type", ""),
            document_id=c.get("document_id", ""),
            score=c.get("score", 0.0),
            rank=c.get("rank", 0),
        )
        for c in trace.retrieved_chunks
    ]

    critic = None
    if trace.critic_feedback:
        critic = CriticInfo(
            verdict=trace.critic_feedback.get("verdict", ""),
            reasoning=trace.critic_feedback.get("reasoning", ""),
            issues=trace.critic_feedback.get("issues", []),
            faithfulness_score=trace.critic_feedback.get("faithfulness_score"),
            completeness_score=trace.critic_feedback.get("completeness_score"),
        )

    events: list[TraceEventInfo] = []
    if include_trace:
        events = [
            TraceEventInfo(
                node=e.get("node", ""),
                status=e.get("status", ""),
                latency_ms=e.get("latency_ms", 0.0),
                data=e.get("data", {}),
                error=e.get("error"),
            )
            for e in trace.events
        ]

    answer = trace.final_answer or trace.generated_answer or ""
    if trace.error:
        answer = f"Error: {trace.error}"

    return QueryResponse(
        answer=answer,
        query=query,
        intent=trace.detected_intent,
        complexity=trace.complexity,
        resolved_mode=trace.resolved_mode,
        chunks=chunks,
        chunks_count=len(chunks),
        critic=critic,
        retry_count=trace.retry_count,
        latency_breakdown_ms=trace.get_latency_breakdown(),
        total_latency_ms=trace.total_latency_ms,
        tokens=trace.generation_tokens,
        status=trace.status or "completed",
        error=trace.error,
        events=events,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute a query through the multi-agent system",
    description=(
        "Synchronous query execution. Returns the full response with answer, "
        "retrieved chunks, critic feedback, and execution trace. "
        "Latency: typically 100-200 seconds for Qwen3:8b on Ollama."
    ),
)
async def query_endpoint(
    request: QueryRequest,
    _api_key: str | None = Depends(verify_api_key),
) -> QueryResponse:
    service = get_service()

    try:
        result = await service.query(
            text=request.query,
            mode=request.mode.value if request.mode else None,
            document_ids=request.document_ids,
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {e}",
        ) from e

    return _trace_to_response(
        result.trace,
        query=request.query,
        include_trace=request.include_trace,
    )


async def _stream_query_events(
    service: "ResearchAssistantService",
    request: QueryRequest,
    query_id: str,
) -> AsyncIterator[dict]:
    t_start = time.perf_counter()

    yield _format_sse_event(
        StreamEvent(
            type=StreamEventType.NODE_START,
            data={"query_id": query_id, "query": request.query},
            timestamp_ms=0.0,
        )
    )

    try:
        result = await service.query(
            text=request.query,
            mode=request.mode.value if request.mode else None,
            document_ids=request.document_ids,
        )

        cumulative_ms = 0.0
        for event in result.trace.events:
            cumulative_ms += event.get("latency_ms", 0.0)

            yield _format_sse_event(
                StreamEvent(
                    type=(
                        StreamEventType.NODE_ERROR
                        if event.get("error")
                        else StreamEventType.NODE_COMPLETE
                    ),
                    data={
                        "query_id": query_id,
                        "node": event.get("node", ""),
                        "status": event.get("status", ""),
                        "latency_ms": event.get("latency_ms", 0.0),
                        "details": event.get("data", {}),
                        "error": event.get("error"),
                    },
                    timestamp_ms=cumulative_ms,
                )
            )
            await asyncio.sleep(0.01)

        final_response = _trace_to_response(
            result.trace,
            query=request.query,
            include_trace=request.include_trace,
        )

        yield _format_sse_event(
            StreamEvent(
                type=StreamEventType.FINAL,
                data={"query_id": query_id, "response": final_response.model_dump()},
                timestamp_ms=(time.perf_counter() - t_start) * 1000,
            )
        )

    except Exception as e:
        logger.exception("Streaming query failed")
        yield _format_sse_event(
            StreamEvent(
                type=StreamEventType.ERROR,
                data={"query_id": query_id, "error": str(e)},
                timestamp_ms=(time.perf_counter() - t_start) * 1000,
            )
        )

    yield _format_sse_event(
        StreamEvent(
            type=StreamEventType.DONE,
            data={"query_id": query_id},
            timestamp_ms=(time.perf_counter() - t_start) * 1000,
        )
    )


def _format_sse_event(event: StreamEvent) -> dict:
    return {
        "event": event.type.value,
        "data": json.dumps(event.model_dump(mode="json"), ensure_ascii=False),
    }


@router.post(
    "/query/stream",
    summary="Execute a query with Server-Sent Events streaming",
    description=(
        "SSE endpoint that streams trace events as they occur. "
        "Useful for UI clients that want to show real-time progress. "
        "Final event contains the complete QueryResponse."
    ),
)
async def query_stream_endpoint(
    request: QueryRequest,
    _api_key: str | None = Depends(verify_api_key),
) -> EventSourceResponse:
    service = get_service()
    query_id = str(uuid.uuid4())

    return EventSourceResponse(
        _stream_query_events(service, request, query_id),
        media_type="text/event-stream",
    )
