from __future__ import annotations

import time
from typing import Any

from contextual_research_agent.agent.llm import LLMProvider
from contextual_research_agent.agent.state import (
    AgentState,
    AgentStatus,
    CriticVerdict,
    TraceEvent,
)
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.config import CognitiveMode, GenerationConfig
from contextual_research_agent.generation.pipeline import GenerationPipeline

logger = get_logger(__name__)


class GeneratorNode:
    def __init__(self, pipeline: GenerationPipeline):
        self._pipeline = pipeline

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        Reads: query, retrieval_context, resolved_mode, critic_feedback, retry_count
        Writes: generated_answer, generation_latency_ms, generation_tokens,
                system_prompt_used, user_prompt_used, status, trace_events
        """
        t_start = time.perf_counter()

        if state.get("status") == AgentStatus.FAILED.value:
            return {}

        query = state["query"]  # type: ignore
        context = state.get("retrieval_context", "")
        mode_str = state.get("resolved_mode", "factual_qa")
        retry_count = state.get("retry_count", 0)

        effective_query = query
        if retry_count > 0:
            effective_query = (
                f"{query}\n\n"
                f"[Note: Please ensure all claims are directly supported by the context passages above "
                f"and all cited chunk IDs (e.g. [chunk_id]) exist in the provided passages.]"
            )

        try:
            mode = CognitiveMode(mode_str)

            response = await self._pipeline.generate_from_context(
                query=effective_query,
                context=context,
                mode=mode,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000

            trace_event = TraceEvent(
                node="generator",
                status="completed",
                latency_ms=latency_ms,
                data={
                    "mode": mode.value,
                    "retry": retry_count,
                    "answer_length": len(response.answer),
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                },
            )

            logger.info(
                "Generator: mode=%s, retry=%d, answer_len=%d (%.0fms)",
                mode.value,
                retry_count,
                len(response.answer),
                latency_ms,
            )

            existing_events = list(state.get("trace_events", []))
            existing_events.append(trace_event.to_dict())

            return {
                "generated_answer": response.answer,
                "generation_latency_ms": latency_ms,
                "generation_tokens": {
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                    "total": response.prompt_tokens + response.completion_tokens,
                },
                "system_prompt_used": response.system_prompt,
                "user_prompt_used": response.user_prompt,
                "status": AgentStatus.GENERATING.value,
                "trace_events": existing_events,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.error("Generation failed: %s", e)

            trace_event = TraceEvent(
                node="generator",
                status="failed",
                latency_ms=latency_ms,
                error=str(e),
            )

            existing_events = list(state.get("trace_events", []))
            existing_events.append(trace_event.to_dict())

            return {
                "status": AgentStatus.FAILED.value,
                "error": f"Generation failed: {e}",
                "trace_events": existing_events,
            }
