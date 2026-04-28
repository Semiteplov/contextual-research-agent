from __future__ import annotations

import time
from typing import Any

from contextual_research_agent.agent.llm import LLMProvider
from contextual_research_agent.agent.state import (
    AgentState,
    AgentStatus,
    TraceEvent,
)
from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


_SYNTHESIS_PROMPT = """\
You are a scientific research assistant. You have received multiple partial answers
to different aspects of a complex question. Synthesize them into a single,
coherent, well-structured response.

Original question: {query}

Partial answers:
{partial_answers}

Requirements:
1. Combine the information without redundancy.
2. Maintain all citations [chunk_id] from the partial answers.
3. Ensure logical flow between different aspects.
4. If partial answers conflict, note the discrepancy.
5. Be concise but complete."""


class SynthesizerNode:
    """LangGraph node: merge sub-answers into final response."""

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        Reads: query, generated_answer, sub_answers
        Writes: final_answer, status, trace_events, total_latency_ms
        """
        t_start = time.perf_counter()

        sub_answers = state.get("sub_answers", [])
        generated_answer = state.get("generated_answer", "")

        # Single answer path: no synthesis needed
        if not sub_answers:
            latency_ms = (time.perf_counter() - t_start) * 1000

            trace_event = TraceEvent(
                node="synthesizer",
                status="passthrough",
                latency_ms=latency_ms,
                data={"mode": "single_answer"},
            )

            existing_events = list(state.get("trace_events", []))
            existing_events.append(trace_event.to_dict())

            # Compute total latency
            total_ms = (
                state.get("retrieval_latency_ms", 0)
                + state.get("generation_latency_ms", 0)
                + latency_ms
            )

            return {
                "final_answer": generated_answer,
                "status": AgentStatus.COMPLETED.value,
                "trace_events": existing_events,
                "total_latency_ms": total_ms,
            }

        # Multi-answer path: synthesize via LLM
        try:
            partial_texts = []
            for i, sa in enumerate(sub_answers):
                mode = sa.get("mode", "unknown")
                answer = sa.get("answer", "")
                partial_texts.append(f"--- Aspect {i + 1} ({mode}) ---\n{answer}")

            prompt = _SYNTHESIS_PROMPT.format(
                query=state["query"],  # type: ignore
                partial_answers="\n\n".join(partial_texts),
            )

            result = await self._llm.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a precise scientific research assistant. "
                    "Synthesize partial answers into a coherent response."
                ),
                temperature=0.1,
                max_tokens=2048,
            )

            final_answer = result.text

        except Exception as e:
            logger.error("Synthesis failed: %s — concatenating sub-answers", e)
            parts = [sa.get("answer", "") for sa in sub_answers]
            final_answer = "\n\n---\n\n".join(parts)

        latency_ms = (time.perf_counter() - t_start) * 1000

        trace_event = TraceEvent(
            node="synthesizer",
            status="completed",
            latency_ms=latency_ms,
            data={
                "mode": "multi_answer",
                "num_sub_answers": len(sub_answers),
                "final_length": len(final_answer),
            },
        )

        existing_events = list(state.get("trace_events", []))
        existing_events.append(trace_event.to_dict())

        total_ms = sum(e.get("latency_ms", 0) for e in existing_events)

        logger.info(
            "Synthesizer: merged %d sub-answers, final_len=%d (%.0fms)",
            len(sub_answers),
            len(final_answer),
            latency_ms,
        )

        return {
            "final_answer": final_answer,
            "status": AgentStatus.COMPLETED.value,
            "trace_events": existing_events,
            "total_latency_ms": total_ms,
        }
