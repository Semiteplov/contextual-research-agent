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
You are a scientific research assistant. Multiple specialized agents have produced
partial answers to different aspects of a complex question. Your task is to merge
them into a single, coherent, well-structured response.

Original question: {query}

Partial answers from specialized agents:
{partial_answers}

Requirements:
1. Combine the information without redundancy.
2. Preserve all citations [chunk_id] from the partial answers verbatim.
3. Ensure logical flow between different aspects.
4. If partial answers conflict, note the discrepancy explicitly.
5. Be concise but complete — typical length 6-10 sentences.
6. Do not add information beyond what is in the partial answers.

Provide the merged answer:"""


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
            total_ms = sum(e.get("latency_ms", 0) for e in existing_events)

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
                rationale = sa.get("rationale", "")
                failed = sa.get("failed", False)

                if failed:
                    partial_texts.append(
                        f"--- Aspect {i + 1} ({mode}, FAILED) ---\n"
                        f"Rationale: {rationale}\n"
                        f"[This aspect failed to execute]"
                    )
                else:
                    header = f"--- Aspect {i + 1} ({mode}) ---"
                    if rationale:
                        header += f"\nRationale: {rationale}"
                    partial_texts.append(f"{header}\n{answer}")

            prompt = _SYNTHESIS_PROMPT.format(
                query=state.get("query", ""),
                partial_answers="\n\n".join(partial_texts),
            )

            result = await self._llm.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a precise scientific research assistant. "
                    "Synthesize partial answers from specialized agents into "
                    "a coherent, citation-preserving response."
                ),
                temperature=0.1,
                max_tokens=2048,
            )

            final_answer = result.text

        except Exception as e:
            logger.error("Synthesis failed: %s — concatenating sub-answers", e)
            parts = [sa.get("answer", "") for sa in sub_answers if not sa.get("failed")]
            final_answer = "\n\n---\n\n".join(parts) if parts else "[Synthesis failed]"

        latency_ms = (time.perf_counter() - t_start) * 1000

        succeeded = sum(1 for sa in sub_answers if not sa.get("failed"))
        total_sub_tokens = sum(sa.get("tokens", {}).get("total", 0) for sa in sub_answers)

        trace_event = TraceEvent(
            node="synthesizer",
            status="completed",
            latency_ms=latency_ms,
            data={
                "mode": "multi_answer",
                "num_sub_answers": len(sub_answers),
                "num_succeeded": succeeded,
                "final_length": len(final_answer),
                "sub_query_total_tokens": total_sub_tokens,
            },
        )

        existing_events = list(state.get("trace_events", []))
        existing_events.append(trace_event.to_dict())

        total_ms = sum(e.get("latency_ms", 0) for e in existing_events)

        logger.info(
            "Synthesizer: merged %d sub-answers (%d succeeded), final_len=%d (%.0fms)",
            len(sub_answers),
            succeeded,
            len(final_answer),
            latency_ms,
        )

        return {
            "final_answer": final_answer,
            "generated_answer": final_answer,
            "status": AgentStatus.COMPLETED.value,
            "trace_events": existing_events,
            "total_latency_ms": total_ms,
        }
