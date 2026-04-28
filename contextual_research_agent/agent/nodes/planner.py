from __future__ import annotations

import json
import re
import time
from typing import Any

from contextual_research_agent.agent.llm import LLMProvider
from contextual_research_agent.agent.state import (
    AgentState,
    AgentStatus,
    TraceEvent,
)
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.config import CognitiveMode

logger = get_logger(__name__)


_DECOMPOSITION_PROMPT = """\
You are a query planner for a scientific research assistant.

The user asked a complex question that may require multiple retrieval passes.
Decompose it into 2-3 focused sub-queries that together answer the original question.

For each sub-query, specify:
- "text": the focused sub-question
- "mode": one of [factual_qa, summarization, comparison, critical_review, methodological_audit, idea_generation]
- "rationale": why this sub-query is needed

Original query: {query}

Respond ONLY with a JSON array, no other text:
[
  {{"text": "...", "mode": "...", "rationale": "..."}}
]"""


def _parse_sub_queries(llm_output: str) -> list[dict[str, str]]:
    text = llm_output.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict) and "text" in item:
                    mode = item.get("mode", "factual_qa")
                    try:
                        CognitiveMode(mode)
                    except ValueError:
                        mode = "factual_qa"
                    result.append(
                        {
                            "text": str(item["text"]),
                            "mode": mode,
                            "rationale": str(item.get("rationale", "")),
                        }
                    )
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: return original query as single sub-query
    return []


class PlannerNode:
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """
        Reads: query, complexity, sub_queries, resolved_mode
        Writes: sub_queries, status, trace_events
        """
        t_start = time.perf_counter()

        query = state["query"]  # type: ignore
        existing_sub_queries = state.get("sub_queries", [])

        # If Router already populated sub_queries (multi_aspect), refine them
        if existing_sub_queries:
            latency_ms = (time.perf_counter() - t_start) * 1000
            trace_event = TraceEvent(
                node="planner",
                status="skipped_already_planned",
                latency_ms=latency_ms,
                data={"num_sub_queries": len(existing_sub_queries)},
            )
            existing_events = list(state.get("trace_events", []))
            existing_events.append(trace_event.to_dict())
            return {
                "status": AgentStatus.PLANNING.value,
                "trace_events": existing_events,
            }

        # Decompose via LLM
        try:
            prompt = _DECOMPOSITION_PROMPT.format(query=query)
            result = await self._llm.generate(
                prompt=prompt,
                system_prompt="You are a precise query planner. Output only valid JSON.",
                temperature=0.0,
                max_tokens=512,
            )

            sub_queries = _parse_sub_queries(result.text)

            if not sub_queries:
                # Fallback: treat as single query
                resolved_mode = state.get("resolved_mode", "factual_qa")
                sub_queries = [
                    {
                        "text": query,
                        "mode": resolved_mode,
                        "rationale": "Single query (decomposition failed)",
                    }
                ]

            # Cap at 3 sub-queries
            sub_queries = sub_queries[:3]

        except Exception as e:
            logger.error("Planner LLM call failed: %s", e)
            resolved_mode = state.get("resolved_mode", "factual_qa")
            sub_queries = [
                {
                    "text": query,
                    "mode": resolved_mode,
                    "rationale": f"Fallback (planner error: {e})",
                }
            ]

        latency_ms = (time.perf_counter() - t_start) * 1000

        trace_event = TraceEvent(
            node="planner",
            status="completed",
            latency_ms=latency_ms,
            data={
                "num_sub_queries": len(sub_queries),
                "sub_queries": sub_queries,
            },
        )

        logger.info(
            "Planner: decomposed into %d sub-queries (%.0fms)",
            len(sub_queries),
            latency_ms,
        )

        existing_events = list(state.get("trace_events", []))
        existing_events.append(trace_event.to_dict())

        return {
            "sub_queries": sub_queries,
            "current_sub_query_idx": 0,
            "status": AgentStatus.PLANNING.value,
            "trace_events": existing_events,
        }
