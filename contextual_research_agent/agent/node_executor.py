from __future__ import annotations

import asyncio
import time
from typing import Any

from contextual_research_agent.agent.state import (
    AgentState,
    AgentStatus,
    TraceEvent,
)
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.config import CognitiveMode
from contextual_research_agent.generation.pipeline import GenerationPipeline

logger = get_logger(__name__)


class ParallelExecutorNode:
    def __init__(self, generation_pipeline: GenerationPipeline):
        self._pipeline = generation_pipeline

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        t_start = time.perf_counter()

        sub_queries = state.get("sub_queries", [])
        context = state.get("retrieval_context", "")

        if not sub_queries:
            logger.warning("ParallelExecutor invoked without sub_queries")
            return {
                "sub_answers": [],
                "status": AgentStatus.GENERATING.value,
            }

        if not context:
            logger.warning("ParallelExecutor invoked without retrieval_context")
            return {
                "sub_answers": [],
                "status": AgentStatus.FAILED.value,
                "error": "ParallelExecutor: empty context",
            }

        tasks = []
        for sq in sub_queries:
            tasks.append(
                self._run_sub_query(
                    query=sq["text"],
                    context=context,
                    mode_str=sq["mode"],
                    rationale=sq.get("rationale", ""),
                )
            )

        sub_answers_raw = await asyncio.gather(*tasks, return_exceptions=True)

        sub_answers = []
        errors: list[str] = []
        for i, result in enumerate(sub_answers_raw):
            if isinstance(result, Exception):
                errors.append(f"sub_query[{i}]: {result}")
                logger.error("Sub-query %d failed: %s", i, result)
                sub_answers.append(
                    {
                        "query": sub_queries[i]["text"],
                        "mode": sub_queries[i]["mode"],
                        "answer": f"[ERROR: {result}]",
                        "latency_ms": 0.0,
                        "tokens": {},
                        "failed": True,
                    }
                )
            else:
                sub_answers.append(result)

        latency_ms = (time.perf_counter() - t_start) * 1000

        individual_latencies = [a.get("latency_ms", 0) for a in sub_answers if not a.get("failed")]
        serial_estimate = sum(individual_latencies)
        parallel_lower_bound = max(individual_latencies) if individual_latencies else 0
        speedup_factor = serial_estimate / latency_ms if latency_ms > 0 else 1.0

        trace_event = TraceEvent(
            node="parallel_executor",
            status="completed" if not errors else "completed_with_errors",
            latency_ms=latency_ms,
            data={
                "num_sub_queries": len(sub_queries),
                "num_succeeded": len(sub_answers) - len(errors),
                "num_failed": len(errors),
                "wall_clock_ms": round(latency_ms, 1),
                "serial_estimate_ms": round(serial_estimate, 1),
                "parallel_lower_bound_ms": round(parallel_lower_bound, 1),
                "speedup_factor": round(speedup_factor, 2),
                "sub_query_modes": [a.get("mode") for a in sub_answers],
            },
            error="; ".join(errors) if errors else None,
        )

        logger.info(
            "ParallelExecutor: %d sub-queries, wall=%.0fms, serial_estimate=%.0fms, speedup=%.2fx",
            len(sub_queries),
            latency_ms,
            serial_estimate,
            speedup_factor,
        )

        existing_events = list(state.get("trace_events", []))
        existing_events.append(trace_event.to_dict())

        return {
            "sub_answers": sub_answers,
            "status": AgentStatus.GENERATING.value,
            "trace_events": existing_events,
        }

    async def _run_sub_query(
        self,
        query: str,
        context: str,
        mode_str: str,
        rationale: str,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()

        try:
            mode = CognitiveMode(mode_str)
        except ValueError:
            mode = CognitiveMode.FACTUAL_QA
            logger.warning("Invalid mode '%s', falling back to factual_qa", mode_str)

        response = await self._pipeline.generate_from_context(
            query=query,
            context=context,
            mode=mode,
        )

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "query": query,
            "mode": mode.value,
            "rationale": rationale,
            "answer": response.answer,
            "latency_ms": latency_ms,
            "tokens": {
                "prompt": response.prompt_tokens,
                "completion": response.completion_tokens,
                "total": response.prompt_tokens + response.completion_tokens,
            },
            "failed": False,
        }
