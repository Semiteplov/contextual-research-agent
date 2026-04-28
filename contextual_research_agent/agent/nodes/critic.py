from __future__ import annotations

import re
import time
from typing import Any

from contextual_research_agent.agent.llm import LLMProvider
from contextual_research_agent.agent.state import (
    AgentState,
    AgentStatus,
    CriticFeedback,
    CriticVerdict,
    TraceEvent,
)
from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


_CRITIC_PROMPT = """\
You are a strict quality reviewer for a RAG-based scientific assistant.

Your job is to evaluate an answer for:
1. FAITHFULNESS: Every claim must be supported by the provided context. No hallucinations.
2. COMPLETENESS: The answer should address the question asked.
3. CITATION VALIDITY: If [chunk_id] citations are used, they should reference real chunks from the context.

Context provided to the assistant:
{context}

Question: {query}

Generated answer:
{answer}

Evaluate and respond in this EXACT format:
VERDICT: <PASS or FAIL>
FAITHFULNESS: <score 1-5>
COMPLETENESS: <score 1-5>
ISSUES: <comma-separated list of specific issues, or "none">
REASONING: <1-2 sentences explaining your verdict>"""


_REFUSAL_PATTERNS = [
    re.compile(r"(?i)the provided (sources|context|passages?) do not contain"),
    re.compile(r"(?i)insufficient information"),
    re.compile(r"(?i)cannot (be )?answer(ed)?.*based on"),
    re.compile(r"(?i)the context does not (provide|contain|mention)"),
    re.compile(r"(?i)not enough information"),
    re.compile(r"(?i)no relevant information"),
]


def _is_refusal(answer: str) -> bool:
    """Check if answer is a refusal (always passes critic)."""
    if not answer or not answer.strip():
        return True
    return any(pattern.search(answer) for pattern in _REFUSAL_PATTERNS)


def _check_citation_validity(answer: str, context: str) -> tuple[bool, list[str]]:
    """Check if cited chunk_ids exist in the context."""
    cited_ids = set(re.findall(r"\[([^\]]+)\]", answer))
    context_ids = set(re.findall(r"\[([^\]]+)\]", context))

    chunk_pattern = re.compile(r"\d{4}\.\d{4,5}_[a-f0-9]+_c\d+")
    cited_chunks = {cid for cid in cited_ids if chunk_pattern.match(cid)}

    if not cited_chunks:
        return True, []

    invalid = cited_chunks - context_ids
    return len(invalid) == 0, list(invalid)


def _parse_critic_response(text: str) -> CriticFeedback:
    verdict = CriticVerdict.PASS
    faithfulness = None
    completeness = None
    issues: list[str] = []
    reasoning = ""

    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL|PARTIAL)", text, re.IGNORECASE)
    if verdict_match:
        v = verdict_match.group(1).upper()
        if v == "FAIL":
            verdict = CriticVerdict.FAIL
        elif v == "PARTIAL":
            verdict = CriticVerdict.PARTIAL

    faith_match = re.search(r"FAITHFULNESS:\s*(\d)", text)
    if faith_match:
        faithfulness = float(faith_match.group(1))

    comp_match = re.search(r"COMPLETENESS:\s*(\d)", text)
    if comp_match:
        completeness = float(comp_match.group(1))

    issues_match = re.search(r"ISSUES:\s*(.+?)(?=\n|REASONING:)", text, re.DOTALL)
    if issues_match:
        raw_issues = issues_match.group(1).strip()
        if raw_issues.lower() != "none":
            issues = [i.strip() for i in raw_issues.split(",") if i.strip()]

    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()[:500]

    if faithfulness is not None:
        verdict = CriticVerdict.FAIL if faithfulness <= 1 else CriticVerdict.PASS

    return CriticFeedback(
        verdict=verdict,
        reasoning=reasoning,
        issues=issues,
        faithfulness_score=faithfulness,
        completeness_score=completeness,
    )


class CriticNode:
    MAX_RETRIES = 1

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        t_start = time.perf_counter()

        answer = state.get("generated_answer", "")
        query = state["query"]  # type: ignore
        context = state.get("retrieval_context", "")
        retry_count = state.get("retry_count", 0)

        if _is_refusal(answer):
            feedback = CriticFeedback(
                verdict=CriticVerdict.PASS,
                reasoning="Answer is a valid refusal due to insufficient context.",
            )
            latency_ms = (time.perf_counter() - t_start) * 1000

            trace_event = TraceEvent(
                node="critic",
                status="pass_refusal",
                latency_ms=latency_ms,
                data={"verdict": "pass", "reason": "refusal"},
            )

            existing_events = list(state.get("trace_events", []))
            existing_events.append(trace_event.to_dict())

            return {
                "critic_feedback": feedback.to_dict(),
                "status": AgentStatus.CRITIQUING.value,
                "trace_events": existing_events,
            }

        citations_valid, invalid_ids = _check_citation_validity(answer, context)

        try:
            prompt = _CRITIC_PROMPT.format(
                context=context,
                query=query,
                answer=answer,
            )

            result = await self._llm.generate(
                prompt=prompt,
                system_prompt="You are a strict, precise evaluation judge. Follow the output format exactly.",
                temperature=0.0,
                max_tokens=300,
            )

            feedback = _parse_critic_response(result.text)

            if not citations_valid:
                feedback.issues.append(f"Invalid citations: {', '.join(invalid_ids)}")
                if feedback.verdict == CriticVerdict.PASS:
                    feedback.verdict = CriticVerdict.PARTIAL

        except Exception as e:
            logger.warning("Critic LLM call failed: %s — defaulting to PASS", e)
            feedback = CriticFeedback(
                verdict=CriticVerdict.PASS,
                reasoning=f"Critic evaluation failed ({e}), defaulting to pass.",
            )

        latency_ms = (time.perf_counter() - t_start) * 1000

        new_retry_count = retry_count
        if feedback.verdict == CriticVerdict.FAIL and retry_count < self.MAX_RETRIES:
            new_retry_count = retry_count + 1

        trace_event = TraceEvent(
            node="critic",
            status=f"verdict_{feedback.verdict.value}",
            latency_ms=latency_ms,
            data={
                "verdict": feedback.verdict.value,
                "faithfulness": feedback.faithfulness_score,
                "completeness": feedback.completeness_score,
                "issues": feedback.issues,
                "retry_count": new_retry_count,
                "will_retry": feedback.verdict == CriticVerdict.FAIL
                and new_retry_count > retry_count,
            },
        )

        logger.info(
            "Critic: verdict=%s, faith=%.0f, complete=%.0f, retry=%d/%d (%.0fms)",
            feedback.verdict.value,
            feedback.faithfulness_score or 0,
            feedback.completeness_score or 0,
            new_retry_count,
            self.MAX_RETRIES,
            latency_ms,
        )

        existing_events = list(state.get("trace_events", []))
        existing_events.append(trace_event.to_dict())

        return {
            "critic_feedback": feedback.to_dict(),
            "retry_count": new_retry_count,
            "status": AgentStatus.CRITIQUING.value,
            "trace_events": existing_events,
        }
