from __future__ import annotations

import re
import time
from typing import Any

from contextual_research_agent.agent.state import (
    AgentState,
    AgentStatus,
    QueryComplexity,
    TraceEvent,
)
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.config import CognitiveMode

logger = get_logger(__name__)


_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "comparison",
        [
            r"(?i)\bcompar(e|ing|ison)\b",
            r"(?i)\b(vs\.?|versus)\b",
            r"(?i)\bdifferen(ce|t|ces)\b.*\bbetween\b",
            r"(?i)\b(better|worse|superior|inferior)\b.*\bthan\b",
            r"(?i)\b(advantages?|disadvantages?)\b.*\b(over|compared)\b",
        ],
    ),
    (
        "critique",
        [
            r"(?i)\b(weakness|limitation|drawback|shortcoming|flaw)\b",
            r"(?i)\b(critiqu?e|criticiz|critical analysis)\b",
            r"(?i)\b(problem|issue|concern)\b.*\bwith\b",
        ],
    ),
    (
        "survey",
        [
            r"(?i)\b(survey|review|overview|summarize|summary)\b",
            r"(?i)\b(state.of.the.art|sota|recent (work|advance|progress))\b",
            r"(?i)\b(landscape|taxonomy|categoriz)\b",
        ],
    ),
    (
        "citation_trace",
        [
            r"(?i)\b(which paper|who (first|introduced|proposed))\b",
            r"(?i)\b(cite|citation|reference|cited by)\b",
            r"(?i)\b(build on|extend|follow.up)\b",
        ],
    ),
    (
        "method_explanation",
        [
            r"(?i)\b(explain|describe|overview)\b.*\b(method|approach|technique|algorithm)\b",
            r"(?i)how (does|do|is|are) .+ (implement|work|designed|train)",
            r"(?i)\b(architecture|mechanism|pipeline|framework)\b",
        ],
    ),
    (
        "factual_qa",
        [
            r"(?i)^(what|how much|how many|which|when|where|who)\b",
            r"(?i)what (is|are|was|were|does|did|do)\b",
            r"(?i)how does .+ (work|achieve|perform)",
            r"(?i)what (accuracy|score|result|performance|rank)",
        ],
    ),
]

_MULTI_ASPECT_PATTERNS = [
    r"(?i)\b(and also|and then|additionally)\b",
    r"(?i)\bcompare\b.+\b(critiqu?e|weakness|limitation)\b",
    r"(?i)\b(explain|describe)\b.+\b(and|,)\s*(compare|critiqu?e|evaluat)\b",
    r"(?i)\b(summarize|review)\b.+\b(and|,)\s*(critiqu?e|compare|suggest)\b",
]

_COMPLEX_PATTERNS = [
    r"(?i)\b(across|between|among)\b.*\b(papers?|methods?|approaches?)\b",
    r"(?i)\b(all|every|each)\b.*\b(paper|method|approach)\b",
    r"(?i)\b(evolution|history|timeline|progression)\b",
    r"(?i)\b(comprehensive|thorough|detailed|in.depth)\b.*\b(analysis|review|comparison)\b",
]


def _detect_intent(query: str) -> str:
    """Detect primary intent from query text."""
    for intent, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, query):
                return intent
    return "factual_qa"


def _detect_all_intents(query: str) -> list[str]:
    """Detect ALL matching intents (for multi-aspect detection)."""
    matched = []
    for intent, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, query):
                matched.append(intent)
                break
    return matched or ["factual_qa"]


def _assess_complexity(query: str, intents: list[str]) -> QueryComplexity:
    """Assess query complexity based on intent count and patterns."""
    # Multi-aspect: multiple distinct intents
    if len(intents) >= 2:
        return QueryComplexity.MULTI_ASPECT

    # Explicit multi-aspect patterns
    for pattern in _MULTI_ASPECT_PATTERNS:
        if re.search(pattern, query):
            return QueryComplexity.MULTI_ASPECT

    # Complex: cross-document or deep reasoning
    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, query):
            return QueryComplexity.COMPLEX

    return QueryComplexity.SIMPLE


def _resolve_mode(
    intent: str,
    mode_override: str | None,
    query: str,
) -> CognitiveMode:
    """Resolve cognitive mode from intent or override."""
    if mode_override:
        try:
            return CognitiveMode(mode_override)
        except ValueError:
            logger.warning("Invalid mode override '%s', falling back to intent", mode_override)

    mode = CognitiveMode.from_intent(intent)

    # Refinement: factual patterns inside comparison queries → factual_qa
    if mode == CognitiveMode.COMPARISON:
        factual_overrides = [
            r"(?i)^(how much|how many|what is the|what are the|what accuracy)",
            r"(?i)^(what score|what result|what performance)",
        ]
        for pattern in factual_overrides:
            if re.match(pattern, query.strip()):
                return CognitiveMode.FACTUAL_QA

    return mode


async def router_node(state: AgentState) -> dict[str, Any]:
    t_start = time.perf_counter()

    query = state["query"]  # type: ignore
    mode_override = state.get("mode_override")

    # Detect intents
    all_intents = _detect_all_intents(query)
    primary_intent = all_intents[0]

    # Assess complexity
    complexity = _assess_complexity(query, all_intents)

    # Resolve mode
    resolved_mode = _resolve_mode(primary_intent, mode_override, query)

    # For multi-aspect: pre-build sub-queries from detected intents
    sub_queries = []
    if complexity == QueryComplexity.MULTI_ASPECT and not mode_override:
        seen_modes = set()
        for intent in all_intents:
            mode = CognitiveMode.from_intent(intent)
            if mode.value not in seen_modes:
                seen_modes.add(mode.value)
                sub_queries.append(
                    {
                        "text": query,
                        "mode": mode.value,
                        "rationale": f"Detected intent: {intent}",
                    }
                )

    latency_ms = (time.perf_counter() - t_start) * 1000

    trace_event = TraceEvent(
        node="router",
        status="completed",
        latency_ms=latency_ms,
        data={
            "all_intents": all_intents,
            "primary_intent": primary_intent,
            "complexity": complexity.value,
            "resolved_mode": resolved_mode.value,
            "num_sub_queries": len(sub_queries),
        },
    )

    logger.info(
        "Router: intent=%s, complexity=%s, mode=%s, sub_queries=%d",
        primary_intent,
        complexity.value,
        resolved_mode.value,
        len(sub_queries),
    )

    existing_events = list(state.get("trace_events", []))
    existing_events.append(trace_event.to_dict())

    return {
        "intent": primary_intent,
        "complexity": complexity.value,
        "resolved_mode": resolved_mode.value,
        "sub_queries": sub_queries,
        "status": AgentStatus.ROUTING.value,
        "trace_events": existing_events,
    }
