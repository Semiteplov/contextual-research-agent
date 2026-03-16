from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


class QueryIntent(str, Enum):
    """
    Query intent categories mapped to agent cognitive modes.
    """

    FACTUAL_QA = "factual_qa"
    METHOD_EXPLANATION = "method_explanation"
    COMPARISON = "comparison"
    CRITIQUE = "critique"
    SURVEY = "survey"
    CITATION_TRACE = "citation_trace"
    GENERAL = "general"


@dataclass
class QueryPlan:
    intent: QueryIntent
    original_query: str

    section_types: list[str] | None = None

    channel_weight_overrides: dict[str, float] = field(default_factory=dict)

    expanded_queries: list[str] = field(default_factory=list)

    document_ids: list[str] | None = None

    analysis_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "original_query": self.original_query,
            "section_types": self.section_types,
            "channel_weight_overrides": self.channel_weight_overrides,
            "expanded_queries": self.expanded_queries,
            "analysis_ms": round(self.analysis_ms, 2),
        }


_INTENT_SECTION_MAP: dict[QueryIntent, list[str]] = {
    QueryIntent.METHOD_EXPLANATION: ["method", "background", "introduction"],
    QueryIntent.COMPARISON: ["results", "experiments", "related_work"],
    QueryIntent.CRITIQUE: ["method", "experiments", "limitations", "discussion"],
    QueryIntent.SURVEY: ["related_work", "introduction", "background"],
    QueryIntent.CITATION_TRACE: [],
    QueryIntent.FACTUAL_QA: [],
    QueryIntent.GENERAL: [],
}

_INTENT_WEIGHT_OVERRIDES: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.CITATION_TRACE: {
        "graph_citation": 1.5,
        "graph_entity": 0.5,
    },
    QueryIntent.SURVEY: {
        "graph_citation": 1.2,
        "paper_level": 1.2,
    },
    QueryIntent.COMPARISON: {
        "graph_entity": 1.3,
    },
}

_INTENT_PATTERNS: list[tuple[re.Pattern, QueryIntent]] = [
    # Citation trace
    (
        re.compile(r"(cit(e|es|ed|ation)|reference[sd]?|who\s+(cit|refer))", re.IGNORECASE),
        QueryIntent.CITATION_TRACE,
    ),
    (
        re.compile(r"(based\s+on|build[s]?\s+on|extend[s]?|follow[s]?\s+up)", re.IGNORECASE),
        QueryIntent.CITATION_TRACE,
    ),
    # Comparison
    (
        re.compile(
            r"(compar|vs\.?|versus|differ|better|worse|outperform|benchmark)", re.IGNORECASE
        ),
        QueryIntent.COMPARISON,
    ),
    (re.compile(r"(SOTA|state.of.the.art|baseline[s]?)", re.IGNORECASE), QueryIntent.COMPARISON),
    # Critique
    (
        re.compile(r"(weak|flaw|limit|shortcoming|problem|issue|bias|assumption)", re.IGNORECASE),
        QueryIntent.CRITIQUE,
    ),
    (re.compile(r"(critic|review|evaluat|assess|valid)", re.IGNORECASE), QueryIntent.CRITIQUE),
    # Method explanation
    (
        re.compile(
            r"(how\s+(does|do|is|are|works?)|explain|descri|method|approach|algorithm|architecture)",
            re.IGNORECASE,
        ),
        QueryIntent.METHOD_EXPLANATION,
    ),
    (
        re.compile(r"(propos|present|introduc|design)", re.IGNORECASE),
        QueryIntent.METHOD_EXPLANATION,
    ),
    # Survey
    (
        re.compile(r"(survey|overview|summar|review\s+of|landscape|taxonomy)", re.IGNORECASE),
        QueryIntent.SURVEY,
    ),
    (
        re.compile(r"(what\s+(are|is)\s+the\s+(main|key|recent|latest))", re.IGNORECASE),
        QueryIntent.SURVEY,
    ),
]


class QueryAnalyzer:
    """
    Analyze query to produce a QueryPlan for retrieval routing.

    Two modes:
      - "rule": keyword/regex pattern matching.
      - "llm": LLM-based intent classification.
    """

    def __init__(self, method: str = "rule"):
        self._method = method

    def analyze(
        self,
        query: str,
        document_ids: list[str] | None = None,
    ) -> QueryPlan:
        """
        Produce a QueryPlan from the input query.

        Args:
            query: Raw query text.
            document_ids: Optional document scope constraint.

        Returns:
            QueryPlan with intent, section filters, and weight overrides.
        """
        t0 = time.perf_counter()

        intent = self._detect_intent_rule(query) if self._method == "rule" else QueryIntent.GENERAL

        section_types = _INTENT_SECTION_MAP.get(intent)
        section_types = section_types if section_types else None

        weight_overrides = _INTENT_WEIGHT_OVERRIDES.get(intent, {})

        analysis_ms = (time.perf_counter() - t0) * 1000

        plan = QueryPlan(
            intent=intent,
            original_query=query,
            section_types=section_types,
            channel_weight_overrides=weight_overrides,
            document_ids=document_ids,
            analysis_ms=analysis_ms,
        )

        logger.debug(
            "Query analysis: intent=%s, sections=%s (%.1fms)",
            intent.value,
            section_types,
            analysis_ms,
        )

        return plan

    @staticmethod
    def _detect_intent_rule(query: str) -> QueryIntent:
        for pattern, intent in _INTENT_PATTERNS:
            if pattern.search(query):
                return intent
        return QueryIntent.GENERAL
