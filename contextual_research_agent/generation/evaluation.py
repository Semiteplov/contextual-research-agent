from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contextual_research_agent.agent.llm import LLMProvider
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.pipeline import RAGResponse
from contextual_research_agent.ingestion.embeddings.base import Embedder

logger = get_logger(__name__)


@dataclass
class GenerationMetrics:
    query: str = ""
    category: str = ""

    semantic_similarity: float | None = None

    # LLM-as-judge
    faithfulness_score: float | None = None
    faithfulness_reasoning: str = ""

    relevance_score: float | None = None
    relevance_reasoning: str = ""

    # Binary flags
    is_refusal: bool = False
    expected_refusal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "category": self.category,
            "semantic_similarity": self.semantic_similarity,
            "faithfulness_score": self.faithfulness_score,
            "faithfulness_reasoning": self.faithfulness_reasoning,
            "relevance_score": self.relevance_score,
            "relevance_reasoning": self.relevance_reasoning,
            "is_refusal": self.is_refusal,
            "expected_refusal": self.expected_refusal,
        }


@dataclass
class AggregatedGenerationMetrics:
    num_queries: int = 0

    # Semantic similarity
    mean_semantic_similarity: float = 0.0
    median_semantic_similarity: float = 0.0

    # Faithfulness
    mean_faithfulness: float = 0.0
    faithfulness_pass_rate: float = 0.0

    # Relevance
    mean_relevance: float = 0.0
    relevance_pass_rate: float = 0.0

    # Refusal
    refusal_rate: float = 0.0
    refusal_accuracy: float = 0.0

    # Per-category breakdown
    category_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "mean_semantic_similarity": round(self.mean_semantic_similarity, 4),
            "median_semantic_similarity": round(self.median_semantic_similarity, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "faithfulness_pass_rate": round(self.faithfulness_pass_rate, 4),
            "mean_relevance": round(self.mean_relevance, 4),
            "relevance_pass_rate": round(self.relevance_pass_rate, 4),
            "refusal_rate": round(self.refusal_rate, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "category_metrics": self.category_metrics,
        }


# Refusal detection patterns
_REFUSAL_PATTERNS = [
    r"(?i)the provided (sources|context|passages?) (do|does) not contain",
    r"(?i)insufficient information",
    r"(?i)cannot (answer|determine|find)",
    r"(?i)no (relevant|sufficient) (information|context|evidence)",
    r"(?i)not enough (information|context|evidence) to answer",
    r"(?i)I (cannot|can't|don't have enough)",
]


class GenerationEvaluator:
    def __init__(
        self,
        embedder: Embedder,
        judge_llm: LLMProvider | None = None,
    ):
        self._embedder = embedder
        self._judge = judge_llm

    async def evaluate_single(
        self,
        response: RAGResponse,
        expected_answer: str | None = None,
        context: str | None = None,
        category: str = "",
    ) -> GenerationMetrics:
        metrics = GenerationMetrics(
            query=response.query,
            category=category,
        )

        metrics.is_refusal = self._detect_refusal(response.answer)

        answer_text = response.answer.strip()
        if not answer_text:
            logger.warning("Empty answer for query: %s", response.query[:80])
            metrics.is_refusal = True
            metrics.faithfulness_reasoning = "Empty answer from LLM"
            return metrics

        if expected_answer and self._embedder and not metrics.is_refusal:
            metrics.semantic_similarity = await self._compute_semantic_similarity(
                response.answer, expected_answer
            )

        ctx = context or response.user_prompt
        if self._judge and not metrics.is_refusal:
            faith = await self._evaluate_faithfulness(
                answer=response.answer,
                context=ctx,
                query=response.query,
            )
            metrics.faithfulness_score = faith["score"]
            metrics.faithfulness_reasoning = faith["reasoning"]

        if self._judge and not metrics.is_refusal:
            rel = await self._evaluate_relevance(
                answer=response.answer,
                query=response.query,
            )
            metrics.relevance_score = rel["score"]
            metrics.relevance_reasoning = rel["reasoning"]

        return metrics

    def aggregate(
        self,
        per_query: list[GenerationMetrics],
    ) -> AggregatedGenerationMetrics:
        if not per_query:
            return AggregatedGenerationMetrics()

        n = len(per_query)

        # Semantic similarity
        sim_scores = [m.semantic_similarity for m in per_query if m.semantic_similarity is not None]
        mean_sim = float(np.mean(sim_scores)) if sim_scores else 0.0
        median_sim = float(np.median(sim_scores)) if sim_scores else 0.0

        # Faithfulness
        faith_scores = [m.faithfulness_score for m in per_query if m.faithfulness_score is not None]
        mean_faith = float(np.mean(faith_scores)) if faith_scores else 0.0
        faith_pass = (
            sum(1 for s in faith_scores if s >= 4) / len(faith_scores) if faith_scores else 0.0
        )

        # Relevance
        rel_scores = [m.relevance_score for m in per_query if m.relevance_score is not None]
        mean_rel = float(np.mean(rel_scores)) if rel_scores else 0.0
        rel_pass = sum(1 for s in rel_scores if s >= 4) / len(rel_scores) if rel_scores else 0.0

        # Refusal
        refusals = [m for m in per_query if m.is_refusal]
        refusal_rate = len(refusals) / n if n > 0 else 0.0
        correct_refusals = sum(1 for m in refusals if m.expected_refusal)
        refusal_accuracy = correct_refusals / len(refusals) if refusals else 0.0

        # Per-category breakdown
        category_metrics = self._compute_category_breakdown(per_query)

        return AggregatedGenerationMetrics(
            num_queries=n,
            mean_semantic_similarity=mean_sim,
            median_semantic_similarity=median_sim,
            mean_faithfulness=mean_faith,
            faithfulness_pass_rate=faith_pass,
            mean_relevance=mean_rel,
            relevance_pass_rate=rel_pass,
            refusal_rate=refusal_rate,
            refusal_accuracy=refusal_accuracy,
            category_metrics=category_metrics,
        )

    @staticmethod
    def _detect_refusal(answer: str) -> bool:
        return any(re.search(pattern, answer) for pattern in _REFUSAL_PATTERNS)

    async def _compute_semantic_similarity(self, answer: str, expected: str) -> float:
        emb_answer = await self._embedder.embed_query(answer)
        emb_expected = await self._embedder.embed_query(expected)

        a = np.array(emb_answer)
        b = np.array(emb_expected)

        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        return max(0.0, min(1.0, cos_sim))

    async def _evaluate_faithfulness(self, answer: str, context: str, query: str) -> dict[str, Any]:
        prompt = f"""\
            You are evaluating a RAG system's answer for FAITHFULNESS.

            Faithfulness means: every factual claim in the answer is supported by the provided context.
            A claim not found in the context is a hallucination.

            Context:
            {context[:6000]}

            Question: {query}

            Answer to evaluate:
            {answer}

            Rate faithfulness on a scale of 1-5:
            1 = Multiple hallucinated claims, answer contradicts context
            2 = Some claims not supported by context
            3 = Mostly faithful, minor unsupported claims
            4 = Faithful, all major claims supported, minor imprecisions
            5 = Perfectly faithful, every claim directly supported by context

            Respond in this exact format:
            SCORE: <number 1-5>
            REASONING: <1-2 sentences explaining your rating>
        """

        return await self._judge_call(prompt)

    async def _evaluate_relevance(self, answer: str, query: str) -> dict[str, Any]:
        prompt = f"""\
            You are evaluating a RAG system's answer for RELEVANCE.

            Relevance means: the answer directly addresses the question asked.

            Question: {query}

            Answer to evaluate:
            {answer}

            Rate relevance on a scale of 1-5:
            1 = Completely irrelevant, does not address the question
            2 = Partially relevant, addresses a related but different question
            3 = Relevant but incomplete or tangential
            4 = Relevant and mostly complete
            5 = Directly and completely addresses the question

            Respond in this exact format:
            SCORE: <number 1-5>
            REASONING: <1-2 sentences explaining your rating>
        """

        return await self._judge_call(prompt)

    async def _judge_call(self, prompt: str) -> dict[str, Any]:
        if self._judge is None:
            return {"score": None, "reasoning": "Judge is not initiated"}

        try:
            result = await self._judge.generate(
                prompt=prompt,
                system_prompt="You are a precise evaluation judge. Follow the output format exactly.",  # noqa: E501
                temperature=0.0,
                max_tokens=256,
            )
            return self._parse_judge_response(result.text)
        except Exception as e:
            logger.warning("Judge LLM call failed: %s", e)
            return {"score": None, "reasoning": f"Judge error: {e}"}

    @staticmethod
    def _parse_judge_response(text: str) -> dict[str, Any]:
        score = None
        reasoning = ""

        score_match = re.search(r"SCORE:\s*(\d(?:\.\d)?)", text)
        if score_match:
            score = float(score_match.group(1))
            score = max(1.0, min(5.0, score))

        reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        if score is None:
            nums = re.findall(r"\b([1-5])\b", text)
            if nums:
                score = float(nums[0])
            reasoning = reasoning if reasoning else text.strip()

        return {"score": score, "reasoning": reasoning}

    @staticmethod
    def _compute_category_breakdown(
        per_query: list[GenerationMetrics],
    ) -> dict[str, dict[str, float]]:
        by_cat: dict[str, list[GenerationMetrics]] = defaultdict(list)
        for m in per_query:
            cat = m.category or "unknown"
            by_cat[cat].append(m)

        result = {}
        for cat, metrics in sorted(by_cat.items()):
            sim = [m.semantic_similarity for m in metrics if m.semantic_similarity is not None]
            faith = [m.faithfulness_score for m in metrics if m.faithfulness_score is not None]
            rel = [m.relevance_score for m in metrics if m.relevance_score is not None]

            result[cat] = {
                "count": len(metrics),
                "mean_semantic_similarity": round(float(np.mean(sim)), 4) if sim else 0.0,
                "mean_faithfulness": round(float(np.mean(faith)), 4) if faith else 0.0,
                "mean_relevance": round(float(np.mean(rel)), 4) if rel else 0.0,
                "refusal_rate": round(sum(1 for m in metrics if m.is_refusal) / len(metrics), 4),
            }

        return result
