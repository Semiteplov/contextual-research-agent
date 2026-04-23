from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from contextual_research_agent.agent.llm import (
    GenerationResult as LLMResult,
)
from contextual_research_agent.agent.llm import (
    LLMProvider,
)
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.config import (
    CognitiveMode,
    GenerationConfig,
)
from contextual_research_agent.generation.prompts import (
    format_context_for_prompt,
    get_prompt_template,
)
from contextual_research_agent.retrieval.types import RetrievalResult

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    # Answer
    answer: str
    mode: CognitiveMode

    # LLM metadata
    model: str
    llm_latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    query: str = ""
    retrieval_latency_ms: float = 0.0
    num_chunks_used: int = 0
    chunk_ids_used: list[str] = field(default_factory=list)
    document_ids_used: list[str] = field(default_factory=list)
    intent: str = ""

    total_latency_ms: float = 0.0

    system_prompt: str = ""
    user_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "mode": self.mode.value,
            "model": self.model,
            "query": self.query,
            "intent": self.intent,
            "llm_latency_ms": self.llm_latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "num_chunks_used": self.num_chunks_used,
            "chunk_ids_used": self.chunk_ids_used,
            "document_ids_used": self.document_ids_used,
        }


class GenerationPipeline:
    """Generates answers from retrieval results using mode-specific prompts.

    Usage:
        pipeline = GenerationPipeline(llm=ollama_provider, config=gen_config)
        response = await pipeline.generate(retrieval_result)
        response = await pipeline.generate(retrieval_result, mode=CognitiveMode.CRITICAL_REVIEW)
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: GenerationConfig | None = None,
    ):
        self._llm = llm
        self._config = config or GenerationConfig()
        logger.info(
            "GenerationPipeline initialized: model=%s, default_mode=%s",
            llm.model_name,
            self._config.default_mode.value,
        )

    async def generate(
        self,
        retrieval_result: RetrievalResult,
        mode: CognitiveMode | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RAGResponse:
        """Generate answer from retrieval result.

        Args:
            retrieval_result: Output of RetrievalPipeline.retrieve().
            mode: Override cognitive mode. If None, auto-detect from intent.
            temperature: Override temperature.
            max_tokens: Override max_tokens.

        Returns:
            RAGResponse with answer, provenance, and full latency breakdown.
        """
        t_start = time.perf_counter()

        resolved_mode = self._resolve_mode(retrieval_result, mode)

        template = get_prompt_template(
            mode=resolved_mode,
            require_citation=self._config.require_citation,
        )

        candidates_for_prompt = retrieval_result.candidates
        if len(candidates_for_prompt) > self._config.max_context_chunks:
            candidates_for_prompt = candidates_for_prompt[: self._config.max_context_chunks]
            context_str = self._format_candidates_as_context(candidates_for_prompt)
        else:
            context_str = retrieval_result.context

        context_str = format_context_for_prompt(retrieval_result.context)

        user_prompt = template.user.format(
            context=context_str,
            query=retrieval_result.query,
        )

        effective_max_tokens = max_tokens or self._config.max_tokens
        effective_max_tokens = self._adjust_max_tokens(resolved_mode, effective_max_tokens)

        llm_result: LLMResult = await self._llm.generate(
            prompt=user_prompt,
            system_prompt=template.system,
            temperature=temperature or self._config.temperature,
            max_tokens=effective_max_tokens,
        )

        total_ms = (time.perf_counter() - t_start) * 1000

        chunk_ids = [c.chunk_id for c in candidates_for_prompt]
        doc_ids = list({c.document_id for c in candidates_for_prompt})

        response = RAGResponse(
            answer=llm_result.text,
            mode=resolved_mode,
            model=llm_result.model,
            llm_latency_ms=llm_result.latency_ms,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            query=retrieval_result.query,
            retrieval_latency_ms=retrieval_result.total_latency_ms,
            num_chunks_used=len(candidates_for_prompt),
            chunk_ids_used=chunk_ids,
            document_ids_used=doc_ids,
            intent=retrieval_result.intent,
            total_latency_ms=total_ms,
            system_prompt=template.system,
            user_prompt=user_prompt,
        )

        logger.info(
            "Generation complete: mode=%s, model=%s, "
            "chunks=%d, llm=%.0fms, total=%.0fms, tokens=%d+%d",
            resolved_mode.value,
            llm_result.model,
            len(chunk_ids),
            llm_result.latency_ms,
            total_ms,
            llm_result.prompt_tokens,
            llm_result.completion_tokens,
        )

        return response

    async def generate_from_context(
        self,
        query: str,
        context: str,
        mode: CognitiveMode | str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RAGResponse:
        resolved_mode = self._parse_mode(mode) if mode else self._config.default_mode

        template = get_prompt_template(
            mode=resolved_mode,
            require_citation=self._config.require_citation,
        )

        user_prompt = template.user.format(context=context, query=query)

        t_start = time.perf_counter()
        llm_result = await self._llm.generate(
            prompt=user_prompt,
            system_prompt=template.system,
            temperature=temperature or self._config.temperature,
            max_tokens=max_tokens or self._config.max_tokens,
        )
        total_ms = (time.perf_counter() - t_start) * 1000

        return RAGResponse(
            answer=llm_result.text,
            mode=resolved_mode,
            model=llm_result.model,
            llm_latency_ms=llm_result.latency_ms,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            query=query,
            total_latency_ms=total_ms,
            system_prompt=template.system,
            user_prompt=user_prompt,
        )

    def _resolve_mode(
        self,
        retrieval_result: RetrievalResult,
        mode_override: CognitiveMode | str | None,
    ) -> CognitiveMode:
        if mode_override:
            return self._parse_mode(mode_override)

        if self._config.auto_detect_mode and retrieval_result.intent:
            mode = CognitiveMode.from_intent(retrieval_result.intent)
            if mode == CognitiveMode.COMPARISON:
                mode = self._refine_comparison_intent(retrieval_result.query, mode)
            return mode

        return self._config.default_mode

    @staticmethod
    def _refine_comparison_intent(query: str, current: CognitiveMode) -> CognitiveMode:
        factual_patterns = [
            r"^(how much|how many|what is the|what are the|what accuracy|what percentage)",
            r"^(what score|what result|what performance|what rank|how does .+ achieve)",
            r"^(what .+ does .+ use|how .+ parameters)",
        ]
        q_lower = query.lower().strip()
        for pattern in factual_patterns:
            if re.match(pattern, q_lower):
                return CognitiveMode.FACTUAL_QA
        return current

    @staticmethod
    def _parse_mode(mode: CognitiveMode | str) -> CognitiveMode:
        if isinstance(mode, CognitiveMode):
            return mode
        return CognitiveMode(mode)

    @property
    def config(self) -> GenerationConfig:
        return self._config

    @staticmethod
    def _format_candidates_as_context(candidates: list) -> str:
        parts: list[str] = []
        for c in candidates:
            section_type = c.chunk.metadata.get("section_type", "unknown")
            parts.append(
                f"[{c.chunk_id}] (section: {section_type}, score: {c.score:.3f})\n{c.chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _adjust_max_tokens(mode: CognitiveMode, base_max: int) -> int:
        mode_minimums = {
            CognitiveMode.FACTUAL_QA: 256,
            CognitiveMode.SUMMARIZATION: 512,
            CognitiveMode.COMPARISON: 1024,
            CognitiveMode.CRITICAL_REVIEW: 1024,
            CognitiveMode.METHODOLOGICAL_AUDIT: 1024,
            CognitiveMode.IDEA_GENERATION: 1024,
        }
        minimum = mode_minimums.get(mode, 512)
        return max(base_max, minimum)
