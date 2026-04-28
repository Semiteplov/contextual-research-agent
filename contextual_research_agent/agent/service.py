from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from contextual_research_agent.agent.graph import build_agent_graph
from contextual_research_agent.agent.llm import LLMProvider, create_llm_provider
from contextual_research_agent.agent.state import AgentState, create_initial_state
from contextual_research_agent.agent.tracing.trace import AgentTrace
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.config import GenerationConfig
from contextual_research_agent.generation.pipeline import GenerationPipeline
from contextual_research_agent.retrieval.factory import build_pipeline

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """User-facing result from the multi-agent system."""

    answer: str
    trace: AgentTrace

    @property
    def debug_summary(self) -> dict[str, Any]:
        return self.trace.to_debug_summary()

    @property
    def chunks_for_display(self) -> list[dict[str, Any]]:
        return self.trace.get_chunks_for_display()

    @property
    def latency_breakdown(self) -> dict[str, float]:
        return self.trace.get_latency_breakdown()


class ResearchAssistantService:
    """
    Stateful service holding initialized pipelines and compiled graph.

    Usage:
        service = await ResearchAssistantService.create(config)
        result = await service.query("How does LoRA work?")
        print(result.answer)
        print(result.debug_summary)

        await service.shutdown()
    """

    def __init__(
        self,
        graph,
        llm: LLMProvider,
        retrieval_pipeline,
        generation_pipeline: GenerationPipeline,
    ):
        self._graph = graph
        self._llm = llm
        self._retrieval_pipeline = retrieval_pipeline
        self._generation_pipeline = generation_pipeline

    @classmethod
    async def create(
        cls,
        # Retrieval params
        collection: str = "peft_hybrid",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        rerank: bool = True,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        channels: str = "dense,sparse,graph_entity,paper_level",
        # Generation params
        llm_provider: str = "ollama",
        llm_model: str = "qwen3:8b",
        llm_host: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> ResearchAssistantService:
        logger.info("Initializing ResearchAssistantService...")
        t_start = time.perf_counter()

        # Parse channels
        channel_list = [c.strip() for c in channels.split(",") if c.strip()]

        # Build retrieval pipeline
        retrieval_pipeline, _ = await build_pipeline(
            collection=collection,
            embedding_model=embedding_model,
            rerank_enabled=rerank,
            rerank_model=rerank_model,
            device=device,
            enabled_channels=channel_list,
        )

        # Build LLM + generation pipeline
        llm = create_llm_provider(
            provider=llm_provider,
            model=llm_model,
            host=llm_host,
        )
        gen_config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            auto_detect_mode=True,
        )
        gen_pipeline = GenerationPipeline(llm=llm, config=gen_config)

        # Build graph
        graph = build_agent_graph(
            retrieval_pipeline=retrieval_pipeline,
            generation_pipeline=gen_pipeline,
            llm=llm,
        )

        elapsed = time.perf_counter() - t_start
        logger.info("Service initialized in %.1fs", elapsed)

        return cls(
            graph=graph,
            llm=llm,
            retrieval_pipeline=retrieval_pipeline,
            generation_pipeline=gen_pipeline,
        )

    async def query(
        self,
        text: str,
        mode: str | None = None,
        document_ids: list[str] | None = None,
    ) -> QueryResult:
        """
        Execute a query through the multi-agent graph.

        Args:
            text: User query.
            mode: Optional mode override (bypasses Router intent detection).
            document_ids: Optional document filter.

        Returns:
            QueryResult with answer and full trace.
        """
        initial_state = create_initial_state(
            query=text,
            mode_override=mode,
            document_ids=document_ids,
        )

        t_start = time.perf_counter()
        final_state = await self._graph.ainvoke(initial_state)
        total_ms = (time.perf_counter() - t_start) * 1000

        final_state["total_latency_ms"] = total_ms

        trace = AgentTrace.from_state(final_state)

        answer = trace.final_answer or trace.generated_answer or ""
        if trace.error:
            answer = f"Error: {trace.error}"

        return QueryResult(answer=answer, trace=trace)

    async def shutdown(self) -> None:
        """Clean up resources."""
        try:
            await self._llm.close()
        except Exception as e:
            logger.warning("LLM close failed: %s", e)
        logger.info("Service shut down")
