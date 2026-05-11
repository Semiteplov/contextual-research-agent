from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from contextual_research_agent.agent.llm import LLMProvider
from contextual_research_agent.agent.node_executor import ParallelExecutorNode
from contextual_research_agent.agent.nodes.critic import CriticNode
from contextual_research_agent.agent.nodes.generator import GeneratorNode
from contextual_research_agent.agent.nodes.planner import PlannerNode
from contextual_research_agent.agent.nodes.retriever import RetrieverNode
from contextual_research_agent.agent.nodes.router import router_node
from contextual_research_agent.agent.nodes.synthesizer import SynthesizerNode
from contextual_research_agent.agent.state import (
    AgentState,
    AgentStatus,
    CriticVerdict,
    QueryComplexity,
)
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.pipeline import GenerationPipeline
from contextual_research_agent.retrieval.pipeline import RetrievalPipeline

logger = get_logger(__name__)


def _route_after_router(state: AgentState) -> str:
    """Decide next node after Router."""
    complexity = state.get("complexity", QueryComplexity.SIMPLE.value)
    if complexity in (QueryComplexity.COMPLEX.value, QueryComplexity.MULTI_ASPECT.value):
        return "planner"
    return "retriever"


def _route_after_retriever(state: AgentState) -> str:
    if state.get("status") == AgentStatus.FAILED.value:
        return "synthesizer"

    complexity = state.get("complexity", QueryComplexity.SIMPLE.value)
    if complexity in (QueryComplexity.COMPLEX.value, QueryComplexity.MULTI_ASPECT.value):
        return "parallel_executor"
    return "generator"


def _route_after_critic(state: AgentState) -> str:
    """Decide next node after Critic: retry or finish."""
    feedback = state.get("critic_feedback", {})
    verdict = feedback.get("verdict", CriticVerdict.PASS.value)
    retry_count = state.get("retry_count", 0)
    complexity = state.get("complexity", QueryComplexity.SIMPLE.value)

    if state.get("status") == AgentStatus.FAILED.value:
        return "end"

    if complexity in (QueryComplexity.COMPLEX.value, QueryComplexity.MULTI_ASPECT.value):
        return "end"

    if verdict == CriticVerdict.FAIL.value and retry_count == 1:
        events = state.get("trace_events", [])
        generator_retries = sum(
            1
            for e in events
            if e.get("node") == "generator" and e.get("data", {}).get("retry", 0) > 0
        )
        if generator_retries == 0:
            return "generator"

    return "synthesizer"


def _check_failed(state: AgentState) -> str:
    if state.get("status") == AgentStatus.FAILED.value:
        return "end"
    return "continue"


def build_agent_graph(
    retrieval_pipeline: RetrievalPipeline,
    generation_pipeline: GenerationPipeline,
    llm: LLMProvider,
) -> CompiledStateGraph:
    planner = PlannerNode(llm=llm)
    retriever = RetrieverNode(pipeline=retrieval_pipeline)
    generator = GeneratorNode(pipeline=generation_pipeline)
    parallel_executor = ParallelExecutorNode(generation_pipeline=generation_pipeline)
    critic = CriticNode(llm=llm)
    synthesizer = SynthesizerNode(llm=llm)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("generator", generator)
    graph.add_node("parallel_executor", parallel_executor)
    graph.add_node("critic", critic)
    graph.add_node("synthesizer", synthesizer)

    # Entry point
    graph.set_entry_point("router")

    # Router → Planner or Retriever
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "planner": "planner",
            "retriever": "retriever",
        },
    )

    # Planner → Retriever
    graph.add_edge("planner", "retriever")

    # Retriever → Generator or ParallelExecutor
    graph.add_conditional_edges(
        "retriever",
        _route_after_retriever,
        {
            "generator": "generator",
            "parallel_executor": "parallel_executor",
            "synthesizer": "synthesizer",
        },
    )

    # Generator → Critic
    graph.add_conditional_edges(
        "generator",
        _check_failed,
        {
            "continue": "critic",
            "end": "synthesizer",
        },
    )

    graph.add_conditional_edges(
        "parallel_executor",
        _check_failed,
        {
            "continue": "synthesizer",
            "end": "synthesizer",
        },
    )

    # Critic → Generator (retry) or Synthesizer (done)
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "generator": "generator",
            "end": END,
        },
    )

    def _after_synthesizer(state: AgentState) -> str:
        if state.get("status") == AgentStatus.FAILED.value:
            return "end"
        events = state.get("trace_events", [])
        critic_already_ran = any(e.get("node") == "critic" for e in events)
        if critic_already_ran:
            return "end"
        return "critic"

    graph.add_conditional_edges(
        "synthesizer",
        _after_synthesizer,
        {
            "critic": "critic",
            "end": END,
        },
    )

    compiled = graph.compile()
    logger.info("Multi-agent graph compiled: 7 nodes, conditional routing")

    return compiled
