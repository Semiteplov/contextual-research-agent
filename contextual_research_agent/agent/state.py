from typing import TypedDict

from contextual_research_agent.agent.llm import GenerationResult
from contextual_research_agent.agent.retriever import RetrievalResult


class QueryState(TypedDict):
    query: str
    mode: str
    document_ids: list[str] | None
    top_k: int
    retrieval: RetrievalResult | None
    context: str
    generation: GenerationResult | None
    citations: list[str]
    latency: dict[str, float]
    status: str
    error: str | None


class QueryPatch(TypedDict, total=False):
    retrieval: RetrievalResult | None
    context: str
    generation: GenerationResult | None
    citations: list[str]
    latency: dict[str, float]
    status: str
    error: str | None
