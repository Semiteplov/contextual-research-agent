from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from contextual_research_agent.agent.config import AgentConfig, MLflowConfig, load_config
from contextual_research_agent.agent.llm import LLMProvider, create_llm_provider
from contextual_research_agent.agent.prompts import (
    CognitiveMode,
    format_context,
    format_user_prompt,
    get_system_prompt,
)
from contextual_research_agent.agent.retriever import RetrievalResult, Retriever, create_retriever
from contextual_research_agent.agent.state import QueryPatch, QueryState
from contextual_research_agent.common import logging
from contextual_research_agent.data.storage.s3_client import S3Client
from contextual_research_agent.ingestion.embeddings import create_hf_embedder
from contextual_research_agent.ingestion.parsers import create_docling_parser
from contextual_research_agent.ingestion.pipeline import (
    IngestionPipeline,
    IngestionResult,
    create_ingestion_pipeline,
)
from contextual_research_agent.ingestion.vectorstores.qdrant_store import create_qdrant_store
from contextual_research_agent.tracking.mlflow_tracking import (
    MLflowManager,
    setup_mlflow,
)

logger = logging.get_logger(__name__)


@dataclass
class AgentResponse:
    query: str
    mode: CognitiveMode
    answer: str
    citations: list[str]
    retrieval: RetrievalResult
    context_used: str
    latency: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return self.latency.get("total_ms", 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode.value,
            "answer": self.answer,
            "citations": self.citations,
            "num_chunks_retrieved": len(self.retrieval.chunks),
            "latency": self.latency,
            "tokens": self.tokens,
        }


class RAGAgent:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        ingestion: IngestionPipeline,
        config: AgentConfig,
        mlflow_manager: MLflowManager | None = None,
    ):
        self._retriever = retriever
        self._llm = llm
        self._ingestion = ingestion
        self._config = config
        self._mlflow = mlflow_manager

        self._query_graph = self._build_query_graph()

        logger.info(
            f"RAGAgent initialized: modes={config.modes.available}, "
            f"llm={llm.model_name}, "
            f"mlflow={mlflow_manager.enabled if mlflow_manager else False}"
        )

    @classmethod
    async def create(
        cls,
        config_path: str | None = None,
        config: AgentConfig | None = None,
    ) -> RAGAgent:
        if config is None:
            config = load_config(config_path)

        logger.info(f"Creating RAGAgent: {config.name}")

        mlflow_manager = None
        if config.mlflow.enabled:
            mlflow_config = MLflowConfig(
                enabled=config.mlflow.enabled,
                tracking_uri=config.mlflow.tracking_uri,
                experiment_name=config.mlflow.experiment_name,
                artifact_location=config.mlflow.artifact_location,
                autolog_enabled=config.mlflow.autolog_enabled,
                log_inputs_outputs=config.mlflow.log_inputs_outputs,
                default_tags=[
                    config.name,
                    str(config.version),
                ],
            )
            mlflow_manager = setup_mlflow(mlflow_config)

        s3_client = S3Client()

        embedder = create_hf_embedder(
            model=config.embedding.model,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            normalize=config.embedding.normalize,
            query_instruction=config.embedding.query_instruction,
            passage_instruction=config.embedding.passage_instruction,
        )

        parser = create_docling_parser(
            s3_client=s3_client,
            embedding_model=config.parser.tokenizer_model,
            max_tokens=config.parser.max_tokens,
            include_section_context=config.parser.include_section_context,
        )

        vector_store = create_qdrant_store(
            collection_name=config.vector_store.collection_name,
            embedding_dim=embedder.dimension,
            distance=config.vector_store.distance,
            on_disk=config.vector_store.on_disk,
        )

        retriever = create_retriever(
            embedder=embedder,
            vector_store=vector_store,
            default_top_k=config.retrieval.default_top_k,
            default_score_threshold=config.retrieval.score_threshold,
        )

        llm = create_llm_provider(
            provider=config.llm.provider,
            model=config.llm.model,
            host=config.llm.host,
        )

        ingestion = create_ingestion_pipeline(
            parser=parser,
            embedder=embedder,
            vector_store=vector_store,
        )

        return cls(
            retriever=retriever,
            llm=llm,
            ingestion=ingestion,
            config=config,
            mlflow_manager=mlflow_manager,
        )

    async def _retrieve_node(self, state: QueryState) -> QueryPatch:
        start = time.perf_counter()

        try:
            retrieval = await self._retriever.retrieve(
                query=state["query"],
                top_k=state["top_k"],
                document_ids=state.get("document_ids"),
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            latency = dict(state.get("latency", {}))
            latency["retrieve_ms"] = elapsed_ms

            chunks_for_context = [
                {
                    "id": rc.chunk.id,
                    "text": rc.chunk.text,
                    "score": rc.score,
                    "section": rc.chunk.section,
                }
                for rc in retrieval.chunks
            ]
            context = format_context(chunks_for_context, max_tokens=4000)

            logger.debug(f"Retrieved {len(retrieval.chunks)} chunks ({elapsed_ms:.0f}ms)")

            return {
                "retrieval": retrieval,
                "context": context,
                "latency": latency,
                "status": "retrieved",
            }

        except Exception as e:
            logger.exception(f"Retrieval failed: {e}")
            return {
                "status": "failed",
                "error": f"Retrieval failed: {e}",
            }

    async def _generate_node(self, state: QueryState) -> QueryPatch:
        start = time.perf_counter()

        if state.get("status") == "failed":
            return {}

        try:
            mode = CognitiveMode(state["mode"])
            system_prompt = get_system_prompt(mode)
            user_prompt = format_user_prompt(mode, state["context"], state["query"])

            generation = await self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self._config.llm.temperature,
                max_tokens=self._config.llm.max_tokens,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            latency = dict(state.get("latency", {}))
            latency["generate_ms"] = elapsed_ms

            retrieval = state.get("retrieval")
            citations = self._extract_citations(generation.text, retrieval) if retrieval else []

            logger.debug(f"Generated response ({elapsed_ms:.0f}ms)")

            return {
                "generation": generation,
                "citations": citations,
                "latency": latency,
                "status": "completed",
            }

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return {
                "status": "failed",
                "error": f"Generation failed: {e}",
            }

    def _should_continue(self, state: QueryState) -> str:
        if state.get("status") == "failed":
            return "end"
        return "continue"

    def _build_query_graph(self) -> CompiledStateGraph:
        graph = StateGraph(QueryState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("retrieve")

        graph.add_conditional_edges(
            "retrieve",
            self._should_continue,
            {"continue": "generate", "end": END},
        )

        graph.add_edge("generate", END)

        return graph.compile()

    async def query(
        self,
        query: str,
        mode: str | CognitiveMode = "qa",
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> AgentResponse:
        start_total = time.perf_counter()

        mode_str = mode.value if isinstance(mode, CognitiveMode) else mode.lower()

        if mode_str not in self._config.modes.available:
            raise ValueError(
                f"Mode '{mode}' not available. Available: {self._config.modes.available}"
            )

        initial_state: QueryState = {
            "query": query,
            "mode": mode_str,
            "document_ids": document_ids,
            "top_k": top_k or self._config.retrieval.default_top_k,
            "retrieval": None,
            "context": "",
            "generation": None,
            "citations": [],
            "latency": {},
            "status": "pending",
            "error": None,
        }

        final_state = await self._query_graph.ainvoke(initial_state)

        latency = dict(final_state.get("latency", {}))
        latency["total_ms"] = (time.perf_counter() - start_total) * 1000

        if final_state.get("status") == "failed":
            error = final_state.get("error", "Unknown error")
            logger.error(f"Query failed: {error}")

            return AgentResponse(
                query=query,
                mode=CognitiveMode(mode_str),
                answer=f"Error: {error}",
                citations=[],
                retrieval=final_state.get("retrieval")
                or RetrievalResult(query=query, chunks=[], latency_ms=0),
                context_used="",
                latency=latency,
                tokens={},
            )

        generation = final_state.get("generation")

        logger.info(
            f"Query processed: mode={mode_str}, "
            f"chunks={len(final_state['retrieval'].chunks) if final_state.get('retrieval') else 0}, "
            f"latency={latency.get('total_ms', 0):.0f}ms"
        )

        retrieval = final_state.get("retrieval")
        if retrieval is None:
            retrieval = RetrievalResult(query=query, chunks=[], latency_ms=0)

        return AgentResponse(
            query=query,
            mode=CognitiveMode(mode_str),
            answer=generation.text if generation else "",
            citations=final_state.get("citations", []),
            retrieval=retrieval,
            context_used=final_state.get("context", ""),
            latency=latency,
            tokens={
                "prompt": generation.prompt_tokens if generation else 0,
                "completion": generation.completion_tokens if generation else 0,
                "total": generation.total_tokens if generation else 0,
            },
        )

    async def summarize(
        self,
        document_id: str,
        top_k: int = 15,
    ) -> AgentResponse:
        return await self.query(
            query="Summarize this paper",
            mode=CognitiveMode.SUMMARIZE,
            document_ids=[str(document_id)],
            top_k=top_k,
        )

    async def ask(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> AgentResponse:
        return await self.query(
            query=question,
            mode=CognitiveMode.QA,
            document_ids=document_ids,
            top_k=top_k,
        )

    async def ingest(self, file_path: str) -> IngestionResult:
        try:
            results = await self._ingestion.ingest(file_path)
        finally:
            await self._ingestion._vector_store.close()
        return results

    async def ingest_batch(
        self,
        file_paths: list[str],
        continue_on_error: bool = True,
    ) -> list[IngestionResult]:
        try:
            results = await self._ingestion.ingest_batch(file_paths, continue_on_error)
        finally:
            await self._ingestion._vector_store.close()
        return results

    def _extract_citations(
        self,
        text: str,
        retrieval: RetrievalResult,
    ) -> list[str]:
        pattern = r"\[([^\]]+)\]"
        cited_ids = set(re.findall(pattern, text))

        valid_ids = {rc.chunk.id for rc in retrieval.chunks}

        return sorted(cited_ids & valid_ids)

    async def get_stats(self) -> dict[str, Any]:
        return await self._ingestion._vector_store.get_stats()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> RetrievalResult:
        return await self._retriever.retrieve(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )

    async def reindex(self, document_id: str, file_path: str) -> IngestionResult:
        await self._ingestion._vector_store.delete_by_document(document_id)
        return await self.ingest(file_path)

    @property
    def mlflow_enabled(self) -> bool:
        return self._mlflow is not None and self._mlflow.enabled

    def get_mlflow_manager(self) -> MLflowManager | None:
        return self._mlflow


async def create_agent(
    config_path: str | None = None,
) -> RAGAgent:
    return await RAGAgent.create(config_path=config_path)
