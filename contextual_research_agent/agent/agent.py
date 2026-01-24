from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextual_research_agent.agent.retriever import RetrievalResult, Retriever, create_retriever
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

logger = logging.get_logger(__name__)


@dataclass
class AssistantConfig:
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_device: str | None = None
    embedding_batch_size: int = 8

    tokenizer_model: str = "Qwen/Qwen3-Embedding-4B"
    max_chunk_tokens: int = 512
    include_section_context: bool = True

    default_top_k: int = 10
    score_threshold: float | None = None


class ResearchAssistant:
    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        retriever: Retriever,
        config: AssistantConfig,
    ):
        self._ingestion = ingestion_pipeline
        self._retriever = retriever
        self._config = config

        logger.info("ResearchAssistant initialized")

    @classmethod
    async def create(
        cls,
        config: AssistantConfig | None = None,
        s3_client: S3Client | None = None,
    ) -> ResearchAssistant:
        config = config or AssistantConfig()
        s3_client = s3_client or S3Client()

        logger.info(f"Creating ResearchAssistant with embedding model: {config.embedding_model}")

        embedder = create_hf_embedder(
            model=config.embedding_model,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
        )

        parser = create_docling_parser(
            s3_client=s3_client,
            embedding_model=config.tokenizer_model,
            max_tokens=config.max_chunk_tokens,
            include_section_context=config.include_section_context,
        )

        vector_store = create_qdrant_store(
            embedding_dim=embedder.dimension,
        )

        ingestion_pipeline = create_ingestion_pipeline(
            parser=parser,
            embedder=embedder,
            vector_store=vector_store,
        )

        retriever = create_retriever(
            embedder=embedder,
            vector_store=vector_store,
            default_top_k=config.default_top_k,
            default_score_threshold=config.score_threshold,
        )

        return cls(
            ingestion_pipeline=ingestion_pipeline,
            retriever=retriever,
            config=config,
        )

    async def ingest(self, file_path: str) -> IngestionResult:
        return await self._ingestion.ingest(file_path)

    async def ingest_batch(
        self,
        file_paths: list[str],
        continue_on_error: bool = True,
    ) -> list[IngestionResult]:
        return await self._ingestion.ingest_batch(file_paths, continue_on_error)

    async def reindex(self, document_id: str, file_path: str) -> IngestionResult:
        return await self._ingestion.reindex_document(document_id, file_path)

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

    async def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        result = await self.retrieve(query, top_k, document_ids)
        return result.context

    async def get_stats(self) -> dict[str, Any]:
        return await self._ingestion._vector_store.get_stats()


async def create_research_assistant(
    embedding_model: str = "Qwen/Qwen3-Embedding-4B",
    max_chunk_tokens: int = 512,
    default_top_k: int = 10,
) -> ResearchAssistant:
    config = AssistantConfig(
        embedding_model=embedding_model,
        tokenizer_model=embedding_model,
        max_chunk_tokens=max_chunk_tokens,
        default_top_k=default_top_k,
    )

    return await ResearchAssistant.create(config)
