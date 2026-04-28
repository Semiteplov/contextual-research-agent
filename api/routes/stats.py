from __future__ import annotations

import logging

from api.auth import verify_api_key
from api.config import AgentSettings, APISettings, get_agent_settings, get_api_settings
from api.lifespan import get_service
from api.schemas import ConfigResponse, StatsResponse
from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["stats"])


@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Corpus statistics (chunks, documents, configuration)",
)
async def stats(
    settings: AgentSettings = Depends(get_agent_settings),
    _api_key: str | None = Depends(verify_api_key),
) -> StatsResponse:
    """Return corpus statistics from Qdrant."""
    service = get_service()

    try:
        retrieval = service._retrieval_pipeline
        store = getattr(retrieval, "_vector_store", None)

        total_chunks = 0
        total_documents = 0
        if store is not None:
            qdrant_stats = await store.get_stats()
            total_chunks = qdrant_stats.get("vectors_count", 0)
            total_documents = qdrant_stats.get("indexed_documents", 0)

    except Exception as e:
        logger.warning("Failed to fetch Qdrant stats: %s", e)
        total_chunks = 0
        total_documents = 0

    channels = [c.strip() for c in settings.channels.split(",") if c.strip()]

    return StatsResponse(
        collection=settings.collection,
        total_chunks=total_chunks,
        total_documents=total_documents,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        channels_enabled=channels,
    )


@router.get(
    "/config",
    response_model=ConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Current system configuration (debug)",
)
async def config(
    agent_settings: AgentSettings = Depends(get_agent_settings),
    api_settings: APISettings = Depends(get_api_settings),
    _api_key: str | None = Depends(verify_api_key),
) -> ConfigResponse:
    """Return current configuration (no secrets)."""
    return ConfigResponse(
        agent={
            "collection": agent_settings.collection,
            "embedding_model": agent_settings.embedding_model,
            "rerank": agent_settings.rerank,
            "rerank_model": agent_settings.rerank_model,
            "device": agent_settings.device,
            "channels": agent_settings.channels,
            "llm_provider": agent_settings.llm_provider,
            "llm_model": agent_settings.llm_model,
            "llm_host": agent_settings.llm_host,
            "temperature": agent_settings.temperature,
            "max_tokens": agent_settings.max_tokens,
        },
        api={
            "host": api_settings.host,
            "port": api_settings.port,
            "log_level": api_settings.log_level,
            "auth_enabled": api_settings.auth_enabled,
            "max_query_length": api_settings.max_query_length,
        },
    )
