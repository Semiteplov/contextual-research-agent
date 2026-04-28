from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from api.config import get_agent_settings
from contextual_research_agent.agent.service import ResearchAssistantService
from fastapi import FastAPI

logger = logging.getLogger(__name__)


_service: "ResearchAssistantService | None" = None
_startup_time: float | None = None


def get_service() -> "ResearchAssistantService":
    if _service is None:
        raise RuntimeError(
            "ResearchAssistantService not initialized. Wait for startup to complete."
        )
    return _service


def get_startup_time() -> float:
    return _startup_time or 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service, _startup_time

    logger.info("Initializing ResearchAssistantService...")
    t0 = time.perf_counter()

    settings = get_agent_settings()

    _service = await ResearchAssistantService.create(
        collection=settings.collection,
        embedding_model=settings.embedding_model,
        rerank=settings.rerank,
        rerank_model=settings.rerank_model,
        device=settings.device,
        channels=settings.channels,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_host=settings.llm_host,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    _startup_time = time.time()
    elapsed = time.perf_counter() - t0
    logger.info("Service initialized in %.1fs — ready to serve requests", elapsed)

    try:
        yield
    finally:
        logger.info("Shutting down service...")
        if _service is not None:
            try:
                await _service.shutdown()
            except Exception as e:
                logger.warning("Service shutdown error: %s", e)
        _service = None
        _startup_time = None
        logger.info("Service shut down")
