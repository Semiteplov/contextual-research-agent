from __future__ import annotations

import logging
import time

import httpx
from api.config import AgentSettings, APISettings, get_agent_settings, get_api_settings
from api.lifespan import get_startup_time
from api.schemas import HealthResponse
from contextual_research_agent.common.settings import get_settings
from fastapi import APIRouter, Depends, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness check — always returns OK if the process is alive",
)
async def health_check() -> HealthResponse:
    startup = get_startup_time()
    uptime = time.time() - startup if startup > 0 else 0.0

    return HealthResponse(
        status="ok",
        service_initialized=startup > 0,
        uptime_seconds=round(uptime, 1),
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness check — verifies dependencies (Qdrant, LLM)",
)
async def readiness_check(
    agent_settings: AgentSettings = Depends(get_agent_settings),
    api_settings: APISettings = Depends(get_api_settings),
) -> HealthResponse:
    startup = get_startup_time()
    uptime = time.time() - startup if startup > 0 else 0.0

    qdrant_ok = await _check_qdrant()
    llm_ok = await _check_llm(
        provider=agent_settings.llm_provider,
        host=agent_settings.llm_host,
    )

    overall_status = "ok" if (startup > 0 and qdrant_ok and llm_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        service_initialized=startup > 0,
        qdrant_reachable=qdrant_ok,
        llm_reachable=llm_ok,
        uptime_seconds=round(uptime, 1),
    )


async def _check_qdrant() -> bool:
    try:
        settings = get_settings()
        qdrant_url = f"http://{settings.qdrant.host}:{settings.qdrant.port}"

        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{qdrant_url}/collections")
            return response.status_code == 200
    except Exception as e:
        logger.debug("Qdrant check failed: %s", e)
        return False


async def _check_llm(provider: str, host: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            if provider == "ollama":
                response = await client.get(f"{host}/api/tags")
            else:
                response = await client.get(f"{host}/v1/models")
            return response.status_code == 200
    except Exception as e:
        logger.debug("LLM check failed: %s", e)
        return False
