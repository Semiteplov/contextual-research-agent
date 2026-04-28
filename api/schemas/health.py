from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded | unhealthy")
    version: str = "0.1.0"
    service_initialized: bool = False
    qdrant_reachable: bool = False
    llm_reachable: bool = False
    uptime_seconds: float = 0.0


class StatsResponse(BaseModel):
    collection: str
    total_chunks: int = 0
    total_documents: int = 0
    embedding_model: str = ""
    llm_model: str = ""
    channels_enabled: list[str] = Field(default_factory=list)


class ConfigResponse(BaseModel):
    agent: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    api: dict[str, str | int | bool] = Field(default_factory=dict)
