from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, description="Bind port")
    log_level: str = Field(default="info", description="uvicorn log level")
    workers: int = Field(default=1, description="Number of uvicorn workers")
    reload: bool = Field(default=False, description="Auto-reload on changes (dev only)")

    api_keys: str = Field(
        default="",
        description="Comma-separated valid API keys. Empty = auth disabled.",
    )

    cors_origins: str = Field(
        default="http://localhost:7860,http://localhost:3000",
        description="Comma-separated allowed origins",
    )

    max_query_length: int = Field(default=2000, description="Max query string length")
    request_timeout: float = Field(
        default=600.0,
        description="Request timeout in seconds (10 min for slow LLM)",
    )

    @property
    def api_keys_set(self) -> set[str]:
        if not self.api_keys.strip():
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys_set)

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    collection: str = Field(default="peft_hybrid")
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    rerank: bool = Field(default=True)
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    device: str | None = Field(default=None)
    channels: str = Field(default="dense,sparse,graph_entity,paper_level")

    llm_provider: str = Field(default="ollama")
    llm_model: str = Field(default="qwen3:8b")
    llm_host: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=64, le=16384)


def get_api_settings() -> APISettings:
    return APISettings()


def get_agent_settings() -> AgentSettings:
    return AgentSettings()
