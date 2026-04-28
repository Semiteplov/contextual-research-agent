from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class UISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_url: str = Field(
        default="http://localhost:8000",
        description="FastAPI backend base URL (HTTP)",
    )
    ws_url: str = Field(
        default="ws://localhost:8000",
        description="FastAPI backend WebSocket URL",
    )
    api_key: str = Field(
        default="",
        description="API key for backend authentication",
    )

    ui_host: str = Field(default="0.0.0.0")
    ui_port: int = Field(default=7860)
    ui_share: bool = Field(default=False, description="Create public Gradio link")
    ui_debug: bool = Field(default=False)

    request_timeout: float = Field(default=600.0)
    ws_timeout: float = Field(default=600.0)


def get_ui_settings() -> UISettings:
    return UISettings()
