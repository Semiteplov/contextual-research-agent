from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / "env" / ".env.local"


def _make_settings_config(env_prefix: str = "") -> SettingsConfigDict:
    """Factory for creating SettingsConfigDict."""
    return SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix=env_prefix,
    )


class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = _make_settings_config("POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    db: str = "assistant"
    user: str = "assistant"
    password: SecretStr

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class S3Settings(BaseSettings):
    """S3/MinIO storage settings."""

    model_config = _make_settings_config("S3_")

    bucket: str = ""
    endpoint_url: str | None = None
    access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    secret_access_key: SecretStr = Field(default=SecretStr(""), alias="AWS_SECRET_ACCESS_KEY")
    region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION")

    @property
    def uri(self) -> str:
        """S3 bucket URI."""
        return f"s3://{self.bucket}"


class MLflowSettings(BaseSettings):
    """MLflow tracking settings."""

    model_config = _make_settings_config("MLFLOW_")

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "contextual-research-agent"
    artifact_root: str | None = None


class QdrantSettings(BaseSettings):
    """Qdrant connection settings."""

    model_config = _make_settings_config("QDRANT_")

    host: str = "localhost"
    port: int = 6333
    collection_name: str = "documents"


class AppSettings(BaseSettings):
    """Application settings."""

    model_config = _make_settings_config()

    env: Literal["dev", "prod"] = "dev"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    @cached_property
    def postgres(self) -> PostgresSettings:
        return PostgresSettings()  # type: ignore[call-arg]

    @cached_property
    def s3(self) -> S3Settings:
        return S3Settings()

    @cached_property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings()

    @cached_property
    def mlflow(self) -> MLflowSettings:
        mlflow = MLflowSettings()
        if mlflow.artifact_root is None and self.s3.bucket:
            object.__setattr__(mlflow, "artifact_root", f"{self.s3.uri}/mlflow")
        return mlflow

    @property
    def is_production(self) -> bool:
        return self.env == "prod"


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()  # pyright: ignore[reportCallIssue]
