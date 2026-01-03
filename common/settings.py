from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Secrets configuration.
    """

    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # -------------------------------------------------
    # Postgres
    # -------------------------------------------------
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="assistant")
    postgres_user: str = Field(default="assistant")
    postgres_password: SecretStr

    # -------------------------------------------------
    # MinIO / S3-compatible storage
    # -------------------------------------------------
    minio_endpoint: str = Field(
        default="localhost:9000",
        description="MinIO endpoint without scheme",
    )
    minio_access_key: str
    minio_secret_key: SecretStr
    minio_bucket: str = Field(default="rag")
    minio_secure: bool = Field(
        default=False,
        description="Use HTTPS for MinIO connection",
    )

    # -------------------------------------------------
    # MLflow
    # -------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
    )
    mlflow_experiment_name: str = Field(
        default="rag-research-assistant",
    )

    # -------------------------------------------------
    # Pydantic settings config
    # -------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def postgres_url(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg2://"
            f"{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()
