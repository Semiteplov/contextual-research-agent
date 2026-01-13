import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_DIR = Path(__file__).resolve().parents[1] / "env"
DEFAULT_ENV_FILE = ENV_DIR / ".env.local"


class AppSettings(BaseSettings):
    """
    Secrets configuration.
    """

    env: Literal["dev", "prod"] = Field(default="dev")
    debug: bool = Field(default=False)

    # Postgres
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="assistant", alias="POSTGRES_DB")
    postgres_user: str = Field(default="assistant", alias="POSTGRES_USER")
    postgres_password: SecretStr = Field(..., alias="POSTGRES_PASSWORD")

    # MinIO / S3
    s3_bucket: str = Field(..., alias="S3_BUCKET")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")

    aws_access_key_id: str = Field(..., alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: SecretStr = Field(..., alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION")

    # MLflow
    mmlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="contextual-research-agent", alias="MLFLOW_EXPERIMENT_NAME"
    )
    mlflow_artifact_root: str | None = Field(default=None, alias="MLFLOW_ARTIFACT_ROOT")

    # Pydantic settings config
    model_config = SettingsConfigDict(
        env_file=(Path(os.getenv("ENV_FILE", str(DEFAULT_ENV_FILE))),),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def postgres_url(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg2://"
            f"{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_db}"
        )

    @property
    def s3_bucket_uri(self) -> str:
        return f"s3://{self.s3_bucket}"

    @property
    def effective_mlflow_artifact_root(self) -> str:
        return self.mlflow_artifact_root or f"s3://{self.s3_bucket}/mlflow"


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()  # pyright: ignore[reportCallIssue]
