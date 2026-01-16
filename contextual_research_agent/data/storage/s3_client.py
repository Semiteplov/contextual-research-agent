import hashlib
from io import BytesIO
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.common.settings import get_settings

logger = get_logger(__name__)


class S3ClientError(Exception):
    """S3 Client operation failed."""


class S3Client:
    def __init__(
        self,
        bucket: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        settings = get_settings()

        self.bucket = bucket or settings.s3.bucket
        self._endpoint_url = endpoint_url or settings.s3.endpoint_url

        self._client = boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=settings.s3.access_key_id,
            aws_secret_access_key=settings.s3.secret_access_key.get_secret_value(),
            region_name=settings.s3.region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        try:
            self._client.upload_fileobj(
                BytesIO(data),
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            uri = f"s3://{self.bucket}/{key}"
            logger.debug("Uploaded %d bytes to %s", len(data), uri)
            return uri

        except ClientError as e:
            logger.exception("Failed to upload to S3: %s", key)
            raise S3ClientError(f"Upload failed: {e}") from e

    def upload_file(
        self,
        file_path: Path,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        try:
            self._client.upload_file(
                str(file_path),
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            uri = f"s3://{self.bucket}/{key}"
            logger.debug("Uploaded %s to %s", file_path, uri)
            return uri

        except ClientError as e:
            logger.exception("Failed to upload file to S3: %s", key)
            raise S3ClientError(f"Upload failed: {e}") from e

    def download_bytes(self, key: str) -> bytes:
        try:
            buffer = BytesIO()
            self._client.download_fileobj(self.bucket, key, buffer)
            buffer.seek(0)
            return buffer.read()

        except ClientError as e:
            logger.exception("Failed to download from S3: %s", key)
            raise S3ClientError(f"Download failed: {e}") from e

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            logger.debug("Deleted s3://%s/%s", self.bucket, key)
        except ClientError as e:
            logger.exception("Failed to delete from S3: %s", key)
            raise S3ClientError(f"Delete failed: {e}") from e


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
