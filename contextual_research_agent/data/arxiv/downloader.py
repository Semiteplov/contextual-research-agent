import gzip
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from ssl import SSLError

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.storage.s3_client import compute_sha256

logger = get_logger(__name__)


class FileType(Enum):
    PDF = "pdf"
    SRC = "source"


class SourceFormat(Enum):
    GZ = "gz"
    TAR_GZ = "tar.gz"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DownloadedFile:
    arxiv_id: str
    file_type: FileType
    content: bytes
    size_bytes: int
    checksum_sha256: str
    source_format: SourceFormat | None = None

    @property
    def extension(self) -> str:
        if self.file_type == FileType.PDF:
            return "pdf"
        if self.source_format == SourceFormat.TAR_GZ:
            return "tar.gz"
        if self.source_format == SourceFormat.GZ:
            return "gz"
        return "gz"

    @property
    def content_type(self) -> str:
        if self.file_type == FileType.PDF:
            return "application/pdf"
        return "application/gzip"


class ArxivDownloadError(Exception):
    """Failed to download from arXiv."""


class ArxivDownloaderConfig:
    DELAY_SECONDS: float = 3.0
    MAX_RETRIES: int = 5
    TIMEOUT_SECONDS: int = 60
    BACKOFF_BASE_SECONDS: float = 2.0
    BACKOFF_MAX_SECONDS: float = 60.0

    PDF_URL_TEMPLATE: str = "https://arxiv.org/pdf/{arxiv_id}.pdf"
    SOURCE_URL_TEMPLATE: str = "https://arxiv.org/e-print/{arxiv_id}"


class ArxivDownloader:
    def __init__(self, config: ArxivDownloaderConfig | None = None) -> None:
        self.config = config or ArxivDownloaderConfig()
        self._last_request_time: float = 0

    def download_pdf(self, arxiv_id: str) -> DownloadedFile:
        url = self.config.PDF_URL_TEMPLATE.format(arxiv_id=arxiv_id)
        content = self._download_with_retry(url, arxiv_id)

        return DownloadedFile(
            arxiv_id=arxiv_id,
            file_type=FileType.PDF,
            content=content,
            size_bytes=len(content),
            checksum_sha256=compute_sha256(content),
            source_format=None,
        )

    def download_source(self, arxiv_id: str) -> DownloadedFile:
        url = self.config.SOURCE_URL_TEMPLATE.format(arxiv_id=arxiv_id)
        content = self._download_with_retry(url, arxiv_id)
        source_format = self._detect_source_format(content)

        return DownloadedFile(
            arxiv_id=arxiv_id,
            file_type=FileType.SRC,
            content=content,
            size_bytes=len(content),
            checksum_sha256=compute_sha256(content),
            source_format=source_format,
        )

    def download(
        self,
        arxiv_id: str,
        file_type: FileType = FileType.PDF,
    ) -> DownloadedFile:
        if file_type == FileType.PDF:
            return self.download_pdf(arxiv_id)
        return self.download_source(arxiv_id)

    def _download_with_retry(self, url: str, arxiv_id: str) -> bytes:
        self._respect_rate_limit()

        last_error: Exception | None = None

        for attempt in range(1, self.config.MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url)

                with urllib.request.urlopen(req, timeout=self.config.TIMEOUT_SECONDS) as response:
                    content = response.read()

                logger.debug(
                    "Downloaded %s (%d bytes)",
                    arxiv_id,
                    len(content),
                )
                return content

            except urllib.error.HTTPError as e:
                if e.code == 404:  # noqa: PLR2004
                    raise ArxivDownloadError(f"Paper not found: {arxiv_id}") from e

                last_error = e
                logger.warning(
                    "HTTP %d for %s (attempt %d/%d)",
                    e.code,
                    arxiv_id,
                    attempt,
                    self.config.MAX_RETRIES,
                )

            except (
                TimeoutError,
                SSLError,
                urllib.error.URLError,
                ConnectionError,
            ) as e:
                last_error = e
                logger.warning(
                    "Download failed for %s (attempt %d/%d): %r",
                    arxiv_id,
                    attempt,
                    self.config.MAX_RETRIES,
                    e,
                )

            if attempt < self.config.MAX_RETRIES:
                sleep_seconds = min(
                    self.config.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                    self.config.BACKOFF_MAX_SECONDS,
                )
                sleep_seconds *= 0.7 + random.random() * 0.6
                time.sleep(sleep_seconds)

        raise ArxivDownloadError(
            f"Download failed after {self.config.MAX_RETRIES} attempts: {arxiv_id}"
        ) from last_error

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.DELAY_SECONDS:
            sleep_time = self.config.DELAY_SECONDS - elapsed
            logger.debug("Rate limiting: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _detect_source_format(self, content: bytes) -> SourceFormat:
        if len(content) < 2:  # noqa: PLR2004
            return SourceFormat.UNKNOWN

        if content[:2] != b"\x1f\x8b":
            return SourceFormat.UNKNOWN

        try:
            decompressed = gzip.decompress(content)

            if len(decompressed) > 262 and decompressed[257:262] == b"ustar":  # noqa: PLR2004
                return SourceFormat.TAR_GZ
            return SourceFormat.GZ
        except Exception:
            return SourceFormat.UNKNOWN
