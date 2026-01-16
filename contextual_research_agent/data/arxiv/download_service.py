from dataclasses import dataclass

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.arxiv.downloader import (
    ArxivDownloader,
    ArxivDownloadError,
    DownloadedFile,
    FileType,
)
from contextual_research_agent.data.storage.s3_client import S3Client, S3ClientError
from contextual_research_agent.db import get_connection_context
from contextual_research_agent.db.repositories.paper_files import PaperFilesRepository

logger = get_logger(__name__)


@dataclass
class DownloadState:
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


class PaperDownloadService:
    S3_PREFIX = "arxiv/papers"

    def __init__(
        self,
        downloader: ArxivDownloader | None = None,
        s3_client: S3Client | None = None,
    ) -> None:
        self._downloader = downloader or ArxivDownloader()
        self._s3_client = s3_client or S3Client()

    def download_and_store(
        self,
        arxiv_id: str,
        file_type: FileType = FileType.PDF,
        skip_existing: bool = True,
    ) -> bool:
        with get_connection_context() as conn:
            repo = PaperFilesRepository(conn)

            if skip_existing and repo.exists(arxiv_id, file_type):
                logger.debug("Skipping %s (already exists)", arxiv_id)
                return False

            try:
                downloaded = self._downloader.download(arxiv_id, file_type)

                s3_key = self._make_s3_key(downloaded)
                storage_path = self._s3_client.upload_bytes(
                    downloaded.content,
                    s3_key,
                    content_type=self._get_content_type(file_type),
                )

                repo.insert(
                    arxiv_id=arxiv_id,
                    storage_path=storage_path,
                    file_type=file_type,
                    file_size_bytes=downloaded.size_bytes,
                    checksum_sha256=downloaded.checksum_sha256,
                )

                logger.info(
                    "Downloaded %s (%s, %d KB)",
                    arxiv_id,
                    file_type.value,
                    downloaded.size_bytes // 1024,
                )
                return True

            except ArxivDownloadError as e:
                logger.warning("Failed to download %s: %s", arxiv_id, e)
                return False

            except S3ClientError as e:
                logger.error("Failed to upload %s to S3: %s", arxiv_id, e)
                return False

    def download_batch(
        self,
        arxiv_ids: list[str],
        file_type: FileType = FileType.PDF,
        skip_existing: bool = True,
    ) -> DownloadState:
        stats = DownloadState(total=len(arxiv_ids))

        for arxiv_id in arxiv_ids:
            try:
                result = self.download_and_store(arxiv_id, file_type, skip_existing)
                if result:
                    stats.downloaded += 1
                else:
                    stats.skipped += 1

            except Exception:
                logger.exception("Unexpected error downloading %s", arxiv_id)
                stats.failed += 1

        logger.info(
            "Batch complete: %d downloaded, %d skipped, %d failed",
            stats.downloaded,
            stats.skipped,
            stats.failed,
        )
        return stats

    def download_missing(
        self,
        file_type: FileType = FileType.PDF,
        limit: int = 100,
    ) -> DownloadState:
        with get_connection_context() as conn:
            repo = PaperFilesRepository(conn)
            missing_ids = repo.get_missing_arxiv_ids(file_type, limit)

        logger.info("Found %d papers to download", len(missing_ids))
        return self.download_batch(missing_ids, file_type, skip_existing=False)

    def _make_s3_key(self, downloaded: DownloadedFile) -> str:
        arxiv_id = downloaded.arxiv_id
        extension = "pdf" if downloaded.file_type == FileType.PDF else "tar.gz"

        return f"{self.S3_PREFIX}/{downloaded.file_type.value}/{arxiv_id}.{extension}"

    @staticmethod
    def _get_content_type(file_type: FileType) -> str:
        if file_type == FileType.PDF:
            return "application/pdf"
        return "application/gzip"
