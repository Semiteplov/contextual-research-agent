from dataclasses import dataclass
from pathlib import Path

import yaml

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.arxiv.downloader import FileType
from contextual_research_agent.data.extraction.pdf_extractor import (
    ExtractionMethod,
    ExtractionResult,
    PDFExtractor,
)
from contextual_research_agent.data.storage.s3_client import S3Client, S3ClientError
from contextual_research_agent.db import get_connection_context
from contextual_research_agent.db.repositories.datasets import DatasetsRepository
from contextual_research_agent.db.repositories.extracted_texts import (
    ExtractedTextsRepository,
)
from contextual_research_agent.db.repositories.paper_files import PaperFilesRepository

logger = get_logger(__name__)


S3_EXTRACTED_PREFIX = "arxiv/extracted"


@dataclass
class ExtractionProgress:
    total: int = 0
    extracted: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def processed(self) -> int:
        return self.extracted + self.skipped + self.failed

    @property
    def progress_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return self.processed / self.total * 100


class TextExtractionService:
    def __init__(
        self,
        s3_client: S3Client | None = None,
        pdf_extractor: PDFExtractor | None = None,
    ) -> None:
        self._s3_client = s3_client or S3Client()
        self._pdf_extractor = pdf_extractor or PDFExtractor()

    def extract_dataset(
        self,
        dataset_name: str,
        method: ExtractionMethod = ExtractionMethod.PYMUPDF,
        skip_existing: bool = True,
        limit: int | None = None,
    ) -> ExtractionProgress:
        with get_connection_context() as conn:
            datasets_repo = DatasetsRepository(conn)
            dataset = datasets_repo.get_by_name(dataset_name)

            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")

            extracted_repo = ExtractedTextsRepository(conn)

            if skip_existing:
                arxiv_ids = extracted_repo.get_unextracted_arxiv_ids(
                    dataset_name, method, limit or 10000
                )
            else:
                papers_repo = PaperFilesRepository(conn)
                all_ids = datasets_repo.get_arxiv_ids(dataset_name)
                downloaded = papers_repo.bulk_check_exists(all_ids, file_type=FileType.PDF)
                arxiv_ids = list(downloaded)

        if limit:
            arxiv_ids = arxiv_ids[:limit]

        if not arxiv_ids:
            logger.info("No papers to extract")
            return ExtractionProgress()

        logger.info("Extracting text from %d papers using %s", len(arxiv_ids), method.value)

        progress = ExtractionProgress(total=len(arxiv_ids))

        for arxiv_id in arxiv_ids:
            try:
                result = self._extract_paper(arxiv_id, method)

                if result.is_successful:
                    progress.extracted += 1
                else:
                    progress.failed += 1

                if progress.processed % 10 == 0:
                    logger.info(
                        "Progress: %d/%d (%.1f%%) - extracted=%d, failed=%d",
                        progress.processed,
                        progress.total,
                        progress.progress_pct,
                        progress.extracted,
                        progress.failed,
                    )

            except Exception:
                logger.exception("Unexpected error extracting %s", arxiv_id)
                progress.failed += 1

        logger.info(
            "Extraction complete: extracted=%d, skipped=%d, failed=%d",
            progress.extracted,
            progress.skipped,
            progress.failed,
        )

        return progress

    def _extract_paper(
        self,
        arxiv_id: str,
        method: ExtractionMethod,
    ) -> ExtractionResult:
        with get_connection_context() as conn:
            papers_repo = PaperFilesRepository(conn)
            storage_path = papers_repo.get_storage_path(arxiv_id)

        if storage_path is None:
            return ExtractionResult(
                arxiv_id=arxiv_id,
                text="",
                method=method,
                num_pages=0,
                num_characters=0,
                num_words=0,
                status="failed",
                error_message="PDF not found in database",
            )

        try:
            s3_key = storage_path.replace(f"s3://{self._s3_client.bucket}/", "")
            pdf_content = self._s3_client.download_bytes(s3_key)
        except S3ClientError as e:
            return ExtractionResult(
                arxiv_id=arxiv_id,
                text="",
                method=method,
                num_pages=0,
                num_characters=0,
                num_words=0,
                status="failed",
                error_message=f"Failed to download PDF: {e}",
            )

        if method == ExtractionMethod.PYMUPDF:
            result = self._pdf_extractor.extract_from_bytes(pdf_content, arxiv_id)
        else:
            raise ValueError(f"Unsupported extraction method: {method}")

        if result.is_successful and result.text:
            text_s3_key = f"{S3_EXTRACTED_PREFIX}/{method.value}/{arxiv_id}.txt"
            try:
                text_storage_path = self._s3_client.upload_bytes(
                    result.text.encode("utf-8"),
                    text_s3_key,
                    content_type="text/plain; charset=utf-8",
                )
            except S3ClientError as e:
                logger.error("Failed to upload extracted text for %s: %s", arxiv_id, e)
                text_storage_path = ""
        else:
            text_storage_path = ""

        with get_connection_context() as conn:
            extracted_repo = ExtractedTextsRepository(conn)
            extracted_repo.insert(
                arxiv_id=arxiv_id,
                extraction_method=method,
                storage_path=text_storage_path,
                num_pages=result.num_pages,
                num_characters=result.num_characters,
                num_words=result.num_words,
                status=result.status,
                error_message=result.error_message,
            )

        logger.debug(
            "Extracted %s: %d pages, %d words, status=%s",
            arxiv_id,
            result.num_pages,
            result.num_words,
            result.status,
        )

        return result

    def get_extracted_text(self, arxiv_id: str) -> str | None:
        with get_connection_context() as conn:
            extracted_repo = ExtractedTextsRepository(conn)
            record = extracted_repo.get_by_arxiv_id(arxiv_id)

        if record is None or not record.storage_path:
            return None

        try:
            s3_key = record.storage_path.replace(f"s3://{self._s3_client.bucket}/", "")
            content = self._s3_client.download_bytes(s3_key)
            return content.decode("utf-8")
        except S3ClientError:
            logger.exception("Failed to download extracted text for %s", arxiv_id)
            return None

    def get_extraction_stats(self, dataset_name: str) -> dict:
        with get_connection_context() as conn:
            extracted_repo = ExtractedTextsRepository(conn)
            return extracted_repo.get_stats(dataset_name)

    def export_stats(
        self,
        dataset_name: str,
        output_dir: str = "configs/datasets",
    ) -> Path:
        stats = self.get_extraction_stats(dataset_name)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats_file = output_path / f"{dataset_name}_extraction_stats.yaml"
        with stats_file.open("w") as f:
            yaml.dump(
                {
                    "dataset": dataset_name,
                    "extraction_stats": stats,
                },
                f,
                default_flow_style=False,
            )

        logger.info("Exported extraction stats to %s", stats_file)
        return stats_file
