from dataclasses import dataclass
from pathlib import Path

import yaml

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.arxiv.downloader import (
    ArxivDownloader,
    ArxivDownloadError,
    FileType,
)
from contextual_research_agent.data.storage.s3_client import S3Client, S3ClientError
from contextual_research_agent.db import get_connection_context
from contextual_research_agent.db.repositories.datasets import (
    DatasetRecord,
    DatasetsRepository,
    DatasetStats,
)
from contextual_research_agent.db.repositories.paper_files import PaperFilesRepository

logger = get_logger(__name__)


S3_PAPERS_PREFIX = "arxiv/papers"


@dataclass
class DownloadProgress:
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def processed(self) -> int:
        return self.downloaded + self.skipped + self.failed

    @property
    def progress_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return self.processed / self.total * 100


@dataclass
class DatasetConfig:
    name: str
    description: str | None
    version: int
    purpose: str
    created_at: str

    categories: list[str]
    min_date: str
    keywords: list[str] | None
    total_requested: int

    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int

    total_papers: int
    downloaded_papers: int
    train_count: int
    val_count: int
    test_count: int

    s3_bucket: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "purpose": self.purpose,
            "created_at": self.created_at,
            "selection_criteria": {
                "categories": self.categories,
                "min_date": self.min_date,
                "keywords": self.keywords,
                "total_requested": self.total_requested,
            },
            "splits": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "random_seed": self.random_seed,
            },
            "statistics": {
                "total_papers": self.total_papers,
                "downloaded_papers": self.downloaded_papers,
                "train_count": self.train_count,
                "val_count": self.val_count,
                "test_count": self.test_count,
            },
            "storage": {
                "s3_bucket": self.s3_bucket,
                "s3_prefix": S3_PAPERS_PREFIX,
            },
        }


class DatasetService:
    def __init__(
        self,
        downloader: ArxivDownloader | None = None,
        s3_client: S3Client | None = None,
    ) -> None:
        self._downloader = downloader or ArxivDownloader()
        self._s3_client = s3_client or S3Client()

    def create_dataset(  # noqa: PLR0913
        self,
        name: str,
        categories: list[str],
        min_date: str,
        total: int,
        keywords: list[str] | None = None,
        description: str | None = None,
        purpose: str = "training",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        download_pdfs: bool = True,
        config_output_dir: str = "configs/datasets",
        overwrite: bool = False,
    ) -> tuple[DatasetRecord, DatasetStats, Path]:
        limit_per_category = max(1, total // len(categories))

        selection_criteria = {
            "categories": categories,
            "min_date": min_date,
            "keywords": keywords,
            "limit_per_category": limit_per_category,
            "total_requested": total,
        }

        logger.info("Creating dataset '%s'", name)
        logger.info("  Categories: %s", categories)
        logger.info("  Min date: %s", min_date)
        logger.info("  Keywords: %s", keywords or "none")
        logger.info(
            "  Splits: train=%.0f%%, val=%.0f%%, test=%.0f%%",
            train_ratio * 100,
            val_ratio * 100,
            test_ratio * 100,
        )

        with get_connection_context() as conn:
            papers_repo = PaperFilesRepository(conn)
            arxiv_ids = papers_repo.get_papers_for_download(
                limit_per_category=limit_per_category,
                categories=categories,
                min_date=min_date,
                keywords=keywords,
                exclude_downloaded=False,
            )

            if not arxiv_ids:
                raise ValueError("No papers match the selection criteria")

            logger.info("Selected %d papers", len(arxiv_ids))

            datasets_repo = DatasetsRepository(conn)
            datasets_repo.create(
                name=name,
                arxiv_ids=arxiv_ids,
                selection_criteria=selection_criteria,
                description=description,
                purpose=purpose,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
                overwrite=overwrite,
            )

        if download_pdfs:
            logger.info("Starting PDF download...")
            self._download_dataset_pdfs(name)

        with get_connection_context() as conn:
            datasets_repo = DatasetsRepository(conn)
            dataset = datasets_repo.get_by_name(name)
            stats = datasets_repo.get_stats(name)

        if dataset is None:
            raise ValueError(f"Failed to retrieve created dataset '{name}'")

        if stats is None:
            raise ValueError(f"Failed to retrieve stats for dataset '{name}'")

        config_path = self._export_config(
            name=name,
            output_dir=config_output_dir,
        )

        return dataset, stats, config_path

    def _export_config(self, name: str, output_dir: str) -> Path:
        with get_connection_context() as conn:
            datasets_repo = DatasetsRepository(conn)
            dataset = datasets_repo.get_by_name(name)
            stats = datasets_repo.get_stats(name)

        if dataset is None or stats is None:
            raise ValueError(f"Dataset '{name}' not found")

        config = DatasetConfig(
            name=dataset.name,
            description=dataset.description,
            version=dataset.version,
            purpose=dataset.purpose,
            created_at=dataset.created_at.isoformat(),
            categories=dataset.selection_criteria.get("categories", []),
            min_date=dataset.selection_criteria.get("min_date", ""),
            keywords=dataset.selection_criteria.get("keywords"),
            total_requested=dataset.selection_criteria.get("total_requested", 0),
            train_ratio=dataset.train_ratio,
            val_ratio=dataset.val_ratio,
            test_ratio=dataset.test_ratio,
            random_seed=dataset.random_seed,
            total_papers=stats.total,
            downloaded_papers=stats.downloaded,
            train_count=stats.train,
            val_count=stats.val,
            test_count=stats.test,
            s3_bucket=self._s3_client.bucket,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        config_file = output_path / f"{name}.yaml"
        with config_file.open("w") as f:
            yaml.dump(
                config.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info("Exported config to %s", config_file)
        return config_file

    def get_download_urls(
        self,
        dataset_name: str,
        split: str | None = None,
        expires_in: int = 3600,
    ) -> dict[str, str]:
        with get_connection_context() as conn:
            datasets_repo = DatasetsRepository(conn)
            papers = datasets_repo.get_papers_with_paths(
                dataset_name=dataset_name,
                split=split,
                only_downloaded=True,
            )

        urls = {}
        for paper in papers:
            if paper.storage_path:
                key = paper.storage_path.replace(f"s3://{self._s3_client.bucket}/", "")
                urls[paper.arxiv_id] = self._s3_client.get_presigned_url(key, expires_in)

        return urls

    def resume_download(
        self, dataset_name: str, config_output_dir: str = "configs/datasets"
    ) -> DownloadProgress:
        with get_connection_context() as conn:
            datasets_repo = DatasetsRepository(conn)

            dataset = datasets_repo.get_by_name(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")

            arxiv_ids = datasets_repo.get_undownloaded_arxiv_ids(dataset_name)

        if not arxiv_ids:
            logger.info("All papers already downloaded")
            self._export_config(dataset_name, config_output_dir)
            return DownloadProgress()

        logger.info("Resuming download: %d papers remaining", len(arxiv_ids))

        progress = self._download_papers(arxiv_ids)
        self._export_config(dataset_name, config_output_dir)

        return progress

    def verify_and_clean(self, dataset_name: str) -> int:
        with get_connection_context() as conn:
            datasets_repo = DatasetsRepository(conn)
            papers = datasets_repo.get_papers_with_paths(
                dataset_name=dataset_name,
                only_downloaded=True,
            )

        orphaned_ids = []

        for paper in papers:
            if paper.storage_path:
                key = paper.storage_path.replace(f"s3://{self._s3_client.bucket}/", "")
                if not self._s3_client.exists(key):
                    logger.warning("S3 file missing for %s: %s", paper.arxiv_id, key)
                    orphaned_ids.append(paper.arxiv_id)

        if orphaned_ids:
            with get_connection_context() as conn:
                papers_repo = PaperFilesRepository(conn)
                papers_repo.delete_by_arxiv_ids(orphaned_ids)
            logger.info("Cleaned %d orphaned records", len(orphaned_ids))

        return len(orphaned_ids)

    def _download_papers(self, arxiv_ids: list[str]) -> DownloadProgress:
        progress = DownloadProgress(total=len(arxiv_ids))

        for arxiv_id in arxiv_ids:
            try:
                downloaded = self._downloader.download(arxiv_id, FileType.PDF)

                s3_key = f"{S3_PAPERS_PREFIX}/pdf/{downloaded.arxiv_id}.pdf"
                storage_path = self._s3_client.upload_bytes(
                    downloaded.content,
                    s3_key,
                    content_type="application/pdf",
                )

                with get_connection_context() as conn:
                    papers_repo = PaperFilesRepository(conn)
                    papers_repo.insert(
                        arxiv_id=downloaded.arxiv_id,
                        storage_path=storage_path,
                        file_type=FileType.PDF,
                        file_size_bytes=downloaded.size_bytes,
                        checksum_sha256=downloaded.checksum_sha256,
                    )

                progress.downloaded += 1

                logger.info(
                    "Downloaded %s (%d KB) [%d/%d %.1f%%]",
                    arxiv_id,
                    downloaded.size_bytes // 1024,
                    progress.processed,
                    progress.total,
                    progress.progress_pct,
                )

            except ArxivDownloadError as e:
                logger.warning("Failed to download %s: %s", arxiv_id, e)
                progress.failed += 1

            except S3ClientError as e:
                logger.error("Failed to upload %s to S3: %s", arxiv_id, e)
                progress.failed += 1

            except Exception:
                logger.exception("Unexpected error for %s", arxiv_id)
                progress.failed += 1

        logger.info(
            "Download complete: downloaded=%d, skipped=%d, failed=%d",
            progress.downloaded,
            progress.skipped,
            progress.failed,
        )

        return progress

    def _download_dataset_pdfs(self, dataset_name: str) -> DownloadProgress:
        with get_connection_context() as conn:
            datasets_repo = DatasetsRepository(conn)
            arxiv_ids = datasets_repo.get_undownloaded_arxiv_ids(dataset_name)

        if not arxiv_ids:
            logger.info("All papers already downloaded")
            return DownloadProgress()

        logger.info("Downloading %d PDFs...", len(arxiv_ids))

        return self._download_papers(arxiv_ids)
