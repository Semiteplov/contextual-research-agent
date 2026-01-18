from pathlib import Path

import requests

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.arxiv.downloader import FileType
from contextual_research_agent.db.connection import get_connection_context
from contextual_research_agent.db.repositories.datasets import DatasetsRepository
from contextual_research_agent.services.dataset_service import DatasetService

logger = get_logger(__name__)


def create_dataset(  # noqa: PLR0913
    name: str,
    total: int = 1000,
    categories: str = "cs.CL,cs.LG,cs.AI,cs.CV,stat.ML",
    min_date: str = "2022-01-01",
    keywords: str | None = None,
    purpose: str = "training",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    description: str | None = None,
    no_download: bool = False,
    config_dir: str = "configs/datasets",
    overwrite: bool = False,
) -> None:
    """
    Args:
        name: Unique dataset name (e.g., 'mvp-v1').
        total: Total papers to include.
        categories: Comma-separated arXiv categories.
        min_date: Minimum paper date (YYYY-MM-DD).
        keywords: Comma-separated keywords to filter by title/abstract.
        purpose: Dataset purpose (training, validation, test, exploration).
        train_ratio: Training split ratio (default 0.8).
        val_ratio: Validation split ratio (default 0.1).
        test_ratio: Test split ratio (default 0.1).
        random_seed: Random seed for reproducible splits.
        description: Human-readable description.
        no_download: If True, skip PDF download.
        config_dir: Directory for config YAML files.

    Example:
        python main.py create-dataset \\
            --name="mvp-v1" \\
            --total=1000 \\
            --categories="cs.CL,cs.LG" \\
            --min-date="2023-01-01" \\
            --keywords="language model,transformer,retrieval"
    """
    cat_list = [c.strip() for c in categories.split(",")]
    kw_list = [k.strip() for k in keywords.split(",")] if keywords else None

    service = DatasetService()

    dataset, stats, config_path = service.create_dataset(
        name=name,
        categories=cat_list,
        min_date=min_date,
        total=total,
        keywords=kw_list,
        description=description or f"Dataset: {name}",
        purpose=purpose,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        download_pdfs=not no_download,
        config_output_dir=config_dir,
        overwrite=overwrite,
    )

    if dataset is None:
        print(f"Dataset '{name}' not found")
        return

    if stats is None:
        print(f"Stats for dataset '{name}' not found")
        return

    print("\n" + "=" * 60)
    print(f"Dataset '{name}' created successfully!")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Categories:    {cat_list}")
    print(f"  Min date:      {min_date}")
    print(f"  Keywords:      {kw_list or 'none'}")
    print(f"  Random seed:   {random_seed}")
    print("\nSplit ratios:")
    print(f"  Train:         {train_ratio * 100:.0f}%")
    print(f"  Validation:    {val_ratio * 100:.0f}%")
    print(f"  Test:          {test_ratio * 100:.0f}%")
    print("\nStatistics:")
    print(f"  Total papers:  {stats.total}")
    print(f"  Train:         {stats.train}")
    print(f"  Validation:    {stats.val}")
    print(f"  Test:          {stats.test}")
    print(f"  Downloaded:    {stats.downloaded}")
    print("\nStorage:")
    print(f"  Config file:   {config_path}")
    print("=" * 60)


def list_datasets(purpose: str | None = None) -> None:
    with get_connection_context() as conn:
        repo = DatasetsRepository(conn)
        datasets = repo.list_datasets(purpose)

        if not datasets:
            print("No datasets found")
            return

        print("\nAvailable datasets:")
        print("-" * 90)
        print(
            f"{'Name':<20} {'Purpose':<12} {'Total':<8} "
            f"{'Downloaded':<12} {'Train/Val/Test':<18} Created"
        )
        print("-" * 90)

        for ds in datasets:
            stats = repo.get_stats(ds.name)
            downloaded = stats.downloaded if stats else 0
            split_str = f"{stats.train}/{stats.val}/{stats.test}" if stats else "0/0/0"
            print(
                f"{ds.name:<20} {ds.purpose:<12} {ds.total_papers:<8} "
                f"{downloaded:<12} {split_str:<18} {ds.created_at.strftime('%Y-%m-%d %H:%M')}"
            )

        print("-" * 90)


def show_dataset(name: str) -> None:
    with get_connection_context() as conn:
        repo = DatasetsRepository(conn)
        dataset = repo.get_by_name(name)

        if dataset is None:
            print(f"Dataset '{name}' not found")
            return

        stats = repo.get_stats(name)
        if stats is None:
            print(f"Stats for dataset '{name}' not found")
            return

        print(f"\nDataset: {dataset.name}")
        print("=" * 50)
        print(f"Description:     {dataset.description}")
        print(f"Purpose:         {dataset.purpose}")
        print(f"Version:         {dataset.version}")
        print(f"Created:         {dataset.created_at}")
        print("\nSelection criteria:")
        for key, value in dataset.selection_criteria.items():
            print(f"  {key}: {value}")
        print("\nSplit configuration:")
        print(f"  Train ratio:   {dataset.train_ratio * 100:.0f}%")
        print(f"  Val ratio:     {dataset.val_ratio * 100:.0f}%")
        print(f"  Test ratio:    {dataset.test_ratio * 100:.0f}%")
        print(f"  Random seed:   {dataset.random_seed}")
        print("\nStatistics:")
        print(f"  Total papers:  {stats.total}")
        print(f"  Train:         {stats.train}")
        print(f"  Validation:    {stats.val}")
        print(f"  Test:          {stats.test}")
        download_pct = stats.downloaded / stats.total * 100 if stats.total > 0 else 0
        print(f"  Downloaded:    {stats.downloaded} ({download_pct:.1f}%)")
        print("=" * 50)


def download_dataset(
    name: str,
    output_dir: str = "data/datasets",
    split: str | None = None,
) -> None:
    service = DatasetService()
    urls = service.get_download_urls(name, split)

    if not urls:
        print("No files to download")
        return

    print(f"Downloading {len(urls)} files...")

    output_path = Path(output_dir) / name
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0

    for arxiv_id, url in urls.items():
        file_path = output_path / f"{arxiv_id}.pdf"
        print(f"Downloading {arxiv_id}...", end=" ", flush=True)

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            file_path.write_bytes(response.content)
            print(f"OK ({len(response.content) // 1024} KB)")
            downloaded += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDownloaded {downloaded} files to: {output_path}")
    if failed:
        print(f"Failed: {failed}")


def export_dataset_config(
    name: str,
    output_dir: str = "configs/datasets",
) -> None:
    service = DatasetService()
    config_path = service._export_config(name, output_dir)
    print(f"Exported config to: {config_path}")


def resume_download(
    name: str,
    verify_s3: bool = False,
    config_dir: str = "configs/datasets",
) -> None:
    """
    Resume downloading PDFs for a dataset.

    Args:
        name: Dataset name.
        verify_s3: If True, also verify that files exist in S3 (slower).
        config_dir: Directory for config YAML files.

    Example:
        python main.py resume-download --name="mvp-v1"
        python main.py resume-download --name="mvp-v1" --verify-s3
    """
    service = DatasetService()

    if verify_s3:
        logger.info("Verifying S3 and cleaning orphaned DB records...")
        cleaned = service.verify_and_clean(name)
        if cleaned > 0:
            logger.info("Removed %d orphaned records from arxiv_papers", cleaned)

    stats = service.resume_download(name, config_output_dir=config_dir)

    print("\nResume complete:")
    print(f"  Downloaded: {stats.downloaded}")
    print(f"  Skipped:    {stats.skipped}")
    print(f"  Failed:     {stats.failed}")


def download_sources(
    name: str,
    config_dir: str = "configs/datasets",
) -> None:
    """
    Download source files (LaTeX) for a dataset.

    Args:
        name: Dataset name.
        config_dir: Directory for config YAML files.

    Example:
        python main.py download-sources --name="mvp-v1"
    """
    service = DatasetService()

    logger.info("Downloading source files for dataset '%s'", name)
    stats = service.download_dataset_files(name, file_type=FileType.SRC)

    service._export_config(name, config_dir)

    print("\nSource download complete:")
    print(f"  Downloaded: {stats.downloaded}")
    print(f"  Skipped:    {stats.skipped}")
    print(f"  Failed:     {stats.failed}")
