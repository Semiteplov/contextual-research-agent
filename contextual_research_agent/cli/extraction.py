from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.extraction.pdf_extractor import ExtractionMethod
from contextual_research_agent.services.text_extraction_service import (
    TextExtractionService,
)

logger = get_logger(__name__)


def extract_texts(
    dataset: str,
    method: str = "pymupdf",
    skip_existing: bool = True,
    limit: int | None = None,
) -> None:
    """
    Extract text from PDF papers in a dataset.

    Args:
        dataset: Dataset name.
        method: Extraction method ('pymupdf', 'nougat', 'latex').
        skip_existing: Skip already extracted papers.
        limit: Maximum number of papers to extract.

    Example:
        python main.py extract-texts --dataset="mvp-v1"
        python main.py extract-texts --dataset="mvp-v1" --limit=100
    """
    extraction_method = ExtractionMethod(method)
    service = TextExtractionService()

    progress = service.extract_dataset(
        dataset_name=dataset,
        method=extraction_method,
        skip_existing=skip_existing,
        limit=limit,
    )

    print("\nExtraction complete:")
    print(f"  Total:     {progress.total}")
    print(f"  Extracted: {progress.extracted}")
    print(f"  Skipped:   {progress.skipped}")
    print(f"  Failed:    {progress.failed}")


def extraction_stats(dataset: str, export: bool = False) -> None:
    """
    Show extraction statistics for a dataset.

    Args:
        dataset: Dataset name.
        export: Export stats to YAML file.

    Example:
        python main.py extraction-stats --dataset="mvp-v1"
    """
    service = TextExtractionService()
    stats = service.get_extraction_stats(dataset)

    print(f"\nExtraction statistics for '{dataset}':")
    print("=" * 50)
    print(f"  Total in dataset:  {stats['total_in_dataset']}")
    print(f"  Downloaded:        {stats['total_downloaded']}")
    print(f"  Extracted:         {stats['total_extracted']}")
    print(f"    - Completed:     {stats['completed']}")
    print(f"    - Partial:       {stats['partial']}")
    print(f"    - Failed:        {stats['failed']}")
    print(f"  Avg pages:         {stats['avg_pages']}")
    print(f"  Avg words:         {stats['avg_words']}")
    print("=" * 50)

    if export:
        stats_file = service.export_stats(dataset)
        print(f"\nExported to: {stats_file}")


def show_extracted_text(arxiv_id: str, max_length: int = 2000) -> None:
    """
    Show extracted text for a paper.

    Args:
        arxiv_id: Paper identifier.
        max_length: Maximum characters to display.

    Example:
        python main.py show-text --arxiv-id="2401.12345"
    """
    service = TextExtractionService()
    text = service.get_extracted_text(arxiv_id)

    if text is None:
        print(f"No extracted text found for {arxiv_id}")
        return

    print(f"\nExtracted text for {arxiv_id}:")
    print("=" * 50)

    if len(text) > max_length:
        print(text[:max_length])
        print(f"\n... ({len(text) - max_length} more characters)")
    else:
        print(text)

    print("=" * 50)
    print(f"Total: {len(text)} characters, {len(text.split())} words")
