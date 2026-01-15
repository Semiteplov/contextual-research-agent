from contextual_research_agent.data.arxiv.parser import PaperRow
from contextual_research_agent.data.arxiv.schemas import ParsedEntry


def entry_to_paper_row(entry: ParsedEntry, fallback_category: str) -> PaperRow:
    """Convert ParsedEntry to PaperRow for database storage."""
    return PaperRow(
        arxiv_id=entry.arxiv_id,
        title=entry.title,
        abstract=entry.abstract,
        authors=entry.authors,
        categories=entry.categories or [fallback_category],
        primary_category=entry.primary_category,
        doi=entry.doi,
        journal_ref=entry.journal_ref,
        update_date=entry.updated.date(),
        latest_version=entry.latest_version,
        latest_version_created=entry.updated,
    )


def entries_to_paper_rows(
    entries: list[ParsedEntry],
    fallback_category: str,
) -> list[PaperRow]:
    """Convert list of ParsedEntry to list of PaperRow."""
    return [entry_to_paper_row(e, fallback_category) for e in entries]
