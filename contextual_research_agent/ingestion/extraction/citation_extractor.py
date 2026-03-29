from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.ingestion.domain.entities import Chunk

logger = get_logger(__name__)


@dataclass
class CitationEdge:
    """
    A directed edge: citing_paper → cited_paper.

    Stored in citation_edges table.
    """

    citing_paper_id: str  # arxiv_id of the paper containing the reference
    cited_paper_id: str  # arxiv_id or DOI of the cited paper
    cited_id_type: str = "arxiv"  # "arxiv" | "doi" | "unknown"

    context: str = ""  # sentence containing the citation
    section: str = ""  # section heading where citation appears
    section_type: str = ""  # classified section type (if available)

    ref_key: str = ""  # e.g., "ref_12", "[12]"
    cited_title: str = ""  # title from the bibliography
    cited_authors: str = ""  # authors string from the bibliography
    cited_year: str = ""  # publication year


@dataclass
class BibEntry:
    """Parsed bibliography entry from Docling."""

    key: str
    title: str = ""
    authors: str = ""
    year: str = ""
    arxiv_id: str | None = None
    doi: str | None = None
    venue: str = ""
    raw_text: str = ""


@dataclass
class CitationExtractionResult:
    """Result of citation extraction for a single document."""

    document_id: str
    edges: list[CitationEdge] = field(default_factory=list)
    bib_entries: list[BibEntry] = field(default_factory=list)
    unresolved_count: int = 0
    total_references: int = 0

    @property
    def resolved_count(self) -> int:
        return len(self.edges)

    @property
    def resolution_rate(self) -> float:
        return self.resolved_count / self.total_references if self.total_references else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "total_references": self.total_references,
            "resolved": self.resolved_count,
            "unresolved": self.unresolved_count,
            "resolution_rate": round(self.resolution_rate, 3),
        }


# Matches: 2106.09685, 2106.09685v1, 2106.09685v2
_ARXIV_ID_PATTERN = re.compile(
    r"(?:arXiv[:\s]*)?(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)

# Matches: arxiv.org/abs/2106.09685, arxiv.org/pdf/2106.09685
_ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)

# Matches old-style: cs/0601078, math.AG/0601001
_ARXIV_OLD_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([\w\-]+/\d{7})",
    re.IGNORECASE,
)

# DOI pattern: 10.XXXX/...
_DOI_PATTERN = re.compile(
    r"(10\.\d{4,9}/[^\s,;}\]]+)",
)

# Inline citation anchors: [1], [12], [1, 3], [1-3], [12, 15, 18]
_CITATION_ANCHOR_RE = re.compile(r"\[(\d+(?:\s*[-,–]\s*\d+)*)\]")  # noqa: RUF001

# Sentence boundary (approximate)
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+")


def _normalize_arxiv_id(raw_id: str) -> str:
    """Strip version suffix: 2106.09685v2 → 2106.09685."""
    return re.sub(r"v\d+$", "", raw_id.strip())


def _extract_arxiv_id(text: str) -> str | None:
    """Try to extract arxiv ID from a text string (URL, raw text, etc.)."""
    match = _ARXIV_URL_PATTERN.search(text)
    if match:
        return _normalize_arxiv_id(match.group(1))

    match = _ARXIV_OLD_PATTERN.search(text)
    if match:
        return match.group(1)

    match = _ARXIV_ID_PATTERN.search(text)
    if match:
        return _normalize_arxiv_id(match.group(1))

    return None


def _extract_doi(text: str) -> str | None:
    """Try to extract DOI from text."""
    match = _DOI_PATTERN.search(text)
    return match.group(1).rstrip(".") if match else None


def _extract_sentence_around(text: str, position: int, max_chars: int = 500) -> str:
    """
    Extract the sentence containing the given position in text.

    Falls back to a window of max_chars if sentence boundaries aren't found.
    """
    search_start = max(0, position - max_chars)
    prefix = text[search_start:position]
    boundaries = list(_SENTENCE_BOUNDARY_RE.finditer(prefix))
    sent_start = search_start + boundaries[-1].end() if boundaries else search_start

    search_end = min(len(text), position + max_chars)
    suffix = text[position:search_end]
    boundary = _SENTENCE_BOUNDARY_RE.search(suffix)
    sent_end = position + boundary.end() if boundary else search_end

    return text[sent_start:sent_end].strip()


def _find_citation_contexts(
    chunks: list[Chunk],
    ref_key_to_numbers: dict[str, list[str]],
) -> dict[str, list[tuple[str, str, str]]]:
    """
    Find inline citation contexts in chunks.

    Args:
        chunks: Document chunks with text and section info.
        ref_key_to_numbers: Mapping from bib key → list of anchor numbers ["12", "15"].

    Returns:
        Dict: ref_key → [(context_sentence, section, section_type), ...]
    """
    number_to_key: dict[str, str] = {}
    for key, numbers in ref_key_to_numbers.items():
        for num in numbers:
            number_to_key[num] = key

    contexts: dict[str, list[tuple[str, str, str]]] = {}

    for chunk in chunks:
        text = chunk.text
        section = chunk.section
        section_type = chunk.metadata.get("section_type", "")

        for match in _CITATION_ANCHOR_RE.finditer(text):
            anchor_text = match.group(1)
            numbers = _parse_anchor_numbers(anchor_text)

            for num in numbers:
                ref_key = number_to_key.get(num)
                if ref_key is None:
                    continue

                sentence = _extract_sentence_around(text, match.start())
                if ref_key not in contexts:
                    contexts[ref_key] = []
                contexts[ref_key].append((sentence, section, section_type))

    return contexts


def _parse_anchor_numbers(anchor: str) -> list[str]:
    """
    Parse citation anchor text into individual reference numbers.

    "12" → ["12"]
    "1, 3, 5" → ["1", "3", "5"]
    "1-3" → ["1", "2", "3"]
    """
    numbers: list[str] = []
    parts = re.split(r"[,;]\s*", anchor)
    for part in parts:
        part_stripped = part.strip()
        range_match = re.match(r"(\d+)\s*[-–]\s*(\d+)", part_stripped)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            numbers.extend(str(i) for i in range(start, end + 1))
        elif part_stripped.isdigit():
            numbers.append(part_stripped)
    return numbers


def _parse_bib_entries(docling_doc) -> list[BibEntry]:
    """
    Extract bibliography entries from DoclingDocument.

    Strategy:
      1. Try docling_doc.bib_entries (structured, preferred)
      2. Try docling_doc.references (some Docling versions)
      3. Fallback: parse References section from exported markdown
    """
    entries: list[BibEntry] = []

    bib_entries = getattr(docling_doc, "bib_entries", None)
    if bib_entries:
        entries = _parse_structured_bib(bib_entries)
        if entries:
            return entries

    for attr_name in ("references", "ref_entries", "bibliography"):
        alt = getattr(docling_doc, attr_name, None)
        if alt and isinstance(alt, dict) and len(alt) > 0:
            entries = _parse_structured_bib(alt)
            if entries:
                return entries

    try:
        md = docling_doc.export_to_markdown()
        entries = _parse_references_from_markdown(md)
    except Exception:
        pass

    return entries


def _parse_structured_bib(bib_entries: dict) -> list[BibEntry]:
    """Parse structured bibliography dict from Docling."""
    entries: list[BibEntry] = []

    for key, entry in bib_entries.items():
        raw_text = ""
        title = ""
        authors = ""
        year = ""

        if hasattr(entry, "text"):
            raw_text = str(entry.text)
        elif isinstance(entry, str):
            raw_text = entry
        elif isinstance(entry, dict):
            raw_text = entry.get("text", str(entry))
            title = entry.get("title", "")
            authors = entry.get("authors", "")
            year = entry.get("year", "")

        if not title and raw_text:
            title = _guess_title_from_raw(raw_text)
        if not year and raw_text:
            year_match = re.search(r"\b(19|20)\d{2}\b", raw_text)
            year = year_match.group(0) if year_match else ""

        arxiv_id = _extract_arxiv_id(raw_text)
        doi = _extract_doi(raw_text)

        entries.append(
            BibEntry(
                key=str(key),
                title=title,
                authors=authors,
                year=year,
                arxiv_id=arxiv_id,
                doi=doi,
                raw_text=raw_text[:1000],
            )
        )

    return entries


def _parse_references_from_markdown(md: str) -> list[BibEntry]:
    """
    Fallback: extract references from the References section in markdown.

    Handles multiple formats:
      1. [N] Author, A. Title. Venue, year.
      2. - Author, A. Title. Venue, year.      (markdown bullet list)
      3. * Author, A. Title. Venue, year.      (markdown bullet list)
      4. 1. Author, A. Title. Venue, year.     (numbered list)
      5. Double-newline separated paragraphs

    Auto-detects format from first few lines.
    """
    entries: list[BibEntry] = []

    # Find References section
    ref_pattern = re.compile(
        r"^#+\s*(?:References|Bibliography)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = ref_pattern.search(md)
    if not match:
        match = re.search(r"^References\s*$", md, re.MULTILINE | re.IGNORECASE)
    if not match:
        return entries

    ref_text = md[match.end() :].strip()

    next_section = re.search(r"\n#{1,3}\s+(?!References|Bibliography)", ref_text)
    if next_section:
        ref_text = ref_text[: next_section.start()]

    raw_refs = _split_references(ref_text)

    for i, raw in enumerate(raw_refs):
        entry = _bib_entry_from_raw(f"ref_{i + 1}", raw)
        if entry:
            entries.append(entry)

    return entries


def _split_references(ref_text: str) -> list[str]:
    """
    Split reference section text into individual reference strings.

    Auto-detects format:
      - Bullet list: lines starting with "- " or "* "
      - Numbered brackets: [1], [2], ...
      - Numbered dot: "1. ", "2. ", ...
      - Paragraph-based: double newline separated

    Returns list of raw reference strings.
    """
    lines = ref_text.strip().split("\n")
    if not lines:
        return []

    bullet_count = sum(1 for l in lines[:10] if re.match(r"^\s*[-*]\s+\S", l))  # noqa: E741
    if bullet_count >= 3:  # noqa: PLR2004
        return _split_by_bullet(lines)

    bracket_count = sum(1 for l in lines[:10] if re.match(r"^\s*\[\d+\]", l))  # noqa: E741
    if bracket_count >= 3:  # noqa: PLR2004
        return _split_by_bracket(lines)

    numbered_count = sum(1 for l in lines[:10] if re.match(r"^\s*\d{1,3}\.\s+[A-Z]", l))  # noqa: E741
    if numbered_count >= 3:  # noqa: PLR2004
        return _split_by_numbered(lines)

    paragraphs = [p.strip() for p in ref_text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:  # noqa: PLR2004
        return [re.sub(r"\s+", " ", p).strip() for p in paragraphs if len(p) > 20]  # noqa: PLR2004

    # Last resort: treat each non-empty line as a reference
    return [l.strip() for l in lines if l.strip() and len(l.strip()) > 30]  # noqa: PLR2004, E741


def _split_by_bullet(lines: list[str]) -> list[str]:
    """Split references from a markdown bullet list."""
    refs: list[str] = []
    current: list[str] = []

    for line in lines:
        if re.match(r"^\s*[-*]\s+", line):
            if current:
                refs.append(" ".join(current))
            current = [re.sub(r"^\s*[-*]\s+", "", line).strip()]
        elif line.strip() and current:
            current.append(line.strip())

    if current:
        refs.append(" ".join(current))

    return [r for r in refs if len(r) > 20]  # noqa: PLR2004


def _split_by_bracket(lines: list[str]) -> list[str]:
    """Split references from [N] numbered format."""
    refs: list[str] = []
    current: list[str] = []

    for line in lines:
        if re.match(r"^\s*\[\d+\]", line):
            if current:
                refs.append(" ".join(current))
            current = [re.sub(r"^\s*\[\d+\]\s*", "", line).strip()]
        elif line.strip() and current:
            current.append(line.strip())

    if current:
        refs.append(" ".join(current))

    return [r for r in refs if len(r) > 20]  # noqa: PLR2004


def _split_by_numbered(lines: list[str]) -> list[str]:
    """Split references from '1. Author...' numbered format."""
    refs: list[str] = []
    current: list[str] = []

    for line in lines:
        if re.match(r"^\s*\d{1,3}\.\s+[A-Z]", line):
            if current:
                refs.append(" ".join(current))
            current = [re.sub(r"^\s*\d{1,3}\.\s+", "", line).strip()]
        elif line.strip() and current:
            current.append(line.strip())

    if current:
        refs.append(" ".join(current))

    return [r for r in refs if len(r) > 20]  # noqa: PLR2004


def _bib_entry_from_raw(key: str, raw_text: str) -> BibEntry | None:
    """Create a BibEntry from raw reference text."""
    raw_text = re.sub(r"\s+", " ", raw_text).strip()

    if len(raw_text) < 20:  # noqa: PLR2004
        return None

    raw_text = re.sub(r"\s*\(pages?\s*[\d,\s–-]+\)\s*$", "", raw_text).strip()

    title = _guess_title_from_raw(raw_text)
    year_match = re.search(r"\b(19|20)\d{2}\b", raw_text)
    year = year_match.group(0) if year_match else ""
    arxiv_id = _extract_arxiv_id(raw_text)
    doi = _extract_doi(raw_text)

    return BibEntry(
        key=key,
        title=title,
        year=year,
        arxiv_id=arxiv_id,
        doi=doi,
        raw_text=raw_text[:1000],
    )


def _guess_title_from_raw(raw_text: str) -> str:  # noqa: C901
    """
    Extract title from a raw reference string.
    """
    text = raw_text.strip()

    quoted = re.search(r'["\u201c](.+?)["\u201d]', text)
    if quoted and len(quoted.group(1)) > 10:  # noqa: PLR2004
        return quoted.group(1)

    year_paren = re.search(r"\(\d{4}[a-z]?\)\.\s*", text)
    if year_paren:
        title = _extract_title_after(text[year_paren.end() :])
        if title:
            return title

    year_comma = re.search(r",\s*\d{4}[a-z]?\.\s+", text)
    if year_comma:
        title = _extract_title_after(text[year_comma.end() :])
        if title:
            return title

    author_end = re.search(
        r"(?:"
        r"[A-Z]\.\s+"
        r"|Jr\.\s+"
        r"|III\.\s+"
        r"|et al\.\s+"
        r")"
        r"(?="
        r"[A-Z][a-z]{2,}"
        r"|[A-Z][a-z]+-"
        r"|['\u2018]"
        r"|\d+(?:st|nd|rd|th)"
        r")",
        text,
    )

    if author_end:
        title = _extract_title_after(text[author_end.end() :])
        if title:
            return title

    segments = re.split(r"\.\s+", text)
    for i, seg in enumerate(segments):
        if i == 0:
            continue
        seg = seg.strip().rstrip(".")  # noqa: PLW2901
        if len(seg) < 15:  # noqa: PLR2004
            continue
        parts = seg.split(",")
        avg_part_len = sum(len(p.strip()) for p in parts) / max(len(parts), 1)
        if len(parts) > 2 and avg_part_len < 15:  # noqa: PLR2004
            continue
        if re.match(
            r"(?:In |Proceedings|Journal|ArXiv|arXiv|Technical|pp\.|vol\.|http|doi|\d)", seg
        ):
            continue
        return seg

    return ""


def _extract_title_after(text: str) -> str:
    """
    Extract title from text that starts right after the author block.
    """
    title_end = re.search(
        r"\.\s+(?:"
        r"In |Proceedings|Journal|ArXiv|arXiv|Technical|"
        r"Unpublished|PhD thesis|Master'?s thesis|"
        r"pp\.\s|vol\.\s|volume\s|http|doi:|"
        r"(?:Springer|Elsevier|Cambridge|Oxford|MIT Press|"
        r"Curran|JMLR|PMLR|IEEE|ACM|SIAM|CRC)"
        r")",
        text,
    )

    if title_end:
        title = text[: title_end.start()].strip().rstrip(".")
    else:
        sent_end = re.search(r"\.\s+(?=[A-Z][a-z]{3,})", text)
        if sent_end:
            title = text[: sent_end.start()].strip().rstrip(".")
        else:
            title = text.strip().rstrip(".")

    if len(title) > 10:  # noqa: PLR2004
        return title
    return ""


class CitationExtractor:
    """
    Extracts citation edges from a parsed scientific document.

    Pipeline:
      1. Parse bib_entries from DoclingDocument → BibEntry list
      2. Resolve each BibEntry → arxiv_id or DOI
      3. Find inline citation contexts in chunks
      4. Produce CitationEdge records
    """

    def _resolve_unresolved(
        self,
        entries: list[BibEntry],
        corpus_titles: dict[str, str] | None,
    ) -> None:
        """
        Try to resolve entries missing arxiv_id/doi via local corpus matching.
        """
        if not corpus_titles:
            return

        unresolved = [e for e in entries if not e.arxiv_id and not e.doi and e.title]
        if not unresolved:
            return

        resolved_count = 0
        for entry in unresolved:
            normalized = entry.title.strip().lower()

            match = corpus_titles.get(normalized)
            if match:
                entry.arxiv_id = match
                resolved_count += 1
                continue

            cleaned = re.sub(r"[^\w\s]", "", normalized).strip()
            for corpus_title, arxiv_id in corpus_titles.items():
                corpus_cleaned = re.sub(r"[^\w\s]", "", corpus_title).strip()
                if cleaned == corpus_cleaned:
                    entry.arxiv_id = arxiv_id
                    resolved_count += 1
                    break

        if resolved_count:
            logger.info("Resolved %d references via corpus matching", resolved_count)

    def extract(
        self,
        docling_doc,
        document_id: str,
        citing_arxiv_id: str,
        chunks: list[Chunk] | None = None,
        corpus_titles: dict[str, str] | None = None,
    ) -> CitationExtractionResult:
        """
        Extract citations from a parsed document.

        Args:
            docling_doc: DoclingDocument from parser.
            document_id: Internal document ID.
            citing_arxiv_id: ArXiv ID of the citing paper.
            chunks: Optional chunks for context extraction.

        Returns:
            CitationExtractionResult with edges and stats.
        """
        # Step 1: Parse bib entries
        bib_entries = _parse_bib_entries(docling_doc)
        total = len(bib_entries)

        if not bib_entries:
            logger.info("No bib entries found for %s", document_id)
            return CitationExtractionResult(
                document_id=document_id,
                total_references=0,
            )

        if corpus_titles:
            self._resolve_unresolved(bib_entries, corpus_titles)

        # Step 2: Build ref_key → numbers mapping for context extraction
        ref_key_to_numbers = self._build_ref_number_mapping(bib_entries)

        # Step 3: Find citation contexts in chunks
        contexts: dict[str, list[tuple[str, str, str]]] = {}
        if chunks:
            contexts = _find_citation_contexts(chunks, ref_key_to_numbers)

        # Step 4: Build edges for resolved references
        edges: list[CitationEdge] = []
        unresolved = 0

        for entry in bib_entries:
            if entry.arxiv_id:
                cited_id = entry.arxiv_id
                id_type = "arxiv"
            elif entry.doi:
                cited_id = entry.doi
                id_type = "doi"
            else:
                unresolved += 1
                continue

            entry_contexts = contexts.get(entry.key, [])
            best_context = entry_contexts[0] if entry_contexts else ("", "", "")

            edge = CitationEdge(
                citing_paper_id=citing_arxiv_id,
                cited_paper_id=cited_id,
                cited_id_type=id_type,
                context=best_context[0][:1000],
                section=best_context[1],
                section_type=best_context[2],
                ref_key=entry.key,
                cited_title=entry.title[:500],
                cited_authors=entry.authors[:500],
                cited_year=entry.year,
            )
            edges.append(edge)

        result = CitationExtractionResult(
            document_id=document_id,
            edges=edges,
            bib_entries=bib_entries,
            unresolved_count=unresolved,
            total_references=total,
        )

        logger.info(
            "Citation extraction: %d/%d resolved (%.0f%%) for %s",
            result.resolved_count,
            total,
            result.resolution_rate * 100,
            document_id,
        )

        return result

    @staticmethod
    def _build_ref_number_mapping(bib_entries: list[BibEntry]) -> dict[str, list[str]]:
        """
        Map bib entry keys to their likely citation numbers.
        """
        mapping: dict[str, list[str]] = {}
        for entry in bib_entries:
            numbers = re.findall(r"\d+", entry.key)
            if numbers:
                mapping[entry.key] = numbers
        return mapping
