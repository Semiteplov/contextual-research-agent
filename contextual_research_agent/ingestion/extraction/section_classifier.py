from __future__ import annotations

import re
from enum import Enum

from contextual_research_agent.ingestion.domain.entities import Chunk


class SectionType(str, Enum):
    """Semantic type of a paper section."""

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHOD = "method"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    LIMITATIONS = "limitations"
    ETHICS = "ethics"
    APPENDIX = "appendix"
    REFERENCES = "references"
    UNKNOWN = "unknown"


_EXACT_PATTERNS: list[tuple[re.Pattern, SectionType]] = [
    # Abstract
    (re.compile(r"^abstract$"), SectionType.ABSTRACT),
    # Introduction
    (re.compile(r"^introduction$"), SectionType.INTRODUCTION),
    (re.compile(r"^overview$"), SectionType.INTRODUCTION),
    # Related work / Background
    (re.compile(r"^related\s*work[s]?$"), SectionType.RELATED_WORK),
    (re.compile(r"^prior\s*work[s]?$"), SectionType.RELATED_WORK),
    (re.compile(r"^literature\s*review$"), SectionType.RELATED_WORK),
    (re.compile(r"^background$"), SectionType.BACKGROUND),
    (re.compile(r"^preliminaries$"), SectionType.BACKGROUND),
    (re.compile(r"^background\s*and\s*related\s*work[s]?$"), SectionType.RELATED_WORK),
    (re.compile(r"^related\s*work[s]?\s*and\s*background$"), SectionType.RELATED_WORK),
    # Method
    (re.compile(r"^method[s]?$"), SectionType.METHOD),
    (re.compile(r"^methodology$"), SectionType.METHOD),
    (re.compile(r"^approach$"), SectionType.METHOD),
    (re.compile(r"^proposed\s*(method|approach|framework|model|system)$"), SectionType.METHOD),
    (re.compile(r"^model$"), SectionType.METHOD),
    (re.compile(r"^framework$"), SectionType.METHOD),
    (re.compile(r"^architecture$"), SectionType.METHOD),
    (re.compile(r"^system\s*(design|description|overview)$"), SectionType.METHOD),
    (re.compile(r"^technical\s*approach$"), SectionType.METHOD),
    (re.compile(r"^formulation$"), SectionType.METHOD),
    (re.compile(r"^problem\s*(formulation|definition|setup|statement)$"), SectionType.METHOD),
    # Experiments
    (re.compile(r"^experiment[s]?$"), SectionType.EXPERIMENTS),
    (
        re.compile(r"^experimental\s*(setup|setting[s]?|design|evaluation)$"),
        SectionType.EXPERIMENTS,
    ),
    (re.compile(r"^evaluation$"), SectionType.EXPERIMENTS),
    (re.compile(r"^empirical\s*(evaluation|study|analysis|results)$"), SectionType.EXPERIMENTS),
    (re.compile(r"^implementation(\s*details)?$"), SectionType.EXPERIMENTS),
    (re.compile(r"^setup$"), SectionType.EXPERIMENTS),
    (re.compile(r"^training\s*(details|setup|procedure)$"), SectionType.EXPERIMENTS),
    (re.compile(r"^dataset[s]?$"), SectionType.EXPERIMENTS),
    (re.compile(r"^benchmark[s]?$"), SectionType.EXPERIMENTS),
    (re.compile(r"^baseline[s]?$"), SectionType.EXPERIMENTS),
    (re.compile(r"^ablation(\s*stud(y|ies))?$"), SectionType.EXPERIMENTS),
    (re.compile(r"^hyperparameter[s]?$"), SectionType.EXPERIMENTS),
    # Results
    (re.compile(r"^results$"), SectionType.RESULTS),
    (re.compile(r"^main\s*results$"), SectionType.RESULTS),
    (re.compile(r"^results\s*and\s*(discussion|analysis)$"), SectionType.RESULTS),
    (re.compile(r"^findings$"), SectionType.RESULTS),
    (re.compile(r"^analysis$"), SectionType.RESULTS),
    (re.compile(r"^quantitative\s*(results|analysis)$"), SectionType.RESULTS),
    (re.compile(r"^qualitative\s*(results|analysis)$"), SectionType.RESULTS),
    (re.compile(r"^case\s*stud(y|ies)$"), SectionType.RESULTS),
    (re.compile(r"^comparison[s]?$"), SectionType.RESULTS),
    # Discussion
    (re.compile(r"^discussion$"), SectionType.DISCUSSION),
    (re.compile(r"^broader\s*impact[s]?$"), SectionType.DISCUSSION),
    (re.compile(r"^societal\s*impact[s]?$"), SectionType.DISCUSSION),
    (re.compile(r"^implications$"), SectionType.DISCUSSION),
    # Limitations
    (re.compile(r"^limitation[s]?$"), SectionType.LIMITATIONS),
    (re.compile(r"^limitations\s*and\s*future\s*work$"), SectionType.LIMITATIONS),
    (re.compile(r"^threats\s*to\s*validity$"), SectionType.LIMITATIONS),
    (re.compile(r"^failure\s*(cases|analysis|modes)$"), SectionType.LIMITATIONS),
    # Conclusion
    (re.compile(r"^conclusion[s]?$"), SectionType.CONCLUSION),
    (re.compile(r"^summary$"), SectionType.CONCLUSION),
    (re.compile(r"^conclusion[s]?\s*and\s*future\s*work$"), SectionType.CONCLUSION),
    (re.compile(r"^concluding\s*remarks$"), SectionType.CONCLUSION),
    (re.compile(r"^future\s*work$"), SectionType.CONCLUSION),
    # Ethics
    (re.compile(r"^ethic[s]?(\s*statement)?$"), SectionType.ETHICS),
    (re.compile(r"^ethical\s*considerations$"), SectionType.ETHICS),
    (re.compile(r"^responsible\s*(ai|use)$"), SectionType.ETHICS),
    # Appendix
    (re.compile(r"^appendix"), SectionType.APPENDIX),
    (re.compile(r"^supplementary"), SectionType.APPENDIX),
    (re.compile(r"^additional\s*(details|experiments|results|material)$"), SectionType.APPENDIX),
    # References
    (re.compile(r"^references$"), SectionType.REFERENCES),
    (re.compile(r"^bibliography$"), SectionType.REFERENCES),
]

_KEYWORD_FALLBACK: list[tuple[str, SectionType]] = [
    ("related work", SectionType.RELATED_WORK),
    ("prior work", SectionType.RELATED_WORK),
    ("background", SectionType.BACKGROUND),
    ("method", SectionType.METHOD),
    ("approach", SectionType.METHOD),
    ("architecture", SectionType.METHOD),
    ("experiment", SectionType.EXPERIMENTS),
    ("evaluation", SectionType.EXPERIMENTS),
    ("ablation", SectionType.EXPERIMENTS),
    ("dataset", SectionType.EXPERIMENTS),
    ("baseline", SectionType.EXPERIMENTS),
    ("training", SectionType.EXPERIMENTS),
    ("result", SectionType.RESULTS),
    ("analysis", SectionType.RESULTS),
    ("finding", SectionType.RESULTS),
    ("discussion", SectionType.DISCUSSION),
    ("limitation", SectionType.LIMITATIONS),
    ("conclusion", SectionType.CONCLUSION),
    ("future work", SectionType.CONCLUSION),
    ("appendix", SectionType.APPENDIX),
    ("supplement", SectionType.APPENDIX),
    ("reference", SectionType.REFERENCES),
    ("introduction", SectionType.INTRODUCTION),
]


# Regex to strip leading section numbers: "3.1", "3.1.", "A.2", "IV.", etc.
_SECTION_NUM_RE = re.compile(
    r"^(?:"
    r"[A-Z]\.(?:\d+\.?)*\s+"  # A., A.1, A.1.2
    r"|[A-Z]\d+(?:\.\d+)*\.?\s+"  # A1, A1.2
    r"|[IVXLC]{2,}\.?\s+"  # Roman numerals: IV., III
    r"|\d+(?:\.\d+)*\.?\s+"  # 3, 3., 3.1, 3.1.2
    r")"
)


def _normalize_heading(heading: str) -> str:
    """
    Normalize a section heading for matching.

    Strips: leading numbers/letters, leading/trailing whitespace,
    punctuation, converts to lowercase.
    """
    text = heading.strip()
    # Strip markdown headers
    text = re.sub(r"^#+\s*", "", text)
    # Strip section numbers
    text = _SECTION_NUM_RE.sub("", text)
    # Strip trailing punctuation and whitespace
    text = text.strip().rstrip(".:;").strip()
    # Lowercase
    return text.lower()


def _extract_last_heading(section_path: str) -> str:
    """
    Extract the most specific heading from a hierarchical path.

    "3 Method > 3.1 Architecture > 3.1.1 Encoder" → "3.1.1 Encoder"
    """
    if " > " in section_path:
        return section_path.rsplit(" > ", 1)[-1]
    return section_path


def _extract_section_number(heading: str) -> str | None:
    """
    Extract the section number prefix from a heading.

    "2.1. Problem Setup" → "2.1"
    "2.2. Wasserstein Variational Inference..." → "2.2"
    """
    match = re.match(r"^([A-Z]?\d+(?:\.\d+)*)", heading.strip())
    return match.group(1) if match else None


def _get_parent_number(section_number: str) -> str | None:
    """
    Get parent section number.

    "2.1" → "2"
    "3.2.1" → "3.2"
    "2" → None
    "A.3" → "A"
    """
    if "." in section_number:
        return section_number.rsplit(".", 1)[0]
    return None


class SectionClassifier:
    """
    Rule-based section type classifier for scientific papers.

    Matching strategy:
      1. Direct classify on the heading text.
      2. If UNKNOWN and heading is a subsection (e.g., "2.1. Problem Setup"),
         find parent section from the document's heading index and inherit
         its classification.
      3. Propagate: consecutive UNKNOWN chunks inherit from the last
         known section type.
    """

    def _classify_single(self, heading: str) -> SectionType:
        """
        Classify a single heading string against patterns.
        """
        if not heading or not heading.strip():
            return SectionType.UNKNOWN

        candidates = [_extract_last_heading(heading)]
        if " > " in heading:
            parent = heading.rsplit(" > ", 1)[0]
            candidates.append(_extract_last_heading(parent))

        for candidate in candidates:
            normalized = _normalize_heading(candidate)
            if not normalized:
                continue

            for pattern, section_type in _EXACT_PATTERNS:
                if pattern.match(normalized):
                    return section_type

            for keyword, section_type in _KEYWORD_FALLBACK:
                if keyword in normalized:
                    return section_type

        return SectionType.UNKNOWN

    def classify(self, heading: str) -> SectionType:
        """Classify a single heading."""
        return self._classify_single(heading)

    def classify_batch(self, headings: list[str]) -> list[SectionType]:
        """Classify multiple headings."""
        return [self.classify(h) for h in headings]

    def enrich_chunks(self, chunks: list[Chunk]) -> list[Chunk]:  # noqa: C901, PLR0912
        """
        Enrich chunks with section_type using three strategies:

        1. Direct classification of chunk.section heading.
        2. Parent section inheritance: if chunk "2.1 Problem Setup" → UNKNOWN,
           look up "2. Background" in the heading index → BACKGROUND.
        3. Propagation: if still UNKNOWN, inherit from the last chunk
           that had a known section type.
        """
        if not chunks:
            return chunks

        #  Build heading index (section_number → heading text)
        number_to_heading: dict[str, str] = {}
        for chunk in chunks:
            sec_num = _extract_section_number(chunk.section)
            if sec_num and chunk.section:
                number_to_heading[sec_num] = chunk.section

        # Classify each chunk with parent fallback
        last_known_type = SectionType.UNKNOWN

        for chunk in chunks:
            normalized = _normalize_heading(chunk.section)

            if (
                chunk.chunk_index == 0
                and not _extract_section_number(chunk.section)
                and normalized != "abstract"
            ):
                section_type = SectionType.TITLE
            elif normalized == "abstract":
                section_type = SectionType.ABSTRACT
            elif not _extract_section_number(chunk.section) and not normalized:
                section_type = SectionType.UNKNOWN  # truly empty
            elif not _extract_section_number(chunk.section):
                section_type = SectionType.UNKNOWN
            else:
                section_type = self._classify_single(chunk.section)

                if section_type == SectionType.UNKNOWN:
                    sec_num = _extract_section_number(chunk.section)
                    if sec_num:
                        parent_num = _get_parent_number(sec_num)
                        while parent_num and section_type == SectionType.UNKNOWN:
                            parent_heading = number_to_heading.get(parent_num)
                            if parent_heading:
                                section_type = self._classify_single(parent_heading)
                            parent_num = _get_parent_number(parent_num) if parent_num else None

            if section_type == SectionType.UNKNOWN and last_known_type != SectionType.UNKNOWN:
                section_type = last_known_type
            elif section_type != SectionType.UNKNOWN:
                last_known_type = section_type

            chunk.metadata["section_type"] = section_type.value

        return chunks

    def get_classification_stats(self, chunks: list[Chunk]) -> dict[str, int]:
        """
        Get section type distribution for a set of chunks.
        """
        stats: dict[str, int] = {}
        for chunk in chunks:
            st = chunk.metadata.get("section_type", SectionType.UNKNOWN.value)
            stats[st] = stats.get(st, 0) + 1
        return stats


_default_classifier = SectionClassifier()


def classify_section(heading: str) -> SectionType:
    """Convenience function using default classifier."""
    return _default_classifier.classify(heading)


def enrich_chunks_with_section_types(chunks: list[Chunk]) -> list[Chunk]:
    """Convenience function using default classifier."""
    return _default_classifier.enrich_chunks(chunks)
