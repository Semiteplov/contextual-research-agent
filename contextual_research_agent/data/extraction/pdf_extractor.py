import re
from dataclasses import dataclass
from enum import Enum

import pymupdf

from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


class ExtractionMethod(Enum):
    PYMUPDF = "pymupdf"


@dataclass
class ExtractionResult:
    arxiv_id: str
    text: str
    method: ExtractionMethod
    num_pages: int
    num_characters: int
    num_words: int
    status: str  # 'completed', 'failed', 'partial'
    error_message: str | None = None

    @property
    def is_successful(self) -> bool:
        return self.status in ("completed", "partial")


class PDFExtractor:
    def __init__(
        self,
        min_text_length: int = 100,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
    ) -> None:
        self.min_text_length = min_text_length
        self.remove_headers_footers = remove_headers_footers
        self.normalize_whitespace = normalize_whitespace

    def extract_from_bytes(self, content: bytes, arxiv_id: str) -> ExtractionResult:
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as e:
            logger.error("Failed to open PDF %s: %s", arxiv_id, e)
            return ExtractionResult(
                arxiv_id=arxiv_id,
                text="",
                method=ExtractionMethod.PYMUPDF,
                num_pages=0,
                num_characters=0,
                num_words=0,
                status="failed",
                error_message=f"Failed to open PDF: {e}",
            )

        try:
            pages_text = []
            num_pages = len(doc)

            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text("text")
                pages_text.append(text)

            doc.close()

            full_text = "\n\n".join(pages_text)

            if self.remove_headers_footers:
                full_text = self._remove_headers_footers(pages_text)

            if self.normalize_whitespace:
                full_text = self._normalize_whitespace(full_text)

            full_text = self._clean_artifacts(full_text)

            num_characters = len(full_text)
            num_words = len(full_text.split())

            if num_characters < self.min_text_length:
                status = "partial"
                error_message = f"Extracted text too short: {num_characters} chars"
            else:
                status = "completed"
                error_message = None

            return ExtractionResult(
                arxiv_id=arxiv_id,
                text=full_text,
                method=ExtractionMethod.PYMUPDF,
                num_pages=num_pages,
                num_characters=num_characters,
                num_words=num_words,
                status=status,
                error_message=error_message,
            )

        except Exception as e:
            logger.exception("Failed to extract text from %s", arxiv_id)
            return ExtractionResult(
                arxiv_id=arxiv_id,
                text="",
                method=ExtractionMethod.PYMUPDF,
                num_pages=0,
                num_characters=0,
                num_words=0,
                status="failed",
                error_message=f"Extraction failed: {e}",
            )

    def _remove_headers_footers(self, pages_text: list[str]) -> str:
        if len(pages_text) < 3:  # noqa: PLR2004
            return "\n\n".join(pages_text)

        cleaned_pages = []

        for page_text in pages_text:
            lines = page_text.split("\n")

            cleaned_lines = []
            for line in lines:
                stripped = line.strip()

                if not stripped:
                    cleaned_lines.append(line)
                    continue

                if re.match(r"^\d+$", stripped):
                    continue

                if re.match(r"^arXiv:\d+\.\d+", stripped, re.IGNORECASE):
                    continue

                cleaned_lines.append(line)

            cleaned_pages.append("\n".join(cleaned_lines))

        return "\n\n".join(cleaned_pages)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        return text.strip()

    def _clean_artifacts(self, text: str) -> str:
        text = re.sub(r"-\n(\w)", r"\1", text)
        replacements = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "æ": "ae",
            "œ": "oe",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = text.replace("\f", "\n\n")

        return text.replace("\x00", "")
