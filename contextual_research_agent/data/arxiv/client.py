import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from ssl import SSLError

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.data.arxiv.schemas import ParsedEntry

logger = get_logger(__name__)


class ArxivClientConfig:
    """Configuration for arXiv API client."""

    API_URL = "http://export.arxiv.org/api/query"
    USER_AGENT = "contextual-research-agent/1.0"

    PAGE_SIZE = 200
    MAX_PAGES_PER_CATEGORY = 200
    DELAY_SECONDS = 3.0

    MAX_RETRIES = 5
    TIMEOUT_SECONDS = 30
    BACKOFF_BASE_SECONDS = 2.0
    BACKOFF_MAX_SECONDS = 30.0


# XML namespaces for Atom feed parsing
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivClient:
    """Client for fetching papers from arXiv API."""

    def __init__(self, config: ArxivClientConfig | None = None) -> None:
        self.config = config or ArxivClientConfig()

    def fetch_category(
        self,
        category: str,
        last_synced_at: datetime | None = None,
    ) -> tuple[list[ParsedEntry], datetime | None]:
        """
        Fetch all new papers for a category since last sync.

        Args:
            category: arXiv category (e.g., "cs.AI").
            last_synced_at: Timestamp of last successful sync.

        Returns:
            Tuple of (list of new entries, newest published datetime).
        """
        all_entries: list[ParsedEntry] = []
        newest_published: datetime | None = None
        start = 0

        for page in range(self.config.MAX_PAGES_PER_CATEGORY):
            url = self._build_query_url(category, start)
            logger.info("arXiv query: %s", url)

            xml_bytes = self._fetch_with_retry(url)
            entries = self._parse_feed(xml_bytes)

            if newest_published is None and entries:
                newest_published = entries[0].published

            new_entries, reached_cursor = self._filter_new_entries(entries, last_synced_at)
            all_entries.extend(new_entries)

            logger.info(
                "Category=%s page=%d start=%d fetched=%d new=%d",
                category,
                page + 1,
                start,
                len(entries),
                len(new_entries),
            )

            if reached_cursor or len(entries) < self.config.PAGE_SIZE:
                break

            start += self.config.PAGE_SIZE
            time.sleep(self.config.DELAY_SECONDS)
        else:
            logger.warning(
                "Category=%s reached MAX_PAGES=%d; stopping early",
                category,
                self.config.MAX_PAGES_PER_CATEGORY,
            )

        return all_entries, newest_published

    def _build_query_url(self, category: str, start: int) -> str:
        """Build arXiv API query URL."""
        params = {
            "search_query": f"cat:{category}",
            "start": str(start),
            "max_results": str(self.config.PAGE_SIZE),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        return f"{self.config.API_URL}?{urllib.parse.urlencode(params)}"

    def _fetch_with_retry(self, url: str) -> bytes:
        """Fetch URL with exponential backoff retry."""
        last_error: Exception | None = None

        for attempt in range(1, self.config.MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.config.USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.config.TIMEOUT_SECONDS) as resp:
                    return resp.read()

            except (
                TimeoutError,
                SSLError,
                urllib.error.URLError,
                ConnectionError,
            ) as e:
                last_error = e
                if attempt == self.config.MAX_RETRIES:
                    break

                sleep_seconds = min(
                    self.config.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                    self.config.BACKOFF_MAX_SECONDS,
                )

                sleep_seconds *= 0.7 + random.random() * 0.6

                logger.warning(
                    "arXiv fetch failed (attempt %d/%d): %r; retrying in %.1fs",
                    attempt,
                    self.config.MAX_RETRIES,
                    e,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise RuntimeError(
            f"arXiv fetch failed after {self.config.MAX_RETRIES} attempts: {url}"
        ) from last_error

    def _parse_feed(self, xml_bytes: bytes) -> list[ParsedEntry]:
        """Parse Atom feed XML into ParsedEntry objects."""
        root = ET.fromstring(xml_bytes)
        entries: list[ParsedEntry] = []

        for entry_el in root.findall("atom:entry", _ATOM_NS):
            entry = self._parse_entry(entry_el)
            if entry:
                entries.append(entry)

        return entries

    def _parse_entry(self, entry_el: ET.Element) -> ParsedEntry | None:
        """Parse single Atom entry element."""
        entry_id_url = _text(entry_el.find("atom:id", _ATOM_NS))
        if not entry_id_url:
            return None

        arxiv_id = _parse_arxiv_id(entry_id_url)

        # Parse dates
        published = _parse_datetime(_text(entry_el.find("atom:published", _ATOM_NS)))
        updated = _parse_datetime(_text(entry_el.find("atom:updated", _ATOM_NS)))

        # Parse metadata
        title = _text(entry_el.find("atom:title", _ATOM_NS))
        abstract = _text(entry_el.find("atom:summary", _ATOM_NS))

        # Parse authors
        authors = [
            _text(a) for a in entry_el.findall("atom:author/atom:name", _ATOM_NS) if _text(a)
        ]

        # Parse categories
        categories = [
            c.attrib["term"]
            for c in entry_el.findall("atom:category", _ATOM_NS)
            if c.attrib.get("term")
        ]

        # Primary category
        primary_el = entry_el.find("arxiv:primary_category", _ATOM_NS)
        primary_category = (
            primary_el.attrib.get("term")
            if primary_el is not None
            else (categories[0] if categories else None)
        )

        # Optional fields
        doi = _text(entry_el.find("arxiv:doi", _ATOM_NS)) or None
        journal_ref = _text(entry_el.find("arxiv:journal_ref", _ATOM_NS)) or None

        # Parse version from alternate link
        latest_version = None
        for link in entry_el.findall("atom:link", _ATOM_NS):
            if link.attrib.get("rel") == "alternate":
                href = link.attrib.get("href", "")
                latest_version = _parse_version(href)
                break

        return ParsedEntry(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=", ".join(authors),
            categories=categories,
            primary_category=primary_category,
            doi=doi,
            journal_ref=journal_ref,
            published=published,
            updated=updated,
            latest_version=latest_version,
        )

    @staticmethod
    def _filter_new_entries(
        entries: list[ParsedEntry],
        last_synced_at: datetime | None,
    ) -> tuple[list[ParsedEntry], bool]:
        """
        Filter entries newer than last sync.

        Returns:
            Tuple of (new entries, whether we reached the sync cursor).
        """
        if not entries or last_synced_at is None:
            return entries, not entries

        new_entries: list[ParsedEntry] = []
        for entry in entries:
            if entry.published <= last_synced_at:
                return new_entries, True
            new_entries.append(entry)

        return new_entries, False


def _text(element: ET.Element | None) -> str:
    """Extract text content from XML element."""
    return (element.text or "").strip() if element is not None else ""


def _parse_datetime(value: str) -> datetime:
    """Parse ISO datetime string to UTC datetime."""
    if not value:
        return datetime.now(UTC)

    if value.endswith("Z"):
        dt = datetime.fromisoformat(value[:-1]).replace(tzinfo=UTC)
    else:
        dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)


def _parse_arxiv_id(entry_id_url: str) -> str:
    """Extract arXiv ID from entry URL, stripping version suffix."""
    tail = entry_id_url.rstrip("/").split("/")[-1]
    if "v" in tail and tail.split("v")[-1].isdigit():
        return tail.rsplit("v", 1)[0]
    return tail


def _parse_version(url: str) -> int | None:
    """Extract version number from arXiv URL."""
    tail = url.rstrip("/").split("/")[-1]
    if "v" in tail and tail.split("v")[-1].isdigit():
        return int(tail.split("v")[-1])
    return None
