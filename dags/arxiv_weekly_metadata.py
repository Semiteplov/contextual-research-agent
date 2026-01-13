import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from ssl import SSLError

from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

from data_pipeline.arxiv.constants import ML_CATEGORIES
from data_pipeline.arxiv.parser import PaperRow
from db.arxiv.metadata import upsert_papers
from db.arxiv.sync_state import get_sync_state, upsert_sync_state

logger = logging.getLogger(__name__)

ARXIV_DB_NAME = Variable.get("ARXIV_DB_NAME", default_var="arxiv")
ARXIV_API_URL = "http://export.arxiv.org/api/query"

PAGE_SIZE = 200
MAX_PAGES_PER_CATEGORY = 200

DELAY_IN_SECONDS = 3.0
USER_AGENT = "contextual-research-agent/1.0"

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

MAX_FETCH_RETRIES = 5
FETCH_TIMEOUT_SECONDS = 30
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 30.0


@dataclass(frozen=True)
class ParsedEntry:
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    categories: list[str]
    primary_category: str | None
    doi: str | None
    journal_ref: str | None
    published: datetime
    updated: datetime
    latest_version: int | None


def _parse_dt(value: str) -> datetime:
    if value.endswith("Z"):
        dt = datetime.fromisoformat(value[:-1]).replace(tzinfo=UTC)
    else:
        dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _parse_arxiv_id(entry_id_url: str) -> str:
    tail = entry_id_url.rstrip("/").split("/")[-1]
    if "v" in tail and tail.split("v")[-1].isdigit():
        return tail.rsplit("v", 1)[0]
    return tail


def _parse_version_from_abs_url(abs_url: str) -> int | None:
    tail = abs_url.rstrip("/").split("/")[-1]
    if "v" in tail and tail.split("v")[-1].isdigit():
        return int(tail.split("v")[-1])
    return None


def _build_query_url(category: str, start: int, max_results: int) -> str:
    params = {
        "search_query": f"cat:{category}",
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"


def _fetch_feed_xml(url: str) -> bytes:
    last_err: Exception | None = None

    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_SECONDS) as resp:
                return resp.read()

        except (TimeoutError, SSLError, urllib.error.URLError, ConnectionError) as e:
            last_err = e
            if attempt == MAX_FETCH_RETRIES:
                break

            sleep_s = min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)
            sleep_s = sleep_s * (0.7 + random.random() * 0.6)

            logger.warning(
                "arXiv fetch failed (attempt %d/%d). url=%s err=%r; sleeping %.1fs and retrying",
                attempt,
                MAX_FETCH_RETRIES,
                url,
                e,
                sleep_s,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"arXiv fetch failed after {MAX_FETCH_RETRIES} attempts: {url}"
    ) from last_err


def _parse_feed_entries(xml_bytes: bytes) -> list[ParsedEntry]:
    root = ET.fromstring(xml_bytes)
    out: list[ParsedEntry] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id_url = _text(entry.find("atom:id", ATOM_NS))
        arxiv_id = _parse_arxiv_id(entry_id_url)

        published = _parse_dt(_text(entry.find("atom:published", ATOM_NS)))
        updated = _parse_dt(_text(entry.find("atom:updated", ATOM_NS)))

        title = _text(entry.find("atom:title", ATOM_NS))
        abstract = _text(entry.find("atom:summary", ATOM_NS))

        authors = []
        for a in entry.findall("atom:author/atom:name", ATOM_NS):
            name = _text(a)
            if name:
                authors.append(name)
        authors_str = ", ".join(authors)

        cats: list[str] = []
        for c in entry.findall("atom:category", ATOM_NS):
            term = c.attrib.get("term")
            if term:
                cats.append(term)

        primary_el = entry.find("arxiv:primary_category", ATOM_NS)
        primary = (
            primary_el.attrib.get("term") if primary_el is not None else (cats[0] if cats else None)
        )

        doi = None
        doi_el = entry.find("arxiv:doi", ATOM_NS)
        if doi_el is not None:
            doi = _text(doi_el) or None

        journal_ref = None
        jr_el = entry.find("arxiv:journal_ref", ATOM_NS)
        if jr_el is not None:
            journal_ref = _text(jr_el) or None

        latest_version = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("rel") == "alternate":
                alt = link.attrib.get("href") or ""
                v = _parse_version_from_abs_url(alt)
                if v is not None:
                    latest_version = v

        out.append(
            ParsedEntry(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors_str,
                categories=cats,
                primary_category=primary,
                doi=doi,
                journal_ref=journal_ref,
                published=published,
                updated=updated,
                latest_version=latest_version,
            )
        )
    return out


def _filter_new_entries_page(
    entries: list[ParsedEntry],
    last_synced_at: datetime | None,
) -> tuple[list[ParsedEntry], bool]:
    if not entries:
        return [], True

    new_entries: list[ParsedEntry] = []

    for e in entries:
        if last_synced_at is not None and e.published <= last_synced_at:
            return new_entries, True
        new_entries.append(e)

    return new_entries, False


def _to_paper_rows(entries: list[ParsedEntry], fallback_category: str) -> list[PaperRow]:
    out: list[PaperRow] = []
    for e in entries:
        out.append(
            PaperRow(
                arxiv_id=e.arxiv_id,
                title=e.title,
                abstract=e.abstract,
                authors=e.authors,
                categories=e.categories or [fallback_category],
                primary_category=e.primary_category,
                doi=e.doi,
                journal_ref=e.journal_ref,
                update_date=e.updated.date(),
                latest_version=e.latest_version,
                latest_version_created=e.updated,
            )
        )
    return out


def _sync_category(conn, category: str) -> int:
    last_synced_at = get_sync_state(conn, category)
    logger.info("Category=%s last_synced_at=%s", category, last_synced_at)

    total_upserted = 0
    newest_published: datetime | None = None

    start = 0
    pages = 0

    while pages < MAX_PAGES_PER_CATEGORY:
        url = _build_query_url(category, start=start, max_results=PAGE_SIZE)
        logger.info("arXiv query: %s", url)

        xml_bytes = _fetch_feed_xml(url)
        entries = _parse_feed_entries(xml_bytes)

        new_entries, reached_cursor = _filter_new_entries_page(entries, last_synced_at)

        if newest_published is None and entries:
            newest_published = entries[0].published

        if new_entries:
            rows = _to_paper_rows(new_entries, fallback_category=category)
            upserted = upsert_papers(conn, rows)
            total_upserted += upserted
            logger.info(
                "Category=%s page=%d start=%d upserted=%d total_upserted=%d",
                category,
                pages + 1,
                start,
                upserted,
                total_upserted,
            )
        else:
            logger.info("Category=%s page=%d start=%d: no new entries", category, pages + 1, start)

        pages += 1

        if reached_cursor:
            break
        if len(entries) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(DELAY_IN_SECONDS)

    if pages >= MAX_PAGES_PER_CATEGORY:
        logger.warning(
            "Category=%s reached MAX_PAGES_PER_CATEGORY=%d; stopping early. "
            "Consider increasing the cap if you expect more new submissions.",
            category,
            MAX_PAGES_PER_CATEGORY,
        )

    if total_upserted > 0 and newest_published is not None:
        upsert_sync_state(conn, category, newest_published)

    logger.info(
        "Category=%s finished. total_upserted=%d newest_published=%s",
        category,
        total_upserted,
        newest_published,
    )
    return total_upserted


def fetch_and_upsert_metadata() -> None:
    hook = PostgresHook(postgres_conn_id="arxiv_postgres")
    conn = hook.get_conn()
    conn.autocommit = False

    total_upserted = 0
    failed: list[str] = []

    try:
        for cat in sorted(ML_CATEGORIES):
            try:
                upserted = _sync_category(conn, cat)
                conn.commit()
                total_upserted += upserted
            except Exception:
                conn.rollback()
                logger.exception("Category=%s failed; rolled back category txn", cat)
                failed.append(cat)

            time.sleep(DELAY_IN_SECONDS)

        logger.info(
            "Weekly metadata ingestion finished. total_upserted=%d failed_categories=%s",
            total_upserted,
            failed or "[]",
        )

        if failed:
            raise RuntimeError(f"Some categories failed: {failed}")

    finally:
        conn.close()


with DAG(
    dag_id="arxiv_weekly_metadata",
    description="Weekly fetch of arXiv metadata for ML categories into arxiv_papers_metadata",
    start_date=datetime(2026, 1, 11, tzinfo=UTC),
    schedule="0 3 * * 1",  # Mondays 03:00 UTC
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    tags=["arxiv", "metadata", "weekly"],
) as dag:
    PythonOperator(
        task_id="fetch_and_upsert_metadata",
        python_callable=fetch_and_upsert_metadata,
    )
