import asyncio
import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
)
from docling_core.types.io import DocumentStream
from transformers import AutoTokenizer

from contextual_research_agent.common import logging
from contextual_research_agent.data.storage.s3_client import S3Client
from contextual_research_agent.ingestion.domain.entities import (
    Chunk,
    ChunkType,
    DoclingParserConfig,
    Document,
)
from contextual_research_agent.ingestion.domain.types import DocumentStatus
from contextual_research_agent.ingestion.parsers.metrics import (
    ChunkingMetrics,
    compute_chunking_metrics,
)

logger = logging.get_logger(__name__)


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


class ScientificSerializerProvider(ChunkingSerializerProvider):
    """
    Serializer optimized for scientific papers:
    - Tables → Markdown format (preserves structure for retrieval)
    - Figures → annotation-based description when available, placeholder otherwise

    Rationale: default triplet table notation is unreadable for LLMs;
    Markdown tables preserve column/row semantics. Figure descriptions
    capture architecture diagrams and result plots that are otherwise lost.
    """

    def __init__(self, image_placeholder: str = "[Figure]"):
        self._image_placeholder = image_placeholder

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=_AnnotationPictureSerializer(),
            params=MarkdownParams(image_placeholder=self._image_placeholder),
        )


class _AnnotationPictureSerializer(MarkdownPictureSerializer):
    """
    Serialize figures using Docling's annotation metadata when available.
    Falls back to the configured placeholder for unannotated figures.
    """

    def serialize(self, *, item, doc_serializer, doc, **kwargs):
        parts: list[str] = []

        if hasattr(item, "meta") and item.meta is not None:
            # Classification (e.g., "chart", "diagram", "photo")
            classification = getattr(item.meta, "classification", None)
            if classification is not None:
                main_pred = classification.get_main_prediction()
                if main_pred is not None:
                    parts.append(f"Figure type: {main_pred.class_name}")

            # Description (caption / generated description)
            description = getattr(item.meta, "description", None)
            if description is not None and hasattr(description, "text"):
                parts.append(f"Figure description: {description.text}")

            # Molecule SMILES (for chemistry papers)
            molecule = getattr(item.meta, "molecule", None)
            if molecule is not None and hasattr(molecule, "smi"):
                parts.append(f"SMILES: {molecule.smi}")

        text = "\n".join(parts) if parts else ""
        text = doc_serializer.post_process(text=text)
        return create_ser_result(text=text, span_source=item)


def _extract_title_structural(docling_doc, fallback: str) -> str:
    """Extract title from DoclingDocument structure."""
    title = getattr(docling_doc, "title", None)
    if title:
        title_str = str(title).strip()
        if title_str:
            return title_str
    return fallback


def _extract_abstract_structural(docling_doc) -> str:
    """
    Extract abstract using Docling's document item structure.
    Iterates labeled items instead of regex over exported markdown.

    Handles: "Abstract", "ABSTRACT", "A bstract" (OCR artifacts).
    """
    in_abstract = False
    parts: list[str] = []

    try:
        for item, _level in docling_doc.iterate_items():
            label = getattr(item, "label", None)
            label_str = str(label).lower() if label else ""

            # Detect section headers
            if "section_header" in label_str or "heading" in label_str:
                header_text = getattr(item, "text", "").strip().lower()
                # Fuzzy match for "abstract" (handles OCR variations)
                if _is_abstract_header(header_text):
                    in_abstract = True
                    continue
                if in_abstract:
                    # Hit next section → stop
                    break

            elif in_abstract:
                text = getattr(item, "text", "")
                if text and text.strip():
                    parts.append(text.strip())
    except Exception:
        # Fallback: try regex on markdown if structural iteration fails
        return _extract_abstract_regex(docling_doc)

    result = " ".join(parts)
    return result[:2000] if result else _extract_abstract_regex(docling_doc)


def _is_abstract_header(text: str) -> bool:
    """Fuzzy match for 'abstract' header, tolerant to OCR errors."""
    cleaned = re.sub(r"\s+", "", text)
    return cleaned in {"abstract", "abstracts", "абстракт"}


def _extract_abstract_regex(docling_doc) -> str:
    """Fallback: regex-based abstract extraction from markdown."""
    try:
        md = docling_doc.export_to_markdown()
    except Exception:
        return ""

    md_lower = md.lower()
    patterns = [
        r"#+\s*abstract\s*\n",
        r"\*\*abstract\*\*",
        r"^abstract\s*\n",
    ]

    for pattern in patterns:
        match = re.search(pattern, md_lower, re.MULTILINE)
        if match:
            start = match.end()
            next_section = re.search(r"\n#+\s|\n\*\*\d", md_lower[start:])
            end = start + (next_section.start() if next_section else 2000)
            return md[start:end].strip()[:2000]

    return ""


def _extract_sections_structural(docling_doc) -> list[str]:
    """
    Extract section structure from DoclingDocument items.
    Returns ordered list of section header strings.
    """
    sections: list[str] = []
    try:
        for item, _level in docling_doc.iterate_items():
            label = getattr(item, "label", None)
            label_str = str(label).lower() if label else ""
            if "section_header" in label_str or "heading" in label_str:
                text = getattr(item, "text", "").strip()
                if text and not _is_abstract_header(text.lower()):
                    sections.append(text)
    except Exception:
        # Fallback to regex
        try:
            md = docling_doc.export_to_markdown()
            sections = re.findall(r"^#{1,3}\s+(.+)$", md, re.MULTILINE)
        except Exception:
            pass

    return sections[:30]


_LABEL_PATTERNS: list[tuple[str, ChunkType]] = [
    ("table", ChunkType.TABLE),
    ("picture", ChunkType.FIGURE),
    ("figure", ChunkType.FIGURE),
    ("code", ChunkType.CODE),
    ("formula", ChunkType.EQUATION),
    ("equation", ChunkType.EQUATION),
]


def _classify_chunk_type(raw_chunk) -> ChunkType:
    """Determine chunk type from Docling item labels."""
    if not hasattr(raw_chunk, "meta") or raw_chunk.meta is None:
        return ChunkType.TEXT

    doc_items = getattr(raw_chunk.meta, "doc_items", [])
    for item in doc_items:
        label = getattr(item, "label", None)
        if label is None:
            continue
        label_str = str(label).lower()
        for pattern, chunk_type in _LABEL_PATTERNS:
            if pattern in label_str:
                return chunk_type

    return ChunkType.TEXT


def _extract_chunk_metadata(raw_chunk) -> tuple[str, list[int]]:
    """Extract section heading path and page numbers from chunk metadata."""
    section = ""
    page_numbers: list[int] = []

    if not hasattr(raw_chunk, "meta") or raw_chunk.meta is None:
        return section, page_numbers

    headings = getattr(raw_chunk.meta, "headings", [])
    if headings:
        section = " > ".join(headings)

    doc_items = getattr(raw_chunk.meta, "doc_items", [])
    for item in doc_items:
        if hasattr(item, "prov"):
            for prov in item.prov:
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    page_numbers.append(page_no)

    return section, sorted(set(page_numbers))


@dataclass
class ParseResult:
    """
    Result of document parsing.

    Separates Document (serializable, persistable) from DoclingDocument
    (in-memory only, needed for chunking).
    """

    document: Document
    docling_doc: Any | None = None  # DoclingDocument, not stored in Document.metadata


class DoclingParser:
    """
    Async document parser using IBM Docling for PDF → structured document conversion
    and hybrid hierarchical chunking.

    Usage:
        parser = DoclingParser(s3_client=s3, config=config)
        result = await parser.parse(storage_path)
        chunks, metrics = await parser.extract_chunks(result.document, result.docling_doc)
    """

    def __init__(
        self,
        s3_client: S3Client | None = None,
        config: DoclingParserConfig | None = None,
    ):
        self._s3 = s3_client or S3Client()
        self._config = config or DoclingParserConfig()

        self._converter: DocumentConverter | None = None
        self._chunker: HybridChunker | None = None
        self._tokenizer: HuggingFaceTokenizer | None = None

    @property
    def config(self) -> DoclingParserConfig:
        return self._config

    def _get_converter(self) -> DocumentConverter:
        if self._converter is None:
            cfg = self._config

            pipeline = PdfPipelineOptions()
            pipeline.do_ocr = cfg.do_ocr
            pipeline.do_table_structure = cfg.do_table_structure
            pipeline.table_structure_options = TableStructureOptions(do_cell_matching=True)
            pipeline.accelerator_options = AcceleratorOptions(
                num_threads=cfg.num_threads,
                device=cfg.device,
            )

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
            )

        return self._converter

    def _get_tokenizer(self) -> HuggingFaceTokenizer:
        if self._tokenizer is None:
            hf_tokenizer = AutoTokenizer.from_pretrained(self._config.embedding_model)
            self._tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=self._config.max_tokens,
            )
        return self._tokenizer

    def _get_chunker(self) -> HybridChunker:
        if self._chunker is None:
            self._chunker = HybridChunker(
                tokenizer=self._get_tokenizer(),
                merge_peers=self._config.merge_peers,
                serializer_provider=ScientificSerializerProvider(
                    image_placeholder=self._config.image_placeholder,
                ),
            )
        return self._chunker

    async def parse(self, storage_path: str) -> ParseResult:
        """
        Download PDF from S3, convert via Docling, extract metadata.

        Returns ParseResult containing:
          - Document (serializable, safe to persist)
          - DoclingDocument reference (in-memory only, pass to extract_chunks)
        """
        key = storage_path.replace(f"s3://{self._s3.bucket}/", "")
        filename = Path(key).name
        stem = Path(key).stem

        logger.info(f"Parsing (docling): key={key}")

        try:
            content_bytes = await asyncio.to_thread(self._s3.download_bytes, key)
        except Exception as e:
            logger.exception(f"S3 download failed: {key}")
            return ParseResult(
                document=Document(
                    id="",
                    source_path=key,
                    status=DocumentStatus.FAILED,
                    content_hash="",
                    metadata={"error": f"download_failed: {e}", "parser": "docling"},
                ),
            )

        content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
        doc_id = f"{stem}_{content_hash}"

        try:
            source = DocumentStream(name=filename, stream=BytesIO(content_bytes))
            converter = self._get_converter()
            result = await asyncio.to_thread(converter.convert, source)
            docling_doc = result.document

            title = _extract_title_structural(docling_doc, stem)
            abstract = _extract_abstract_structural(docling_doc)
            sections = _extract_sections_structural(docling_doc)

            markdown = docling_doc.export_to_markdown()
            md_key = f"processed/docling/markdown/{doc_id}.md"
            md_uri = await asyncio.to_thread(
                self._s3.upload_bytes,
                markdown.encode("utf-8"),
                md_key,
                "text/markdown",
            )

            document = Document(
                id=doc_id,
                source_path=key,
                title=title,
                abstract=abstract,
                status=DocumentStatus.PARSED,
                content_hash=content_hash,
                metadata={
                    "parser": "docling",
                    "config": {
                        "max_tokens": self._config.max_tokens,
                        "embedding_model": self._config.embedding_model,
                        "include_context": self._config.include_context,
                        "merge_peers": self._config.merge_peers,
                    },
                    "num_pages": getattr(docling_doc, "num_pages", 0),
                    "sections": sections,
                    "markdown_key": md_key,
                    "markdown_uri": md_uri,
                    # NOTE: _docling_doc is intentionally NOT stored here.
                    # Pass it explicitly via ParseResult.docling_doc.
                },
            )

            return ParseResult(document=document, docling_doc=docling_doc)

        except Exception as e:
            logger.exception(f"Docling conversion failed: {key}")
            return ParseResult(
                document=Document(
                    id=doc_id,
                    source_path=key,
                    status=DocumentStatus.FAILED,
                    content_hash=content_hash,
                    metadata={"error": str(e), "parser": "docling"},
                ),
            )

    async def extract_chunks(
        self,
        document: Document,
        docling_doc: Any | None = None,
    ) -> tuple[list[Chunk], ChunkingMetrics]:
        """
        Chunk a parsed document. Returns (chunks, metrics).

        Args:
            document: Parsed Document entity.
            docling_doc: DoclingDocument from ParseResult. If None, falls back
                         to markdown-based chunking.

        Returns:
            Tuple of (chunks, chunking_metrics).
        """
        if docling_doc is None:
            logger.warning(
                "No DoclingDocument provided, using fallback chunking",
                extra={"doc_id": document.id},
            )
            chunks = await self._fallback_chunking(document)
            metrics = compute_chunking_metrics(
                document,
                chunks,
                self._config,
                used_fallback=True,
            )
            return chunks, metrics

        chunker = self._get_chunker()
        tokenizer = self._get_tokenizer()

        try:
            chunk_iter = await asyncio.to_thread(chunker.chunk, dl_doc=docling_doc)
            raw_chunks = list(chunk_iter)
        except Exception as e:
            logger.exception(f"Chunking failed: {e}")
            logger.exception(
                "Hybrid chunking failed, using fallback", extra={"doc_id": document.id}
            )
            chunks = await self._fallback_chunking(document)
            metrics = compute_chunking_metrics(
                document,
                chunks,
                self._config,
                used_fallback=True,
            )
            return chunks, metrics

        chunks: list[Chunk] = []
        tokenizer = self._get_tokenizer()

        try:
            chunk_iter = await asyncio.to_thread(chunker.chunk, dl_doc=docling_doc)
            raw_chunks = list(chunk_iter)
        except Exception:
            logger.exception(
                "Hybrid chunking failed, using fallback", extra={"doc_id": document.id}
            )
            chunks = await self._fallback_chunking(document)
            metrics = compute_chunking_metrics(
                document,
                chunks,
                self._config,
                used_fallback=True,
            )
            return chunks, metrics

        chunks: list[Chunk] = []
        context_overheads: list[float] = []

        for idx, raw_chunk in enumerate(raw_chunks):
            # Get both raw and contextualized text for metrics
            raw_text = raw_chunk.text
            if self._config.include_context:
                text = chunker.contextualize(chunk=raw_chunk)
            else:
                text = raw_text

            token_count = tokenizer.count_tokens(text)

            # Compute context overhead
            if self._config.include_context and raw_text:
                raw_tokens = tokenizer.count_tokens(raw_text)
                overhead = (token_count - raw_tokens) / token_count if token_count > 0 else 0.0
                context_overheads.append(overhead)
            else:
                context_overheads.append(0.0)

            # Validate chunk
            if token_count < self._config.min_chunk_tokens and self._config.filter_empty_chunks:
                logger.debug(
                    "Dropping degenerate chunk",
                    extra={"doc_id": document.id, "idx": idx, "tokens": token_count},
                )
                continue

            chunk_type = _classify_chunk_type(raw_chunk)
            section, page_numbers = _extract_chunk_metadata(raw_chunk)

            chunk = Chunk(
                id=f"{document.id}_c{idx:04d}",
                document_id=document.id,
                text=text,
                token_count=token_count,
                chunk_index=idx,
                page_numbers=page_numbers,
                section=section,
                metadata={
                    "source_key": document.source_path,
                    "chunk_type": chunk_type.value,
                    "contextualized": self._config.include_context,
                    "is_degenerate": token_count < self._config.min_chunk_tokens,
                    "is_oversized": token_count > self._config.max_tokens,
                },
            )
            chunks.append(chunk)

        metrics = compute_chunking_metrics(
            document=document,
            chunks=chunks,
            config=self._config,
            context_overheads=context_overheads,
            used_fallback=False,
        )

        logger.info(
            "Chunking complete",
            extra={
                "doc_id": document.id,
                "total_chunks": metrics.total_chunks,
                "mean_tokens": metrics.mean_tokens,
                "empty": metrics.empty_chunk_count,
                "oversized": metrics.oversized_chunk_count,
                "types": metrics.type_distribution,
            },
        )

        return chunks, metrics

    def _extract_title(self, docling_doc, fallback: str) -> str:
        title = getattr(docling_doc, "title", None)
        if title:
            return str(title).strip()
        return fallback

    def _extract_abstract(self, docling_doc) -> str:
        """Extract abstract using document structure, not regex."""
        in_abstract = False
        parts = []
        for item, _level in docling_doc.iterate_items():
            label = getattr(item, "label", None)
            if label and "section_header" in str(label).lower():
                header_text = getattr(item, "text", "").lower().strip()
                if "abstract" in header_text:
                    in_abstract = True
                    continue
                if in_abstract:
                    break  # Next section started
            elif in_abstract:
                text = getattr(item, "text", "")
                if text:
                    parts.append(text)
        return " ".join(parts)[:1500]

    def _extract_section_structure(self, docling_doc) -> list[str]:
        md = docling_doc.export_to_markdown()

        sections = re.findall(r"^##\s+(.+)$", md, re.MULTILINE)
        return sections[:20]

    async def _fallback_chunking(self, document: Document) -> list[Chunk]:
        """
        Simple paragraph-based chunking from stored markdown.
        Used when DoclingDocument is unavailable or hybrid chunking fails.
        """
        md_key = document.metadata.get("markdown_key")
        if not md_key:
            logger.warning("No markdown_key for fallback", extra={"doc_id": document.id})
            return []

        try:
            md_bytes = await asyncio.to_thread(self._s3.download_bytes, md_key)
            text = md_bytes.decode("utf-8")
        except Exception:
            logger.exception(
                "Failed to download markdown for fallback", extra={"doc_id": document.id}
            )
            return []

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        tokenizer = self._get_tokenizer()

        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_tokens = 0
        idx = 0

        for para in paragraphs:
            pt = tokenizer.count_tokens(para)

            if buf and (buf_tokens + pt > self._config.max_tokens):
                chunks.append(
                    Chunk(
                        id=f"{document.id}_c{idx:04d}",
                        document_id=document.id,
                        text="\n\n".join(buf),
                        token_count=buf_tokens,
                        chunk_index=idx,
                        metadata={
                            "source_key": document.source_path,
                            "chunk_type": ChunkType.TEXT.value,
                            "fallback": True,
                            "contextualized": False,
                            "is_degenerate": buf_tokens < self._config.min_chunk_tokens,
                            "is_oversized": False,
                        },
                    )
                )
                idx += 1
                buf = []
                buf_tokens = 0

            buf.append(para)
            buf_tokens += pt

        if buf:
            chunks.append(
                Chunk(
                    id=f"{document.id}_c{idx:04d}",
                    document_id=document.id,
                    text="\n\n".join(buf),
                    token_count=buf_tokens,
                    chunk_index=idx,
                    metadata={
                        "source_key": document.source_path,
                        "chunk_type": ChunkType.TEXT.value,
                        "fallback": True,
                        "contextualized": False,
                        "is_degenerate": buf_tokens < self._config.min_chunk_tokens,
                        "is_oversized": False,
                    },
                )
            )

        return chunks


async def parse_and_chunk(
    parser: DoclingParser,
    storage_path: str,
) -> tuple[Document, list[Chunk], ChunkingMetrics]:
    """
    End-to-end: download → parse → chunk → metrics.

    Usage:
        parser = create_docling_parser(s3_client=s3)
        doc, chunks, metrics = await parse_and_chunk(parser, "s3://bucket/paper.pdf")
    """
    result = await parser.parse(storage_path)

    if result.document.status == DocumentStatus.FAILED:
        empty_metrics = ChunkingMetrics(
            document_id=result.document.id,
            total_chunks=0,
            total_tokens=0,
        )
        return result.document, [], empty_metrics

    chunks, metrics = await parser.extract_chunks(
        document=result.document,
        docling_doc=result.docling_doc,
    )
    return result.document, chunks, metrics


def create_docling_parser(  # noqa: PLR0913
    s3_client: S3Client | None = None,
    embedding_model: str = "Qwen/Qwen3-4B",
    max_tokens: int = 512,
    include_context: bool = True,
    merge_peers: bool = True,
    filter_empty_chunks: bool = False,
) -> DoclingParser:
    """
    Factory for DoclingParser with sensible defaults for scientific RAG.

    Args:
        s3_client: S3 client for document storage.
        embedding_model: HuggingFace model ID for tokenization.
                         MUST match the embedding model in retrieval pipeline.
        max_tokens: Target chunk size in tokens.
        include_context: Prepend section headings to chunks.
        merge_peers: Merge adjacent small chunks.
        filter_empty_chunks: Drop chunks below min_chunk_tokens threshold.
    """
    config = DoclingParserConfig(
        max_tokens=max_tokens,
        embedding_model=embedding_model,
        include_context=include_context,
        merge_peers=merge_peers,
        do_ocr=False,
        filter_empty_chunks=filter_empty_chunks,
    )
    return DoclingParser(s3_client=s3_client, config=config)
