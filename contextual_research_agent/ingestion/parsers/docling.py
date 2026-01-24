import asyncio
import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
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
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.io import DocumentStream
from transformers import AutoTokenizer

from contextual_research_agent.common import logging
from contextual_research_agent.data.storage.s3_client import S3Client
from contextual_research_agent.ingestion.domain.entities import Chunk, Document
from contextual_research_agent.ingestion.domain.types import DocumentStatus
from contextual_research_agent.ingestion.parsers.base import DocumentParser

logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class DoclingParserConfig:
    max_tokens: int = 512
    tokenizer_model: str = "Qwen/Qwen3-4B"
    merge_peers: bool = True
    include_context: bool = True
    do_ocr: bool = False
    do_table_structure: bool = True
    num_threads: int = 4
    device: AcceleratorDevice = AcceleratorDevice.AUTO


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


class DoclingParser(DocumentParser):
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
            hf_tokenizer = AutoTokenizer.from_pretrained(self._config.tokenizer_model)
            self._tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=self._config.max_tokens,
            )
        return self._tokenizer

    def _get_chunker(self) -> HybridChunker:
        if self._chunker is None:
            tokenizer = self._get_tokenizer()
            serializer = MDTableSerializerProvider()

            self._chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=self._config.merge_peers,
                serializer_provider=serializer,
            )

        return self._chunker

    async def parse(self, storage_path: str) -> Document:
        key = storage_path.replace(f"s3://{self._s3.bucket}/", "")
        filename = Path(key).name
        stem = Path(key).stem

        logger.info(f"Parsing (docling): key={key}")

        try:
            content_bytes = await asyncio.to_thread(self._s3.download_bytes, key)
        except Exception as e:
            logger.exception(f"Failed to download: {key}")
            return Document(
                id="",
                source_path=key,
                status=DocumentStatus.FAILED,
                content_hash="",
                metadata={"error": f"download_failed: {e}", "parser": "docling"},
            )

        content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
        doc_id = f"{stem}_{content_hash}"

        try:
            source = DocumentStream(name=filename, stream=BytesIO(content_bytes))
            converter = self._get_converter()
            result = await asyncio.to_thread(converter.convert, source)
            docling_doc = result.document

            title = self._extract_title(docling_doc, stem)
            abstract = self._extract_abstract(docling_doc)
            sections = self._extract_section_structure(docling_doc)

            markdown = docling_doc.export_to_markdown()
            md_key = f"processed/docling/markdown/{doc_id}.md"
            md_uri = await asyncio.to_thread(
                self._s3.upload_bytes,
                markdown.encode("utf-8"),
                md_key,
                "text/markdown",
            )

            return Document(
                id=doc_id,
                source_path=key,
                title=title,
                abstract=abstract,
                status=DocumentStatus.PARSED,
                content_hash=content_hash,
                metadata={
                    "parser": "docling_arxiv",
                    "config": {
                        "max_tokens": self._config.max_tokens,
                        "tokenizer_model": self._config.tokenizer_model,
                        "include_context": self._config.include_context,
                    },
                    "num_pages": getattr(docling_doc, "num_pages", 0),
                    "sections": sections,
                    "markdown_key": md_key,
                    "markdown_uri": md_uri,
                    "_docling_doc": docling_doc,
                },
            )

        except Exception as e:
            logger.exception(f"Parse failed: {key}")
            return Document(
                id=doc_id,
                source_path=key,
                status=DocumentStatus.FAILED,
                content_hash=content_hash,
                metadata={"error": str(e)},
            )

    async def extract_chunks(  # noqa: C901, PLR0912
        self,
        document: Document,
    ) -> list[Chunk]:
        docling_doc = document.metadata.get("_docling_doc")
        if docling_doc is None:
            logger.warning(f"No docling document for {document.id}, using simple fallback")
            return await self._fallback_chunking(document)

        chunker = self._get_chunker()

        try:
            chunk_iter = await asyncio.to_thread(chunker.chunk, dl_doc=docling_doc)
            raw_chunks = list(chunk_iter)
        except Exception as e:
            logger.exception(f"Chunking failed: {e}")
            return await self._fallback_chunking(document)

        chunks: list[Chunk] = []
        tokenizer = self._get_tokenizer()

        for idx, raw_chunk in enumerate(raw_chunks):
            if self._config.include_context:
                text = chunker.contextualize(chunk=raw_chunk)
            else:
                text = raw_chunk.text

            token_count = tokenizer.count_tokens(text)

            section = ""
            page_numbers = []
            chunk_type = "text"  # text, table, figure, code

            if hasattr(raw_chunk, "meta") and raw_chunk.meta:
                headings = getattr(raw_chunk.meta, "headings", [])
                if headings:
                    section = " > ".join(headings)

                doc_items = getattr(raw_chunk.meta, "doc_items", [])
                for item in doc_items:
                    if hasattr(item, "prov"):
                        for prov in item.prov:
                            if hasattr(prov, "page_no"):
                                page_numbers.append(prov.page_no)

                    label = getattr(item, "label", None)
                    if label:
                        label_str = str(label).lower()
                        if "table" in label_str:
                            chunk_type = "table"
                        elif "picture" in label_str or "figure" in label_str:
                            chunk_type = "figure"
                        elif "code" in label_str:
                            chunk_type = "code"

            chunk = Chunk(
                id=f"{document.id}_c{idx:04d}",
                document_id=document.id,
                text=text,
                token_count=token_count,
                chunk_index=idx,
                page_numbers=sorted(set(page_numbers)) if page_numbers else [],
                section=section,
                metadata={
                    "source_key": document.source_path,
                    "chunk_type": chunk_type,
                    "contextualized": self._config.include_context,
                },
            )
            chunks.append(chunk)

        type_counts = {}
        for c in chunks:
            t = c.metadata.get("chunk_type", "text")
            type_counts[t] = type_counts.get(t, 0) + 1

        logger.info(f"Extracted {len(chunks)} chunks from {document.id}: {type_counts}")

        return chunks

    def _extract_title(self, docling_doc, fallback: str) -> str:
        title = getattr(docling_doc, "title", None)
        if title:
            return str(title).strip()
        return fallback

    def _extract_abstract(self, docling_doc) -> str:
        md = docling_doc.export_to_markdown()
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
                return md[start:end].strip()[:1500]

        return ""

    def _extract_section_structure(self, docling_doc) -> list[str]:
        md = docling_doc.export_to_markdown()

        sections = re.findall(r"^##\s+(.+)$", md, re.MULTILINE)
        return sections[:20]

    async def _fallback_chunking(self, document: Document) -> list[Chunk]:
        md_key = document.metadata.get("markdown_key")
        if not md_key:
            return []

        try:
            md_bytes = await asyncio.to_thread(self._s3.download_bytes, md_key)
            text = md_bytes.decode("utf-8")
        except Exception:
            return []

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        tokenizer = self._get_tokenizer()

        chunks = []
        buf = []
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
                        metadata={"fallback": True},
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
                    metadata={"fallback": True},
                )
            )

        return chunks


def create_docling_parser(
    s3_client: S3Client | None = None,
    embedding_model: str = "Qwen/Qwen3-4B",
    max_tokens: int = 512,
    include_section_context: bool = True,
) -> DocumentParser:
    config = DoclingParserConfig(
        max_tokens=max_tokens,
        tokenizer_model=embedding_model,
        include_context=include_section_context,
        do_ocr=False,
    )

    return DoclingParser(s3_client=s3_client, config=config)
