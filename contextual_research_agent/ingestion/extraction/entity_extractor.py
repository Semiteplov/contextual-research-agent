from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from contextual_research_agent.agent.llm import LlamaCppProvider
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.ingestion.domain.entities import Chunk

logger = get_logger(__name__)


@dataclass
class ExtractedEntity:
    """A single extracted entity."""

    name: str
    entity_type: str  # method, dataset, task, metric, model
    confidence: float = 1.0
    evidence: str = ""  # text snippet where entity was found
    section_type: str = ""  # section where found


@dataclass
class EntityEdge:
    """A paper → entity relationship."""

    paper_id: str
    entity_name: str
    entity_type: str
    relation: str  # uses_method, uses_dataset, targets_task, reports_metric, uses_model
    confidence: float = 1.0
    evidence: str = ""
    section_type: str = ""


@dataclass
class EntityExtractionResult:
    """Result of entity extraction for a single document."""

    document_id: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    edges: list[EntityEdge] = field(default_factory=list)
    extraction_ms: float = 0.0
    llm_calls: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        entity_types: dict[str, int] = {}
        for e in self.entities:
            entity_types[e.entity_type] = entity_types.get(e.entity_type, 0) + 1

        return {
            "document_id": self.document_id,
            "total_entities": len(self.entities),
            "total_edges": len(self.edges),
            "entity_types": entity_types,
            "extraction_ms": round(self.extraction_ms, 1),
            "llm_calls": self.llm_calls,
            "error": self.error,
        }


_ENTITY_TYPES = ["method", "dataset", "task", "metric", "model"]

_RELATION_MAP = {
    "method": "uses_method",
    "dataset": "uses_dataset",
    "task": "targets_task",
    "metric": "reports_metric",
    "model": "uses_model",
}

_SYSTEM_PROMPT = """Extract key scientific entities from text. Return ONLY valid JSON, no other text.

Types: method, dataset, task, metric, model.

Rules:
- Only explicitly mentioned entities
- Canonical names (e.g., "LoRA" not "Low-Rank Adaptation")
- No generic terms ("neural network" too generic, "Transformer" ok)
- Confidence: 1.0 = named, 0.7 = implied

{"entities": [{"name": "...", "type": "...", "confidence": 0.9}]}
/no_think"""

_SECTION_HINTS = {
    "method": "Focus on: proposed methods, algorithms, architectural components, training techniques, loss functions.",
    "experiments": "Focus on: datasets used, baseline models compared, evaluation metrics, implementation details.",
    "results": "Focus on: metrics reported, models compared, datasets evaluated on.",
    "introduction": "Focus on: task definitions, key methods mentioned, problem framing.",
    "related_work": "Focus on: prior methods, competing approaches, referenced models and datasets.",
    "background": "Focus on: foundational methods, mathematical frameworks, prerequisite concepts.",
}


def _build_extraction_prompt(text: str, section_type: str) -> str:
    """Build the user prompt for entity extraction."""
    hint = _SECTION_HINTS.get(section_type, "")
    section_label = section_type if section_type else "unknown"

    prompt = f"Section type: {section_label}\n"
    if hint:
        prompt += f"{hint}\n"
    prompt += f"\nText:\n{text[:1500]}"
    return prompt


class LLMClient:
    """
    Interface for LLM calls.
    """

    async def generate(self, system: str, user: str) -> str:
        """Generate text from system + user prompt. Returns raw string."""
        raise NotImplementedError


class OllamaProviderAdapter(LLMClient):
    """
    Adapter: wraps existing OllamaProvider to match LLMClient interface.
    """

    def __init__(self, provider):
        """
        Args:
            provider: OllamaProvider instance from contextual_research_agent.agent.llm
        """
        self._provider = provider

    async def generate(self, system: str, user: str) -> str:
        result = await self._provider.generate(
            prompt=user,
            system_prompt=system,
            temperature=0.1,
            max_tokens=1024,
        )
        return result.text


class LlamaCppProviderAdapter(LLMClient):
    def __init__(self, provider: LlamaCppProvider):
        self._provider = provider

    async def generate(self, system: str, user: str) -> str:
        result = await self._provider.generate(
            prompt=user,
            system_prompt=system,
            temperature=0.1,
            max_tokens=1024,
        )
        return result.text


def _parse_llm_response(raw: str) -> list[dict[str, Any]]:  # noqa: C901
    """
    Parse LLM JSON response into entity dicts.
    """
    text = raw.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.debug("Failed to parse LLM response: %s", text[:200])
                return []
        else:
            return []

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []

    valid: list[dict[str, Any]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        etype = str(e.get("type", "")).strip().lower()
        confidence = float(e.get("confidence", 0.8))

        if not name or len(name) < 2:  # noqa: PLR2004
            continue
        if etype not in _ENTITY_TYPES:
            continue
        if confidence < 0.3:  # noqa: PLR2004
            continue

        valid.append(
            {
                "name": name,
                "type": etype,
                "confidence": min(confidence, 1.0),
            }
        )

    return valid


def _normalize_entity_name(name: str) -> str:
    """Normalize entity name for deduplication."""
    n = name.strip().lower()
    n = n.rstrip(".,;:")
    return re.sub(r"\s+", " ", n)


def _deduplicate_entities(
    entities: list[ExtractedEntity],
) -> list[ExtractedEntity]:
    """
    Deduplicate entities by (normalized_name, entity_type).
    """
    seen: dict[tuple[str, str], ExtractedEntity] = {}
    for e in entities:
        key = (_normalize_entity_name(e.name), e.entity_type)
        existing = seen.get(key)
        if existing is None or e.confidence > existing.confidence:
            seen[key] = e
    return list(seen.values())


class EntityExtractor:
    """
    LLM-based scientific entity extractor.

    Groups chunks by section_type, sends batched text to LLM,
    parses structured JSON output, deduplicates entities.

    Args:
        llm_client: LLM backend (OllamaClient, VLLMClient, etc.)
        max_chunks_per_call: Max chunks to concatenate per LLM call.
        skip_section_types: Section types to skip (e.g., references, appendix).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_chunks_per_call: int = 5,
        skip_section_types: set[str] | None = None,
    ):
        self._llm = llm_client
        self._max_chunks_per_call = max_chunks_per_call
        self._skip_sections = skip_section_types or {
            "references",
            "appendix",
            "ethics",
            "title",
            "unknown",
        }

    @staticmethod
    def _select_representative_chunks(
        chunks: list[Chunk],
        max_per_section: int = 3,
    ) -> list[Chunk]:
        """Select most informative chunks per section."""
        text_chunks = [
            c for c in chunks if c.metadata.get("chunk_type", "text") in ("text", "table")
        ]
        text_chunks.sort(key=lambda c: len(c.text), reverse=True)
        return text_chunks[:max_per_section]

    async def extract(
        self,
        document_id: str,
        arxiv_id: str,
        chunks: list[Chunk],
    ) -> EntityExtractionResult:
        """
        Extract entities from document chunks.
        """

        t0 = time.perf_counter()

        all_entities: list[ExtractedEntity] = []
        llm_calls = 0
        error_msg = None

        try:
            groups = self._group_chunks_by_section(chunks)

            tasks: list[tuple[str, str, str]] = []

            for section_type, section_chunks in groups.items():
                if section_type in self._skip_sections:
                    continue

                selected = self._select_representative_chunks(
                    section_chunks,
                    max_per_section=3,
                )
                if not selected:
                    continue

                combined_text = "\n\n".join(c.text for c in selected)
                prompt = _build_extraction_prompt(combined_text, section_type)
                tasks.append((section_type, combined_text, prompt))

            sem = asyncio.Semaphore(4)

            async def _run(section_type: str, combined_text: str, prompt: str):
                async with sem:
                    raw_response = await self._llm.generate(_SYSTEM_PROMPT, prompt)
                    return section_type, combined_text, raw_response

            results = await asyncio.gather(
                *[_run(st, ct, p) for st, ct, p in tasks],
                return_exceptions=True,
            )

            for idx, result in enumerate(results):
                llm_calls += 1

                if isinstance(result, BaseException):
                    section_type = tasks[idx][0]
                    logger.warning(
                        "LLM extraction failed for %s/%s: %s",
                        document_id,
                        section_type,
                        result,
                    )
                    continue

                assert isinstance(result, tuple)
                section_type, combined_text, raw_response = result

                try:
                    parsed = _parse_llm_response(raw_response)
                    for entity_dict in parsed:
                        all_entities.append(
                            ExtractedEntity(
                                name=entity_dict["name"],
                                entity_type=entity_dict["type"],
                                confidence=entity_dict["confidence"],
                                evidence=combined_text[:200],
                                section_type=section_type,
                            )
                        )
                except Exception as e:
                    logger.warning(
                        "LLM extraction failed for %s/%s: %s",
                        document_id,
                        section_type,
                        e,
                    )

        except Exception as e:
            error_msg = str(e)
            logger.exception("Entity extraction failed for %s", document_id)

        unique_entities = _deduplicate_entities(all_entities)

        edges = [
            EntityEdge(
                paper_id=arxiv_id,
                entity_name=e.name,
                entity_type=e.entity_type,
                relation=_RELATION_MAP.get(e.entity_type, "related_to"),
                confidence=e.confidence,
                evidence=e.evidence[:500],
                section_type=e.section_type,
            )
            for e in unique_entities
        ]

        extraction_ms = (time.perf_counter() - t0) * 1000

        result = EntityExtractionResult(
            document_id=document_id,
            entities=unique_entities,
            edges=edges,
            extraction_ms=extraction_ms,
            llm_calls=llm_calls,
            error=error_msg,
        )

        logger.info(
            "Entity extraction: %d entities, %d edges, %d LLM calls (%.0fms) for %s",
            len(unique_entities),
            len(edges),
            llm_calls,
            extraction_ms,
            document_id,
        )

        return result

    @staticmethod
    def _group_chunks_by_section(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        """Group chunks by section_type."""
        groups: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            st = chunk.metadata.get("section_type", "unknown")
            if st not in groups:
                groups[st] = []
            groups[st].append(chunk)
        return groups


def store_entity_results(
    graph_repo,
    result: EntityExtractionResult,
) -> int:
    """
    Store entity extraction results in PostgreSQL.

    1. Upsert entities → get entity_ids
    2. Store paper_entity_edges

    Returns total edges stored.
    """
    if not result.edges:
        return 0

    entity_ids: dict[tuple[str, str], int] = {}
    for entity in result.entities:
        eid = graph_repo.upsert_entity(
            name=entity.name,
            entity_type=entity.entity_type,
            source="llm_extracted",
        )
        entity_ids[(_normalize_entity_name(entity.name), entity.entity_type)] = eid

    edge_records: list[dict[str, Any]] = []
    for edge in result.edges:
        key = (_normalize_entity_name(edge.entity_name), edge.entity_type)
        eid = entity_ids.get(key)
        if eid is None:
            continue

        edge_records.append(
            {
                "entity_id": eid,
                "relation": edge.relation,
                "confidence": edge.confidence,
                "evidence": edge.evidence,
                "section_type": edge.section_type,
                "extraction_method": "llm",
            }
        )

    if edge_records:
        return graph_repo.store_paper_entity_edges(
            paper_id=result.edges[0].paper_id,
            edges=edge_records,
        )

    return 0
