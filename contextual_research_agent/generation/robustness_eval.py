from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import mlflow

from contextual_research_agent.agent.llm import create_llm_provider
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.common.settings import get_settings
from contextual_research_agent.generation.config import CognitiveMode, GenerationConfig
from contextual_research_agent.generation.openrouter_provider import OpenRouterProvider
from contextual_research_agent.generation.pipeline import GenerationPipeline, RAGResponse
from contextual_research_agent.ingestion.vectorstores.qdrant_store import (
    QdrantStore,
    create_qdrant_store,
)

logger = get_logger(__name__)


CHECKPOINT_INTERVAL = 25

REFUSAL_PATTERNS = [
    r"(?i)the provided (sources|context|passages?) do not contain",
    r"(?i)insufficient information",
    r"(?i)not enough information",
    r"(?i)cannot (be )?answer(ed)?.*based on",
    r"(?i)no relevant information",
    r"(?i)the context does not (provide|contain|mention|include)",
]


class RobustnessScenario(str, Enum):
    EMPTY = "empty"
    RANDOM = "random"
    PARTIAL = "partial"


@dataclass
class RobustnessQueryResult:
    query: str
    category: str
    scenario: str
    answer: str
    is_refusal: bool
    mode: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_chunks_count: int = 0
    context_document_id: str = ""
    context_section_types: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    relevance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "query": self.query,
            "category": self.category,
            "scenario": self.scenario,
            "answer": self.answer,
            "is_refusal": self.is_refusal,
            "mode": self.mode,
            "latency_ms": round(self.latency_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "context_chunks_count": self.context_chunks_count,
        }
        if self.context_document_id:
            d["context_document_id"] = self.context_document_id
        if self.context_section_types:
            d["context_section_types"] = self.context_section_types
        if self.faithfulness is not None:
            d["faithfulness"] = self.faithfulness
        if self.relevance is not None:
            d["relevance"] = self.relevance
        return d


@dataclass
class ScenarioMetrics:
    scenario: str
    total_queries: int
    refusal_count: int
    refusal_rate: float
    non_refusal_count: int
    mean_latency_ms: float
    mean_faithfulness: float | None = None
    mean_relevance: float | None = None
    faithfulness_pass_rate: float | None = None  # % with score >= 4
    per_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "scenario": self.scenario,
            "total_queries": self.total_queries,
            "refusal_count": self.refusal_count,
            "refusal_rate": round(self.refusal_rate, 4),
            "non_refusal_count": self.non_refusal_count,
            "mean_latency_ms": round(self.mean_latency_ms, 1),
        }
        if self.mean_faithfulness is not None:
            d["mean_faithfulness"] = round(self.mean_faithfulness, 3)
        if self.mean_relevance is not None:
            d["mean_relevance"] = round(self.mean_relevance, 3)
        if self.faithfulness_pass_rate is not None:
            d["faithfulness_pass_rate"] = round(self.faithfulness_pass_rate, 4)
        if self.per_category:
            d["per_category"] = self.per_category
        return d


_COMPILED_REFUSAL_PATTERNS = [re.compile(p) for p in REFUSAL_PATTERNS]


def is_refusal(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    return any(pattern.search(answer) for pattern in _COMPILED_REFUSAL_PATTERNS)


def _format_chunks_as_context(
    chunks: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    for c in chunks:
        chunk_id = c.get("chunk_id", "unknown")
        section_type = c.get("section_type", "unknown")
        score = c.get("score", 0.0)
        text = c.get("text", "")
        parts.append(f"[{chunk_id}] (section: {section_type}, score: {score:.3f})\n{text}")
    return "\n\n---\n\n".join(parts)


def build_empty_context() -> str:
    return ""


def build_random_context(
    all_chunk_ids: list[str],
    chunk_id_to_data: dict[str, dict[str, Any]],
    relevant_ids: set[str],
    n_chunks: int = 10,
    rng: random.Random | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    _rng = rng or random.Random()

    candidates = [cid for cid in all_chunk_ids if cid not in relevant_ids]

    if len(candidates) < n_chunks:
        logger.warning(
            "Only %d non-relevant chunks available (requested %d)",
            len(candidates),
            n_chunks,
        )
        n_chunks = len(candidates)

    sampled_ids = _rng.sample(candidates, n_chunks)
    sampled_chunks = []
    for cid in sampled_ids:
        data = chunk_id_to_data.get(cid)
        if data:
            sampled_chunks.append(data)

    context = _format_chunks_as_context(sampled_chunks)
    return context, sampled_chunks


def build_partial_context(
    document_id: str,
    doc_chunks: list[dict[str, Any]],
    relevant_ids: set[str],
    relevant_sections: list[str] | None = None,
    n_chunks: int = 10,
    rng: random.Random | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    _rng = rng or random.Random()

    # Exclude directly relevant chunks
    non_relevant = [c for c in doc_chunks if c["chunk_id"] not in relevant_ids]

    if not non_relevant:
        # All chunks in document are relevant — can't build partial context
        return "", []

    # Prefer chunks from non-relevant sections
    if relevant_sections:
        rel_sections_set = set(relevant_sections)
        wrong_section = [
            c for c in non_relevant if c.get("section_type", "unknown") not in rel_sections_set
        ]
        # If we have enough from wrong sections, use those
        if len(wrong_section) >= n_chunks:
            non_relevant = wrong_section

    _rng.shuffle(non_relevant)
    selected = non_relevant[:n_chunks]

    context = _format_chunks_as_context(selected)
    return context, selected


async def load_all_chunks_data(
    store: QdrantStore,
) -> dict[str, dict[str, Any]]:
    """Load all chunk IDs and their metadata from Qdrant.

    Returns:
        Dict mapping chunk_id → {chunk_id, text, section_type, document_id}
    """
    logger.info("Loading all chunk data from Qdrant...")

    # Use scroll to get all chunks with full payload
    all_data: dict[str, dict[str, Any]] = {}
    offset = None
    batch_size = 256

    while True:
        points, next_offset = await asyncio.to_thread(
            lambda off=offset: store._client.scroll(
                collection_name=store.collection_name,
                limit=batch_size,
                offset=off,
                with_payload=True,
                with_vectors=False,
            )
        )

        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id", "")
            if chunk_id:
                section_type = "unknown"
                metadata = payload.get("metadata", {})
                if isinstance(metadata, dict):
                    section_type = metadata.get("section_type", "unknown")

                all_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": payload.get("text", ""),
                    "section_type": section_type,
                    "document_id": payload.get("document_id", ""),
                    "score": 0.5,
                }

        if next_offset is None or len(points) == 0:
            break
        offset = next_offset

    logger.info("Loaded %d chunks from Qdrant", len(all_data))
    return all_data


def _save_checkpoint(
    results: list[RobustnessQueryResult],
    checkpoint_path: Path,
    scenario: str,
) -> None:
    """Save intermediate results for crash recovery."""
    data = {
        "scenario": scenario,
        "completed": len(results),
        "results": [r.to_dict() for r in results],
    }
    checkpoint_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Checkpoint saved: %d results → %s", len(results), checkpoint_path)


def _load_checkpoint(
    checkpoint_path: Path,
    scenario: str,
) -> list[RobustnessQueryResult] | None:
    """Load checkpoint if it exists and matches the scenario."""
    if not checkpoint_path.exists():
        return None

    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if data.get("scenario") != scenario:
            logger.warning(
                "Checkpoint scenario mismatch: %s vs %s, ignoring",
                data.get("scenario"),
                scenario,
            )
            return None

        results = []
        for r in data["results"]:
            results.append(
                RobustnessQueryResult(
                    query=r["query"],
                    category=r["category"],
                    scenario=r["scenario"],
                    answer=r["answer"],
                    is_refusal=r["is_refusal"],
                    mode=r["mode"],
                    latency_ms=r["latency_ms"],
                    prompt_tokens=r.get("prompt_tokens", 0),
                    completion_tokens=r.get("completion_tokens", 0),
                    context_chunks_count=r.get("context_chunks_count", 0),
                    context_document_id=r.get("context_document_id", ""),
                    context_section_types=r.get("context_section_types", []),
                    faithfulness=r.get("faithfulness"),
                    relevance=r.get("relevance"),
                )
            )
        logger.info("Loaded checkpoint: %d results from %s", len(results), checkpoint_path)
        return results

    except Exception as e:
        logger.warning("Failed to load checkpoint: %s", e)
        return None


def compute_scenario_metrics(
    results: list[RobustnessQueryResult],
    scenario: str,
) -> ScenarioMetrics:
    """Compute aggregated metrics for a scenario."""
    total = len(results)
    if total == 0:
        return ScenarioMetrics(
            scenario=scenario,
            total_queries=0,
            refusal_count=0,
            refusal_rate=0.0,
            non_refusal_count=0,
            mean_latency_ms=0.0,
        )

    refusal_count = sum(1 for r in results if r.is_refusal)
    non_refusal_count = total - refusal_count
    latencies = [r.latency_ms for r in results]

    non_refusal_faithfulness = [
        r.faithfulness for r in results if not r.is_refusal and r.faithfulness is not None
    ]
    non_refusal_relevance = [
        r.relevance for r in results if not r.is_refusal and r.relevance is not None
    ]

    mean_faith = (
        sum(non_refusal_faithfulness) / len(non_refusal_faithfulness)
        if non_refusal_faithfulness
        else None
    )
    mean_rel = (
        sum(non_refusal_relevance) / len(non_refusal_relevance) if non_refusal_relevance else None
    )
    faith_pass = (
        sum(1 for f in non_refusal_faithfulness if f >= 4) / len(non_refusal_faithfulness)
        if non_refusal_faithfulness
        else None
    )

    categories: dict[str, list[RobustnessQueryResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    per_cat: dict[str, dict[str, Any]] = {}
    for cat, cat_results in sorted(categories.items()):
        cat_total = len(cat_results)
        cat_refusals = sum(1 for r in cat_results if r.is_refusal)
        per_cat[cat] = {
            "total": cat_total,
            "refusal_count": cat_refusals,
            "refusal_rate": round(cat_refusals / cat_total, 4) if cat_total > 0 else 0.0,
        }

    return ScenarioMetrics(
        scenario=scenario,
        total_queries=total,
        refusal_count=refusal_count,
        refusal_rate=refusal_count / total,
        non_refusal_count=non_refusal_count,
        mean_latency_ms=sum(latencies) / len(latencies),
        mean_faithfulness=mean_faith,
        mean_relevance=mean_rel,
        faithfulness_pass_rate=faith_pass,
        per_category=per_cat,
    )


async def run_robustness_scenario(
    scenario: RobustnessScenario,
    eval_set: list[dict[str, Any]],
    gen_pipeline: GenerationPipeline,
    all_chunks_data: dict[str, dict[str, Any]],
    output_path: Path,
    max_queries: int | None = None,
    seed: int = 42,
    judge_provider: Any | None = None,
    n_context_chunks: int = 10,
) -> ScenarioMetrics:
    """Run a single robustness scenario over the eval set.

    Args:
        scenario: Which scenario to run.
        eval_set: List of eval queries from JSON.
        gen_pipeline: Initialized GenerationPipeline.
        all_chunks_data: Mapping chunk_id → {chunk_id, text, section_type, document_id}.
        output_path: Path for results JSON.
        max_queries: Limit for quick tests.
        seed: Random seed for reproducibility.
        judge_provider: Optional LLM provider for faithfulness/relevance scoring.
        n_context_chunks: Number of chunks for random/partial context.

    Returns:
        ScenarioMetrics for this scenario.
    """
    rng = random.Random(seed)
    all_chunk_ids = list(all_chunks_data.keys())

    doc_to_chunks: dict[str, list[dict[str, Any]]] = {}
    for cdata in all_chunks_data.values():
        doc_id = cdata["document_id"]
        doc_to_chunks.setdefault(doc_id, []).append(cdata)

    checkpoint_path = Path(str(output_path) + f".{scenario.value}.checkpoint.json")
    existing_results = _load_checkpoint(checkpoint_path, scenario.value)
    results: list[RobustnessQueryResult] = existing_results or []
    start_idx = len(results)

    queries = eval_set[:max_queries] if max_queries else eval_set

    if start_idx >= len(queries):
        logger.info("All %d queries already completed for %s", len(queries), scenario.value)
    else:
        logger.info(
            "Running %s scenario: %d queries (resuming from %d)",
            scenario.value,
            len(queries),
            start_idx,
        )

    skipped = 0
    for i in range(start_idx, len(queries)):
        q = queries[i]
        query_text = q["query"]
        category = q.get("category", "unknown")
        relevant_ids = set(q.get("relevant_ids", []))
        source_papers = q.get("source_papers", [])
        relevant_sections = q.get("relevant_sections", [])

        context = ""
        context_chunks_count = 0
        context_doc_id = ""
        context_section_types: list[str] = []

        if scenario == RobustnessScenario.EMPTY:
            context = ""
            context_chunks_count = 0

        elif scenario == RobustnessScenario.RANDOM:
            context, used_chunks = build_random_context(
                all_chunk_ids=all_chunk_ids,
                chunk_id_to_data=all_chunks_data,
                relevant_ids=relevant_ids,
                n_chunks=n_context_chunks,
                rng=rng,
            )
            context_chunks_count = len(used_chunks)

        elif scenario == RobustnessScenario.PARTIAL:
            source_doc_id = ""
            if source_papers:
                arxiv_id = source_papers[0]
                for did in doc_to_chunks:
                    if did.startswith(arxiv_id):
                        source_doc_id = did
                        break

            if not source_doc_id:
                skipped += 1
                logger.debug(
                    "Skipping query %d: source paper %s not found in corpus",
                    i,
                    source_papers,
                )
                continue

            doc_chunks = doc_to_chunks.get(source_doc_id, [])
            context, used_chunks = build_partial_context(
                document_id=source_doc_id,
                doc_chunks=doc_chunks,
                relevant_ids=relevant_ids,
                relevant_sections=relevant_sections,
                n_chunks=n_context_chunks,
                rng=rng,
            )
            context_chunks_count = len(used_chunks)
            context_doc_id = source_doc_id
            context_section_types = list({c.get("section_type", "unknown") for c in used_chunks})

            if not used_chunks:
                skipped += 1
                logger.debug(
                    "Skipping query %d: all chunks in %s are relevant",
                    i,
                    source_doc_id,
                )
                continue

        mode = CognitiveMode.from_intent(category)

        try:
            response: RAGResponse = await gen_pipeline.generate_from_context(
                query=query_text,
                context=context,
                mode=mode,
            )
            answer = response.answer
            refusal = is_refusal(answer)

            result = RobustnessQueryResult(
                query=query_text,
                category=category,
                scenario=scenario.value,
                answer=answer,
                is_refusal=refusal,
                mode=response.mode.value,
                latency_ms=response.total_latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                context_chunks_count=context_chunks_count,
                context_document_id=context_doc_id,
                context_section_types=context_section_types,
            )

        except Exception as e:
            logger.error("Generation failed for query %d: %s", i, e)
            result = RobustnessQueryResult(
                query=query_text,
                category=category,
                scenario=scenario.value,
                answer=f"[ERROR: {e}]",
                is_refusal=True,
                mode="error",
                latency_ms=0.0,
                context_chunks_count=context_chunks_count,
                context_document_id=context_doc_id,
                context_section_types=context_section_types,
            )

        results.append(result)

        done = len(results)
        refusal_so_far = sum(1 for r in results if r.is_refusal)
        if done % 10 == 0 or done == len(queries):
            logger.info(
                "[%s] Progress: %d/%d (refusal: %d/%d = %.1f%%)",
                scenario.value,
                done,
                len(queries),
                refusal_so_far,
                done,
                100 * refusal_so_far / done,
            )

        if done % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(results, checkpoint_path, scenario.value)

    if judge_provider is not None:
        await _run_judge_on_non_refusals(results, judge_provider)

    metrics = compute_scenario_metrics(results, scenario.value)

    output_data = {
        "scenario": scenario.value,
        "seed": seed,
        "n_context_chunks": n_context_chunks,
        "total_queries": len(queries),
        "skipped_queries": skipped,
        "metrics": metrics.to_dict(),
        "results": [r.to_dict() for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", output_path)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint removed: %s", checkpoint_path)

    return metrics


async def _run_judge_on_non_refusals(
    results: list[RobustnessQueryResult],
    judge_provider: Any,
) -> None:
    """Run LLM-as-judge on non-refusal answers for faithfulness/relevance.

    Uses the same judge prompt format as the main evaluation.
    Only scores answers that were NOT classified as refusals.
    """
    non_refusal_results = [r for r in results if not r.is_refusal]
    if not non_refusal_results:
        logger.info("No non-refusal answers to judge")
        return

    logger.info("Running judge on %d non-refusal answers", len(non_refusal_results))

    faithfulness_prompt = """\
    You are evaluating an AI assistant's answer for faithfulness to the provided context.

    Question: {query}
    Answer: {answer}

    Rate the faithfulness of the answer on a scale of 1-5:
    1 = Completely unfaithful, contains major hallucinations
    2 = Mostly unfaithful, significant unsupported claims
    3 = Partially faithful, some claims not in context
    4 = Mostly faithful, minor unsupported details
    5 = Completely faithful, all claims supported by context

    Note: The context provided to the assistant was intentionally degraded \
    (random or partially relevant). Low faithfulness is EXPECTED and correct \
    behavior would be to refuse answering.

    Respond with only: SCORE: <number>
    REASONING: <brief explanation>"""

    relevance_prompt = """\
    You are evaluating whether an AI assistant's answer is relevant to the question asked.

    Question: {query}
    Answer: {answer}

    Rate the relevance on a scale of 1-5:
    1 = Completely irrelevant
    2 = Mostly irrelevant
    3 = Partially relevant
    4 = Mostly relevant
    5 = Directly answers the question

    Respond with only: SCORE: <number>
    REASONING: <brief explanation>"""

    score_pattern = re.compile(r"SCORE:\s*(\d)")

    for i, result in enumerate(non_refusal_results):
        try:
            # Faithfulness
            faith_resp = await judge_provider.generate(
                prompt=faithfulness_prompt.format(
                    query=result.query,
                    answer=result.answer,
                ),
                system_prompt="You are a precise evaluation judge.",
                temperature=0.0,
                max_tokens=200,
            )
            faith_match = score_pattern.search(faith_resp.text)
            if faith_match:
                result.faithfulness = float(faith_match.group(1))

            # Relevance
            rel_resp = await judge_provider.generate(
                prompt=relevance_prompt.format(
                    query=result.query,
                    answer=result.answer,
                ),
                system_prompt="You are a precise evaluation judge.",
                temperature=0.0,
                max_tokens=200,
            )
            rel_match = score_pattern.search(rel_resp.text)
            if rel_match:
                result.relevance = float(rel_match.group(1))

        except Exception as e:
            logger.warning("Judge failed for query %d: %s", i, e)

        if (i + 1) % 10 == 0:
            logger.info("Judge progress: %d/%d", i + 1, len(non_refusal_results))


async def run_robustness_evaluation(  # noqa: PLR0913
    eval_set_path: str,
    collection: str = "peft_hybrid",
    scenario: str = "all",
    output_dir: str = "eval/results",
    max_queries: int | None = None,
    seed: int = 42,
    n_context_chunks: int = 10,
    llm_provider: str = "ollama",
    llm_model: str = "qwen3:8b",
    llm_host: str = "http://localhost:11434",
    skip_judge: bool = False,
    judge_model: str | None = None,
    judge_host: str | None = None,
    experiment_name: str = "robustness",
    run_name: str | None = None,
) -> dict[str, ScenarioMetrics]:
    """Run robustness evaluation — main entry point.

    Args:
        eval_set_path: Path to eval set JSON.
        collection: Qdrant collection name.
        scenario: "empty", "random", "partial", or "all".
        output_dir: Directory for result files.
        max_queries: Limit queries per scenario (for quick tests).
        seed: Random seed.
        n_context_chunks: Chunks per context (random/partial).
        llm_provider: LLM backend ("ollama" / "llama_cpp").
        llm_model: Model name.
        llm_host: LLM server URL.
        skip_judge: If True, skip LLM-as-judge evaluation.
        judge_model: Separate model for judge (defaults to llm_model).
        judge_host: Separate host for judge (defaults to llm_host).
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Returns:
        Dict of scenario → ScenarioMetrics.
    """
    eval_set_file = Path(eval_set_path)
    if not eval_set_file.exists():
        raise FileNotFoundError(f"Eval set not found: {eval_set_path}")

    eval_set = json.loads(eval_set_file.read_text(encoding="utf-8"))
    logger.info("Loaded eval set: %d queries from %s", len(eval_set), eval_set_path)

    if scenario == "all":
        scenarios = [
            RobustnessScenario.EMPTY,
            RobustnessScenario.RANDOM,
            RobustnessScenario.PARTIAL,
        ]
    else:
        scenarios = [RobustnessScenario(scenario)]

    gen_llm = create_llm_provider(
        provider=llm_provider,
        model=llm_model,
        host=llm_host,
    )
    gen_config = GenerationConfig()
    gen_pipeline = GenerationPipeline(llm=gen_llm, config=gen_config)

    judge = None
    if not skip_judge:
        settings = get_settings()
        judge_api_key = settings.openrouter.api_key.get_secret_value()
        judge = OpenRouterProvider(
            model=judge_model or "openai/gpt-5.4-mini",
            api_key=judge_api_key,
            host=judge_host or "https://openrouter.ai/api/v1",
        )

    all_chunks_data: dict[str, dict[str, Any]] = {}
    needs_chunks = any(
        s in scenarios for s in [RobustnessScenario.RANDOM, RobustnessScenario.PARTIAL]
    )
    if needs_chunks:
        store = await create_qdrant_store(collection_name=collection)
        all_chunks_data = await load_all_chunks_data(store)
        await store.close()

    output_base = Path(output_dir)
    all_metrics: dict[str, ScenarioMetrics] = {}

    for sc in scenarios:
        output_path = output_base / f"robustness_{sc.value}.json"
        metrics = await run_robustness_scenario(
            scenario=sc,
            eval_set=eval_set,
            gen_pipeline=gen_pipeline,
            all_chunks_data=all_chunks_data,
            output_path=output_path,
            max_queries=max_queries,
            seed=seed,
            judge_provider=judge,
            n_context_chunks=n_context_chunks,
        )
        all_metrics[sc.value] = metrics

        print(f"\n{'=' * 60}")
        print(f"Scenario: {sc.value}")
        print(f"{'=' * 60}")
        print(f"  Total queries:    {metrics.total_queries}")
        print(f"  Refusal count:    {metrics.refusal_count}")
        print(f"  Refusal rate:     {metrics.refusal_rate:.1%}")
        print(f"  Non-refusal:      {metrics.non_refusal_count}")
        print(f"  Mean latency:     {metrics.mean_latency_ms:.0f}ms")
        if metrics.mean_faithfulness is not None:
            print(f"  Mean faithfulness: {metrics.mean_faithfulness:.2f}")
            print(f"  Faith. pass rate:  {metrics.faithfulness_pass_rate:.1%}")
        if metrics.mean_relevance is not None:
            print(f"  Mean relevance:    {metrics.mean_relevance:.2f}")
        print("\n  Per-category breakdown:")
        for cat, cat_data in metrics.per_category.items():
            print(
                f"    {cat:20s}: {cat_data['refusal_count']}/{cat_data['total']} "
                f"refusal ({cat_data['refusal_rate']:.1%})"
            )
        print()

    try:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        mlflow.set_experiment(experiment_name)

        effective_run_name = run_name or f"robustness_{scenario}_{llm_model}"

        with mlflow.start_run(
            run_name=effective_run_name,
        ):
            mlflow.log_params(
                {
                    "eval_set": eval_set_path,
                    "collection": collection,
                    "scenarios": scenario,
                    "llm_model": llm_model,
                    "llm_provider": llm_provider,
                    "seed": seed,
                    "n_context_chunks": n_context_chunks,
                    "max_queries": max_queries or len(eval_set),
                    "skip_judge": skip_judge,
                }
            )

            for sc_name, sc_metrics in all_metrics.items():
                prefix = f"{sc_name}_"
                metrics_dict = {
                    f"{prefix}refusal_rate": sc_metrics.refusal_rate,
                    f"{prefix}refusal_count": sc_metrics.refusal_count,
                    f"{prefix}total_queries": sc_metrics.total_queries,
                    f"{prefix}non_refusal_count": sc_metrics.non_refusal_count,
                    f"{prefix}mean_latency_ms": sc_metrics.mean_latency_ms,
                }
                if sc_metrics.mean_faithfulness is not None:
                    metrics_dict[f"{prefix}mean_faithfulness"] = sc_metrics.mean_faithfulness
                if sc_metrics.mean_relevance is not None:
                    metrics_dict[f"{prefix}mean_relevance"] = sc_metrics.mean_relevance
                if sc_metrics.faithfulness_pass_rate is not None:
                    metrics_dict[f"{prefix}faithfulness_pass_rate"] = (
                        sc_metrics.faithfulness_pass_rate
                    )

                mlflow.log_metrics(metrics_dict)

            summary = {sc_name: sc_metrics.to_dict() for sc_name, sc_metrics in all_metrics.items()}
            summary_path = output_base / "robustness_summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(summary_path))

        logger.info("MLflow run logged: %s", effective_run_name)

    except ImportError:
        logger.warning("MLflow not available, skipping tracking")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    return all_metrics


def print_comparison_table(
    normal_metrics: dict[str, Any],
    robustness_metrics: dict[str, ScenarioMetrics],
) -> None:
    """Print a comparison table: normal RAG vs robustness scenarios.

    Args:
        normal_metrics: Dict with keys like "refusal_rate", "mean_faithfulness"
                       from the main evaluation.
        robustness_metrics: Dict of scenario → ScenarioMetrics.
    """
    print("\n" + "=" * 75)
    print("ROBUSTNESS COMPARISON: Normal RAG vs Degraded Context")
    print("=" * 75)
    print(f"{'Scenario':<16} {'Refusal%':>10} {'Faithfulness':>14} {'Faith.Pass%':>12}")
    print("-" * 75)

    # Normal RAG
    nr = normal_metrics
    print(
        f"{'Normal RAG':<16} "
        f"{nr.get('refusal_rate', 0):.1%}      "
        f"{nr.get('mean_faithfulness', 0):.2f}          "
        f"{nr.get('faithfulness_pass_rate', 0):.1%}"
    )

    # Robustness scenarios
    for sc_name in ["empty", "random", "partial"]:
        if sc_name in robustness_metrics:
            m = robustness_metrics[sc_name]
            faith_str = f"{m.mean_faithfulness:.2f}" if m.mean_faithfulness is not None else "N/A"
            pass_str = (
                f"{m.faithfulness_pass_rate:.1%}" if m.faithfulness_pass_rate is not None else "N/A"
            )
            print(f"{sc_name:<16} {m.refusal_rate:.1%}      {faith_str:>14} {pass_str:>12}")

    print("=" * 75)
