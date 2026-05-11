from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from contextual_research_agent.common.logging import setup_logging
from contextual_research_agent.generation.openrouter_provider import OpenRouterProvider
from contextual_research_agent.ingestion.vectorstores.qdrant_store import (
    create_qdrant_store,
)

logger = logging.getLogger(__name__)


_QUERY_GENERATION_PROMPTS: dict[str, str] = {
    "factual_qa": """\
Read the following passages from a scientific paper. Generate ONE specific factual \
question that can be answered using only these passages. The question should ask \
about concrete facts: numbers, percentages, model names, datasets, accuracy values, \
or technical specifications. Avoid open-ended or interpretive questions.

Passages:
{context}

Output ONLY the question, no preamble or explanation.""",
    "summarization": """\
Read the following passages from a scientific paper. Generate ONE question that \
asks for a concise summary of the main contribution, methodology, or findings. \
The question should require synthesizing information from multiple passages.

Examples:
- "What is the main contribution of this paper?"
- "How does the proposed method work?"
- "What are the key findings of this work?"

Passages:
{context}

Output ONLY the question, no preamble or explanation.""",
    "critical_review": """\
Read the following passages from a scientific paper. Generate ONE question that \
asks for a critical analysis: weaknesses, limitations, methodological gaps, or \
unstated assumptions. The question should require identifying problems, not just \
describing them.

Examples:
- "What are the main weaknesses of this approach?"
- "What methodological limitations does this paper have?"
- "What assumptions does this work make that may not hold in practice?"

Passages:
{context}

Output ONLY the question, no preamble or explanation.""",
}


_ANSWER_GENERATION_PROMPTS: dict[str, str] = {
    "factual_qa": """\
You are an expert scientific research assistant.

Context passages:
{context}

Question: {query}

Provide a precise, factual answer based STRICTLY on the context above. \
Answer in 2-4 sentences with specific numbers, model names, or technical details \
where applicable. Cite passages using [chunk_id] notation. \
Do not use bullet points or headers.""",
    "summarization": """\
You are an expert scientific research assistant.

Context passages:
{context}

Question: {query}

Provide a concise summary in 3-5 sentences. Focus on the most important points: \
main contribution, methodology, key results. Do NOT use bullet points or numbered \
lists. Cite passages using [chunk_id] notation.""",
    "critical_review": """\
You are an expert scientific research assistant focused on critical analysis.

Context passages:
{context}

Question: {query}

Provide a concise critical analysis in 4-6 sentences. Cover: \
(1) what the approach claims, (2) key strengths from context, \
(3) weaknesses, gaps, or limitations. Distinguish between limitations \
acknowledged by authors and those you identify from context. \
Do NOT use section headers or bullet points. Cite passages using [chunk_id] notation.""",
}

_GENERIC_SYSTEM_PROMPT = """\
You are an expert scientific research assistant specializing in machine learning, \
deep learning, and parameter-efficient fine-tuning (PEFT) methods.

You MUST follow these rules strictly:
1. Answer ONLY based on the provided context passages. Do NOT use prior knowledge.
2. If the context does not contain sufficient information, say: \
"The provided sources do not contain enough information to answer this question."
3. Cite specific passages using [chunk_id] notation when making claims.
4. Be precise and technical. Avoid vague or speculative statements.
5. Use correct ML/NLP terminology."""


async def load_corpus_chunks(
    collection: str,
    excluded_chunk_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    store = await create_qdrant_store(collection_name=collection)

    chunks: list[dict[str, Any]] = []
    offset = None
    batch_size = 256
    excluded = excluded_chunk_ids or set()

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
            if not chunk_id or chunk_id in excluded:
                continue

            metadata = payload.get("metadata", {})
            section_type = "unknown"
            if isinstance(metadata, dict):
                section_type = metadata.get("section_type", "unknown")

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": payload.get("text", ""),
                    "section_type": section_type,
                    "document_id": payload.get("document_id", ""),
                }
            )

        if next_offset is None or len(points) == 0:
            break
        offset = next_offset

    await store.close()
    logger.info("Loaded %d chunks from %s (excluded %d)", len(chunks), collection, len(excluded))
    return chunks


def load_excluded_chunks(eval_set_path: str | None) -> set[str]:
    if not eval_set_path or not Path(eval_set_path).exists():
        logger.warning("Eval set not provided — no chunks will be excluded")
        return set()

    eval_data = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
    excluded: set[str] = set()
    for item in eval_data:
        excluded.update(item.get("relevant_ids", []))

    logger.info("Excluded %d chunks (eval set relevant_ids)", len(excluded))
    return excluded


def build_context_for_training(
    primary_chunk: dict[str, Any],
    same_doc_chunks: list[dict[str, Any]],
    n_neighbors: int = 4,
    rng: random.Random | None = None,
) -> tuple[str, list[str]]:
    _rng = rng or random.Random()

    neighbors = [c for c in same_doc_chunks if c["chunk_id"] != primary_chunk["chunk_id"]]
    _rng.shuffle(neighbors)
    selected_neighbors = neighbors[:n_neighbors]

    chunks_to_use = [primary_chunk] + selected_neighbors
    _rng.shuffle(chunks_to_use)

    parts: list[str] = []
    chunk_ids: list[str] = []
    for c in chunks_to_use:
        chunk_id = c["chunk_id"]
        section = c.get("section_type", "unknown")
        text = c["text"]
        parts.append(f"[{chunk_id}] (section: {section})\n{text}")
        chunk_ids.append(chunk_id)

    return "\n\n---\n\n".join(parts), chunk_ids


async def generate_query(
    teacher: OpenRouterProvider,
    mode: str,
    context: str,
    max_retries: int = 2,
) -> str | None:
    prompt = _QUERY_GENERATION_PROMPTS[mode].format(context=context)

    for attempt in range(max_retries):
        try:
            result = await teacher.generate(
                prompt=prompt,
                system_prompt=(
                    "You are a research assistant generating training data. "
                    "Output only what is requested, no preamble."
                ),
                temperature=0.7,
                max_tokens=200,
            )
            query = result.text.strip()
            query = re.sub(r'^["\']|["\']$', "", query).strip()
            query = re.sub(r"^Question:\s*", "", query, flags=re.IGNORECASE).strip()

            if 10 < len(query) < 500:
                return query
            logger.debug("Bad query length %d, retrying", len(query))
        except Exception as e:
            logger.warning("Query generation failed (attempt %d): %s", attempt + 1, e)
            await asyncio.sleep(1)

    return None


async def generate_answer(
    teacher: OpenRouterProvider,
    mode: str,
    context: str,
    query: str,
    max_retries: int = 2,
) -> str | None:
    prompt = _ANSWER_GENERATION_PROMPTS[mode].format(context=context, query=query)

    for attempt in range(max_retries):
        try:
            result = await teacher.generate(
                prompt=prompt,
                system_prompt="You are an expert scientific research assistant.",
                temperature=0.2,
                max_tokens=600,
            )
            answer = result.text.strip()
            if 50 < len(answer) < 3000:
                return answer
        except Exception as e:
            logger.warning("Answer generation failed (attempt %d): %s", attempt + 1, e)
            await asyncio.sleep(1)

    return None


async def generate_for_mode(  # noqa: PLR0913
    mode: str,
    chunks: list[dict[str, Any]],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    teacher: OpenRouterProvider,
    n_examples: int,
    output_path: Path,
    rng: random.Random,
    checkpoint_interval: int = 25,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    with contextlib.suppress(json.JSONDecodeError):
                        existing.append(json.loads(line))
        logger.info("Resuming %s from %d existing examples", mode, len(existing))

    needed = n_examples - len(existing)
    if needed <= 0:
        logger.info("Already have %d examples for %s — done", len(existing), mode)
        return len(existing)

    sampled_chunks = rng.sample(chunks, min(needed, len(chunks)))
    if len(sampled_chunks) < needed:
        extra = rng.choices(chunks, k=needed - len(sampled_chunks))
        sampled_chunks.extend(extra)

    succeeded = 0
    failed = 0

    with output_path.open("a", encoding="utf-8") as f:
        for i, primary_chunk in enumerate(sampled_chunks):
            doc_id = primary_chunk["document_id"]
            same_doc = chunks_by_doc.get(doc_id, [])

            # Build context
            context, chunk_ids = build_context_for_training(
                primary_chunk=primary_chunk,
                same_doc_chunks=same_doc,
                n_neighbors=4,
                rng=rng,
            )

            # Generate query
            query = await generate_query(teacher, mode, context)
            if not query:
                failed += 1
                logger.debug("[%s] %d/%d: query generation failed", mode, i + 1, needed)
                continue

            # Generate answer
            answer = await generate_answer(teacher, mode, context, query)
            if not answer:
                failed += 1
                logger.debug("[%s] %d/%d: answer generation failed", mode, i + 1, needed)
                continue

            # Build training example
            example = {
                "messages": [
                    {"role": "system", "content": _GENERIC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context passages:\n{context}\n\n---\nQuestion: {query}",
                    },
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "mode": mode,
                    "primary_chunk_id": primary_chunk["chunk_id"],
                    "primary_section": primary_chunk["section_type"],
                    "source_document": doc_id,
                    "context_chunk_ids": chunk_ids,
                    "query": query,
                },
            }

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            f.flush()
            succeeded += 1

            if (i + 1) % checkpoint_interval == 0:
                logger.info(
                    "[%s] Progress: %d generated / %d failed (target %d)",
                    mode,
                    succeeded,
                    failed,
                    needed,
                )

    total = len(existing) + succeeded
    logger.info(
        "[%s] Complete: %d total examples (%d new, %d failed)",
        mode,
        total,
        succeeded,
        failed,
    )
    return total


async def main(  # noqa: PLR0913
    collection: str = "peft_hybrid",
    modes: str | tuple | list = "factual_qa,summarization,critical_review",
    examples_per_mode: int = 500,
    teacher_model: str = "openai/gpt-5.4-mini",
    teacher_host: str = "https://openrouter.ai/api/v1",
    api_key: str = "",
    output_dir: str = "training/data",
    eval_set: str | None = "eval/peft_gold_v3_mapped.json",
    seed: int = 42,
) -> None:
    setup_logging()

    key = api_key
    if not key:
        try:
            from contextual_research_agent.common.settings import get_settings

            settings = get_settings()
            key = settings.openrouter.api_key.get_secret_value()
        except Exception:
            pass

    if not key:
        raise RuntimeError(
            "No OpenRouter API key. Set via --api-key or OPENROUTER_API_KEY env var."
        )

    excluded = load_excluded_chunks(eval_set) if eval_set else set()

    chunks = await load_corpus_chunks(collection, excluded_chunk_ids=excluded)
    if not chunks:
        raise RuntimeError(f"No chunks found in collection {collection}")

    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c["document_id"]].append(c)

    logger.info(
        "Corpus: %d chunks across %d documents",
        len(chunks),
        len(chunks_by_doc),
    )

    teacher = OpenRouterProvider(model=teacher_model, api_key=key, host=teacher_host)

    if isinstance(modes, (tuple, list)):
        mode_list = [str(m).strip() for m in modes if str(m).strip()]
    else:
        mode_list = [m.strip() for m in str(modes).split(",") if m.strip()]
    output_path = Path(output_dir)
    rng = random.Random(seed)

    summary: dict[str, int] = {}
    try:
        for mode in mode_list:
            if mode not in _QUERY_GENERATION_PROMPTS:
                logger.warning("Unknown mode '%s', skipping", mode)
                continue

            mode_path = output_path / f"{mode}.jsonl"
            count = await generate_for_mode(
                mode=mode,
                chunks=chunks,
                chunks_by_doc=chunks_by_doc,
                teacher=teacher,
                n_examples=examples_per_mode,
                output_path=mode_path,
                rng=rng,
            )
            summary[mode] = count
    finally:
        await teacher.close()

    print("\n" + "=" * 60)
    print("TRAINING DATA GENERATION COMPLETE")
    print("=" * 60)
    for mode, count in summary.items():
        print(f"  {mode:25s}: {count:>5d} examples → {output_path}/{mode}.jsonl")
    print()


def cli_entry() -> None:
    import fire

    def sync_main(*args, **kwargs):
        return asyncio.run(main(*args, **kwargs))

    fire.Fire(sync_main)


if __name__ == "__main__":
    cli_entry()
