from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from contextual_research_agent.agent.llm import create_llm_provider
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.openrouter_provider import OpenRouterProvider
from contextual_research_agent.ingestion.embeddings.hf_embedder import create_hf_embedder


def no_rag_baseline(  # noqa: PLR0913
    eval_set: str = "eval/peft_gold_v3_mapped.json",
    provider: str = "ollama",
    model: str = "qwen3:8b",
    host: str = "http://localhost:11434",
    temperature: float = 0.1,
    max_tokens: int = 512,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str | None = None,
    output: str | None = None,
    experiment_name: str = "no_rag_baseline",
    run_name: str | None = None,
    max_queries: int | None = None,
    checkpoint_interval: int = 25,
) -> None:
    """Evaluate LLM without RAG context as baseline.

    Args:
        eval_set: Path to eval JSON with queries and expected answers.
        provider: LLM provider (ollama / openrouter).
        model: Model name.
        host: LLM server URL (for ollama).
        temperature: Generation temperature.
        max_tokens: Max generation tokens.
        embedding_model: For computing semantic similarity.
        device: Device for embedding model.
        output: Path to save results JSON.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        max_queries: Limit queries for testing.
        checkpoint_interval: Save checkpoint every N queries.
    """

    async def _run() -> None:
        logger = get_logger(__name__)

        # Load eval data
        eval_data = json.loads(Path(eval_set).read_text(encoding="utf-8"))
        if not isinstance(eval_data, list):
            print("ERROR: Expected JSON array")
            return
        if max_queries:
            eval_data = eval_data[:max_queries]

        print(f"Loaded {len(eval_data)} queries")
        print(f"Provider: {provider}, Model: {model}")
        print(f"NO RAG — queries sent directly to LLM\n")

        # Create LLM
        if provider == "openrouter":
            from contextual_research_agent.common.settings import get_settings  # noqa: PLC0415

            settings = get_settings()
            api_key = settings.openrouter.api_key.get_secret_value()
            llm = OpenRouterProvider(model=model, api_key=api_key, host=host)
        else:
            llm = create_llm_provider(provider=provider, model=model, host=host)

        # Embedder for similarity
        embedder = create_hf_embedder(model=embedding_model, device=device)

        # System prompt — same domain expertise, but no context restriction
        system_prompt = (
            "You are an expert scientific research assistant specializing in "
            "machine learning, deep learning, and parameter-efficient fine-tuning (PEFT) methods. "
            "Answer questions precisely and technically based on your knowledge. "
            "If you don't know the answer, say so."
        )

        # Checkpoint
        output_path = Path(output) if output else None
        checkpoint_path = output_path.with_suffix(".checkpoint.json") if output_path else None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        start_idx = 0
        per_query_results: list[dict[str, Any]] = []

        if checkpoint_path and checkpoint_path.exists():
            try:
                cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                per_query_results = cp.get("per_query", [])
                start_idx = len(per_query_results)
                print(f"Resuming from checkpoint: {start_idx}/{len(eval_data)}")
            except Exception:
                start_idx = 0
                per_query_results = []

        similarities: list[float] = []
        refusals = 0
        t_total = time.perf_counter()

        for i, item in enumerate(eval_data):
            if i < start_idx:
                # Restore stats from checkpoint
                r = per_query_results[i]
                if r.get("semantic_similarity") is not None:
                    similarities.append(r["semantic_similarity"])
                if r.get("is_refusal"):
                    refusals += 1
                continue

            query = item["query"]
            expected = item.get("expected_answer", "")
            category = item.get("category", "")

            try:
                result = await llm.generate(
                    prompt=query,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = result.text.strip()

                # Check refusal
                is_refusal = not answer or any(
                    p in answer.lower()
                    for p in [
                        "i don't know",
                        "i'm not sure",
                        "cannot determine",
                        "don't have enough",
                        "not able to",
                    ]
                )

                # Semantic similarity
                sim = None
                if expected and answer and not is_refusal:
                    emb_a = await embedder.embed_query(answer)
                    emb_e = await embedder.embed_query(expected)
                    a, b = np.array(emb_a), np.array(emb_e)
                    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
                    sim = max(0.0, min(1.0, sim))
                    similarities.append(sim)

                if is_refusal:
                    refusals += 1

                per_query_results.append(
                    {
                        "query": query,
                        "category": category,
                        "expected_answer": expected,
                        "generated_answer": answer,
                        "semantic_similarity": sim,
                        "is_refusal": is_refusal,
                        "latency_ms": result.latency_ms,
                        "tokens": result.total_tokens,
                    }
                )

                sim_s = f"sim={sim:.3f}" if sim is not None else "sim=N/A"
                ref_s = "REFUSAL" if is_refusal else "OK"
                print(f"  [{i + 1}/{len(eval_data)}] {sim_s} ({ref_s}) ({result.latency_ms:.0f}ms)")

            except Exception as e:
                logger.error(f"Query {i + 1} failed: {e}")
                per_query_results.append(
                    {
                        "query": query,
                        "category": category,
                        "expected_answer": expected,
                        "generated_answer": f"ERROR: {e}",
                        "semantic_similarity": None,
                        "is_refusal": True,
                        "latency_ms": 0,
                        "tokens": 0,
                    }
                )
                refusals += 1
                print(f"  [{i + 1}/{len(eval_data)}] ERROR: {e}")

            # Checkpoint
            if checkpoint_path and (i + 1) % checkpoint_interval == 0:
                cp_data = {"per_query": per_query_results}
                checkpoint_path.write_text(
                    json.dumps(cp_data, ensure_ascii=False), encoding="utf-8"
                )
                print(f"  >>> Checkpoint: {i + 1}/{len(eval_data)}")

        elapsed = time.perf_counter() - t_total

        # Aggregate
        mean_sim = sum(similarities) / len(similarities) if similarities else 0
        sorted_sims = sorted(similarities)
        median_sim = sorted_sims[len(sorted_sims) // 2] if sorted_sims else 0
        refusal_rate = refusals / len(eval_data) if eval_data else 0

        by_cat = defaultdict(lambda: {"sims": [], "refs": 0, "total": 0})
        for r in per_query_results:
            cat = r.get("category", "unknown")
            by_cat[cat]["total"] += 1
            if r.get("is_refusal"):
                by_cat[cat]["refs"] += 1
            if r.get("semantic_similarity") is not None:
                by_cat[cat]["sims"].append(r["semantic_similarity"])

        print("\n" + "=" * 60)
        print(f"NO-RAG BASELINE: {model}")
        print("=" * 60)
        print(f"  Mean sim:     {mean_sim:.4f}")
        print(f"  Median sim:   {median_sim:.4f}")
        print(f"  Refusal rate: {refusal_rate:.1%}")
        print(f"  Time:         {elapsed:.1f}s")

        print("\nPer-category:")
        for cat in sorted(by_cat.keys()):
            c = by_cat[cat]
            ms = sum(c["sims"]) / len(c["sims"]) if c["sims"] else 0
            rr = c["refs"] / c["total"] if c["total"] else 0
            print(f"  {cat:<22s}: sim={ms:.3f}, refusal={rr:.1%}, n={c['total']}")

        # Save
        if output_path:
            report = {
                "config": {
                    "provider": provider,
                    "model": model,
                    "mode": "no_rag_baseline",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "num_queries": len(eval_data),
                },
                "aggregated": {
                    "mean_semantic_similarity": round(mean_sim, 4),
                    "median_semantic_similarity": round(median_sim, 4),
                    "refusal_rate": round(refusal_rate, 4),
                    "num_queries": len(eval_data),
                },
                "category_metrics": {
                    cat: {
                        "count": by_cat[cat]["total"],
                        "mean_sim": round(sum(by_cat[cat]["sims"]) / len(by_cat[cat]["sims"]), 4)
                        if by_cat[cat]["sims"]
                        else 0,
                        "refusal_rate": round(by_cat[cat]["refs"] / by_cat[cat]["total"], 4)
                        if by_cat[cat]["total"]
                        else 0,
                    }
                    for cat in sorted(by_cat.keys())
                },
                "per_query": per_query_results,
            }
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\nSaved to: {output}")

            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()

        # MLflow
        try:
            from contextual_research_agent.common.settings import get_settings  # noqa: PLC0415

            settings = get_settings()
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            mlflow.set_experiment(experiment_name)

            auto_name = run_name or f"no_rag_{model.replace('/', '_')}"
            with mlflow.start_run(run_name=auto_name):
                mlflow.log_params({"model": model, "provider": provider, "mode": "no_rag"})
                mlflow.log_metric("mean_sim", mean_sim)
                mlflow.log_metric("refusal_rate", refusal_rate)
            print(f"Logged to MLflow: {experiment_name}/{auto_name}")
        except Exception as e:
            print(f"MLflow logging failed: {e}")

        await llm.close()

    asyncio.run(_run())
