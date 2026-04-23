from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mlflow

from contextual_research_agent.generation.openrouter_provider import OpenRouterProvider


def run_judge(  # noqa: PLR0913
    input: str,
    output: str,
    model: str = "openai/gpt-5.4-mini",
    api_key: str = "",
    host: str = "https://openrouter.ai/api/v1",
    max_queries: int | None = None,
    checkpoint_interval: int = 25,
    experiment_name: str = "judge_evaluation",
    run_name: str | None = None,
) -> None:
    """Run LLM-as-judge on existing generation results.

    Args:
        input: Path to generation results JSON (from evaluate-generation).
        output: Path to save judged results.
        model: OpenRouter model ID for judge.
        api_key: OpenRouter API key (or set via api_key_env).
        api_key_env: Environment variable name for API key.
        host: OpenRouter API base URL.
        max_queries: Limit queries for testing.
        checkpoint_interval: Save checkpoint every N queries.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    async def _run() -> None:
        # Resolve API key
        key = api_key
        if not key:
            with contextlib.suppress(Exception):
                from contextual_research_agent.common.settings import get_settings  # noqa: PLC0415

                settings = get_settings()
                key = settings.openrouter.api_key.get_secret_value()

        if not key:
            print("ERROR: No API key.")
            return

        # Load existing results
        data = json.loads(Path(input).read_text(encoding="utf-8"))
        per_query = data.get("per_query", [])
        if not per_query:
            print("ERROR: No per_query results found in input file")
            return

        if max_queries:
            per_query = per_query[:max_queries]

        print(f"Loaded {len(per_query)} queries from {input}")
        print(f"Judge model: {model}")

        judge = OpenRouterProvider(model=model, api_key=key, host=host)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path.with_suffix(".checkpoint.json")

        start_idx = 0
        if checkpoint_path.exists():
            try:
                cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                start_idx = cp.get("num_completed", 0)
                for i, judged in enumerate(cp.get("judged", [])):
                    if i < len(per_query):
                        per_query[i]["judge"] = judged
                print(f"Resuming from checkpoint: {start_idx}/{len(per_query)}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                start_idx = 0

        t_total = time.perf_counter()

        faith_scores = []
        rel_scores = []
        total_input_tokens = 0
        total_output_tokens = 0
        errors = 0

        for i, item in enumerate(per_query):
            if i < start_idx:
                if "judge" in item:
                    j = item["judge"]
                    if j.get("faithfulness_score") is not None:
                        faith_scores.append(j["faithfulness_score"])
                    if j.get("relevance_score") is not None:
                        rel_scores.append(j["relevance_score"])
                continue

            query = item["query"]
            answer = item.get("generated_answer", "")
            expected = item.get("expected_answer", "")
            is_refusal = item.get("gen_metrics", {}).get("is_refusal", False) or item.get(
                "is_refusal", False
            )

            if is_refusal or not answer.strip():
                item["judge"] = {
                    "faithfulness_score": None,
                    "faithfulness_reasoning": "Skipped: refusal",
                    "relevance_score": None,
                    "relevance_reasoning": "Skipped: refusal",
                    "judge_model": model,
                }
                print(f"  [{i + 1}/{len(per_query)}] SKIP (refusal)")
                continue

            context_note = f"[Context: {len(item.get('chunk_ids_used', []))} chunks used]"

            try:
                # Faithfulness
                faith_prompt = _build_faithfulness_prompt(query, answer, context_note)
                faith_result = await judge.generate(
                    prompt=faith_prompt,
                    system_prompt="You are a precise evaluation judge. Follow the output format exactly.",
                    temperature=0.0,
                    max_tokens=200,
                )
                faith_parsed = _parse_judge_response(faith_result.text)
                total_input_tokens += faith_result.prompt_tokens
                total_output_tokens += faith_result.completion_tokens

                # Relevance
                rel_prompt = _build_relevance_prompt(query, answer, expected)
                rel_result = await judge.generate(
                    prompt=rel_prompt,
                    system_prompt="You are a precise evaluation judge. Follow the output format exactly.",
                    temperature=0.0,
                    max_tokens=200,
                )
                rel_parsed = _parse_judge_response(rel_result.text)
                total_input_tokens += rel_result.prompt_tokens
                total_output_tokens += rel_result.completion_tokens

                item["judge"] = {
                    "faithfulness_score": faith_parsed["score"],
                    "faithfulness_reasoning": faith_parsed["reasoning"],
                    "relevance_score": rel_parsed["score"],
                    "relevance_reasoning": rel_parsed["reasoning"],
                    "judge_model": model,
                }

                if faith_parsed["score"] is not None:
                    faith_scores.append(faith_parsed["score"])
                if rel_parsed["score"] is not None:
                    rel_scores.append(rel_parsed["score"])

                faith_s = f"faith={faith_parsed['score']}" if faith_parsed["score"] else "faith=ERR"
                rel_s = f"rel={rel_parsed['score']}" if rel_parsed["score"] else "rel=ERR"
                print(
                    f"  [{i + 1}/{len(per_query)}] {faith_s} {rel_s} ({faith_result.latency_ms + rel_result.latency_ms:.0f}ms)"
                )

            except Exception as e:
                errors += 1
                item["judge"] = {
                    "faithfulness_score": None,
                    "faithfulness_reasoning": f"Error: {e}",
                    "relevance_score": None,
                    "relevance_reasoning": f"Error: {e}",
                    "judge_model": model,
                }
                print(f"  [{i + 1}/{len(per_query)}] ERROR: {e}")

            if (i + 1) % checkpoint_interval == 0:
                _save_checkpoint(checkpoint_path, per_query, i + 1)
                print(f"  >>> Checkpoint: {i + 1}/{len(per_query)}")

        elapsed = time.perf_counter() - t_total

        mean_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
        mean_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0
        faith_pass = (
            sum(1 for s in faith_scores if s >= 4) / len(faith_scores) if faith_scores else 0
        )
        rel_pass = sum(1 for s in rel_scores if s >= 4) / len(rel_scores) if rel_scores else 0

        print("\n" + "=" * 60)
        print(f"JUDGE RESULTS ({model})")
        print("=" * 60)
        print(f"  Queries judged: {len(faith_scores)}")
        print(f"  Errors: {errors}")
        print(f"  Mean faithfulness: {mean_faith:.2f} / 5.0")
        print(f"  Faithfulness pass rate (>=4): {faith_pass:.1%}")
        print(f"  Mean relevance: {mean_rel:.2f} / 5.0")
        print(f"  Relevance pass rate (>=4): {rel_pass:.1%}")
        print(f"  Total tokens: {total_input_tokens} in + {total_output_tokens} out")
        print(f"  Time: {elapsed:.1f}s")

        by_cat = defaultdict(lambda: {"faith": [], "rel": []})
        for item in per_query:
            j = item.get("judge", {})
            cat = item.get("category", "unknown")
            if j.get("faithfulness_score") is not None:
                by_cat[cat]["faith"].append(j["faithfulness_score"])
            if j.get("relevance_score") is not None:
                by_cat[cat]["rel"].append(j["relevance_score"])

        print("\nPer-category:")
        for cat in sorted(by_cat.keys()):
            f_scores = by_cat[cat]["faith"]
            r_scores = by_cat[cat]["rel"]
            mf = sum(f_scores) / len(f_scores) if f_scores else 0
            mr = sum(r_scores) / len(r_scores) if r_scores else 0
            print(f"  {cat:22s}: faith={mf:.2f} rel={mr:.2f} (n={len(f_scores)})")

        judge_summary = {
            "judge_model": model,
            "num_judged": len(faith_scores),
            "num_errors": errors,
            "mean_faithfulness": round(mean_faith, 4),
            "faithfulness_pass_rate": round(faith_pass, 4),
            "mean_relevance": round(mean_rel, 4),
            "relevance_pass_rate": round(rel_pass, 4),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "elapsed_seconds": round(elapsed, 1),
        }

        data["judge_summary"] = judge_summary
        data["per_query"] = per_query

        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to: {output}")

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # MLflow
        try:
            from contextual_research_agent.common.settings import get_settings  # noqa: PLC0415

            settings = get_settings()
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            mlflow.set_experiment(experiment_name)

            auto_run_name = run_name or f"judge_{model.replace('/', '_')}"
            with mlflow.start_run(run_name=auto_run_name):
                mlflow.log_params(
                    {
                        "judge/model": model,
                        "judge/num_queries": len(per_query),
                        "judge/input_file": input,
                    }
                )
                mlflow.log_metric("judge/mean_faithfulness", mean_faith)
                mlflow.log_metric("judge/faithfulness_pass_rate", faith_pass)
                mlflow.log_metric("judge/mean_relevance", mean_rel)
                mlflow.log_metric("judge/relevance_pass_rate", rel_pass)

                for cat in sorted(by_cat.keys()):
                    f_scores = by_cat[cat]["faith"]
                    r_scores = by_cat[cat]["rel"]
                    if f_scores:
                        mlflow.log_metric(
                            f"judge/{cat}/faithfulness", sum(f_scores) / len(f_scores)
                        )
                    if r_scores:
                        mlflow.log_metric(f"judge/{cat}/relevance", sum(r_scores) / len(r_scores))

            print(f"Logged to MLflow: {experiment_name}/{auto_run_name}")
        except Exception as e:
            print(f"MLflow logging failed: {e}")

        await judge.close()

    asyncio.run(_run())


def _build_faithfulness_prompt(query: str, answer: str, context_note: str) -> str:
    return f"""\
You are evaluating a RAG system's answer for FAITHFULNESS.

Faithfulness means: the answer addresses the question using plausible, \
consistent information. Since you don't have the original retrieved context, \
evaluate whether the answer is internally consistent, makes specific \
claims with citations, and avoids obvious fabrication.

Question: {query}

Answer to evaluate:
{answer}

{context_note}

Rate faithfulness on a scale of 1-5:
1 = Contains clearly fabricated or contradictory claims
2 = Some claims seem unsupported or vague
3 = Mostly plausible, minor concerns
4 = Consistent and specific, cites sources
5 = Highly specific with citations, internally consistent

Respond in this exact format:
SCORE: <number 1-5>
REASONING: <1-2 sentences>"""


def _build_relevance_prompt(query: str, answer: str, expected: str) -> str:
    return f"""\
You are evaluating a RAG system's answer for RELEVANCE.

Relevance means: the answer directly addresses the question asked.

Question: {query}

Expected answer (reference):
{expected}

Generated answer:
{answer}

Rate relevance on a scale of 1-5:
1 = Completely irrelevant
2 = Partially relevant, addresses a related but different question
3 = Relevant but incomplete or tangential
4 = Relevant and mostly complete
5 = Directly and completely addresses the question

Respond in this exact format:
SCORE: <number 1-5>
REASONING: <1-2 sentences>"""


def _parse_judge_response(text: str) -> dict[str, Any]:
    score = None
    reasoning = ""

    score_match = re.search(r"SCORE:\s*(\d(?:\.\d)?)", text)
    if score_match:
        score = float(score_match.group(1))
        score = max(1.0, min(5.0, score))

    reasoning_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()[:500]

    if score is None:
        nums = re.findall(r"\b([1-5])\b", text)
        if nums:
            score = float(nums[0])
        reasoning = reasoning or text.strip()[:500]

    return {"score": score, "reasoning": reasoning}


def _save_checkpoint(path: Path, per_query: list, num_completed: int) -> None:
    judged = [item.get("judge", {}) for item in per_query[:num_completed]]
    checkpoint = {"num_completed": num_completed, "judged": judged}
    path.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")
