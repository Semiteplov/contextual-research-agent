from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.common.settings import get_settings
from contextual_research_agent.generation.openrouter_provider import OpenRouterProvider

logger = get_logger(__name__)


_REFUSAL_PATTERNS = [
    re.compile(r"(?i)the provided (sources|context|passages?) do not contain"),
    re.compile(r"(?i)insufficient information"),
    re.compile(r"(?i)cannot (be )?answer(ed)?.*based on"),
    re.compile(r"(?i)the context does not (provide|contain|mention)"),
    re.compile(r"(?i)not enough information"),
    re.compile(r"(?i)no relevant information"),
]


def _is_refusal(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    return any(pattern.search(answer) for pattern in _REFUSAL_PATTERNS)


def _load_baseline(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    per_query = data.get("per_query", [])

    results = []
    for item in per_query:
        judge = item.get("judge", {})
        gen_metrics = item.get("gen_metrics", {})

        results.append(
            {
                "query": item.get("query", ""),
                "category": item.get("category", "unknown"),
                "expected_answer": item.get("expected_answer", ""),
                "answer": item.get("generated_answer", ""),
                "is_refusal": gen_metrics.get("is_refusal", False)
                or _is_refusal(item.get("generated_answer", "")),
                "faithfulness": judge.get("faithfulness_score"),
                "relevance": judge.get("relevance_score"),
                "latency_ms": item.get("total_latency_ms", 0),
                "mode": item.get("mode", "unknown"),
                "source": "baseline",
            }
        )

    return results


def _load_agent(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if "per_query" in data:
        items = data["per_query"]
        is_judged = True
    elif "results" in data:
        items = data["results"]
        is_judged = False
    else:
        raise ValueError(f"Unknown agent results format in {path}")

    results = []
    for item in items:
        answer = item.get("generated_answer", "")

        if is_judged:
            judge = item.get("judge", {})
            gen_metrics = item.get("gen_metrics", {})
            chunk_ids = item.get("chunk_ids_used", [])

            results.append(
                {
                    "query": item.get("query", ""),
                    "category": item.get("category", "unknown"),
                    "expected_answer": item.get("expected_answer", ""),
                    "answer": answer,
                    "is_refusal": gen_metrics.get("is_refusal", False) or _is_refusal(answer),
                    "faithfulness": judge.get("faithfulness_score"),
                    "relevance": judge.get("relevance_score"),
                    "latency_ms": item.get("total_latency_ms", 0),
                    "mode": item.get("mode", "unknown"),
                    "chunks_retrieved": len(chunk_ids),
                    "source": "agent",
                }
            )
        else:
            debug = item.get("debug", {})
            trace = item.get("trace", {})
            critic = trace.get("critic", {}) if trace else {}
            critic_fb = critic.get("feedback", {})

            results.append(
                {
                    "query": item.get("query", ""),
                    "category": item.get("category", "unknown"),
                    "expected_answer": item.get("expected_answer", ""),
                    "answer": answer,
                    "is_refusal": _is_refusal(answer),
                    "faithfulness": None,
                    "relevance": None,
                    "latency_ms": debug.get("total_ms", 0),
                    "mode": debug.get("mode", "unknown"),
                    "complexity": debug.get("complexity", ""),
                    "intent": debug.get("intent", ""),
                    "critic_verdict": critic_fb.get("verdict", ""),
                    "critic_faithfulness": critic_fb.get("faithfulness_score"),
                    "critic_completeness": critic_fb.get("completeness_score"),
                    "retry_count": debug.get("retry_count", 0),
                    "chunks_retrieved": debug.get("chunks_retrieved", 0),
                    "source": "agent",
                }
            )

    return results


def _compute_metrics(results: list[dict], label: str) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"label": label, "total": 0}

    refusals = sum(1 for r in results if r["is_refusal"])
    non_refusals = total - refusals

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]

    faith_scores = [
        r["faithfulness"] for r in results if not r["is_refusal"] and r["faithfulness"] is not None
    ]
    rel_scores = [
        r["relevance"] for r in results if not r["is_refusal"] and r["relevance"] is not None
    ]
    retries = [r.get("retry_count", 0) for r in results]
    critic_faith = [
        r["critic_faithfulness"] for r in results if r.get("critic_faithfulness") is not None
    ]

    metrics: dict[str, Any] = {
        "label": label,
        "total_queries": total,
        "refusal_count": refusals,
        "refusal_rate": round(refusals / total, 4),
        "non_refusal_count": non_refusals,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0,
    }

    if faith_scores:
        metrics["mean_faithfulness"] = round(sum(faith_scores) / len(faith_scores), 3)
        metrics["faithfulness_pass_rate"] = round(
            sum(1 for s in faith_scores if s >= 4) / len(faith_scores), 4
        )
        metrics["num_judged"] = len(faith_scores)

    if rel_scores:
        metrics["mean_relevance"] = round(sum(rel_scores) / len(rel_scores), 3)
        metrics["relevance_pass_rate"] = round(
            sum(1 for s in rel_scores if s >= 4) / len(rel_scores), 4
        )

    if any(r.get("retry_count") is not None for r in results):
        retry_count = sum(1 for r in retries if r > 0)
        metrics["retry_count"] = retry_count
        metrics["retry_rate"] = round(retry_count / total, 4)

    if critic_faith:
        metrics["critic_mean_faithfulness"] = round(sum(critic_faith) / len(critic_faith), 3)

    return metrics


def _compute_per_category(
    results: list[dict],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Per-category breakdown."""
    categories: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        categories[r["category"]].append(r)

    return {
        cat: _compute_metrics(cat_results, f"{label}/{cat}")
        for cat, cat_results in sorted(categories.items())
    }


def _print_comparison(
    baseline_metrics: dict[str, Any],
    agent_metrics: dict[str, Any],
    baseline_per_cat: dict[str, dict],
    agent_per_cat: dict[str, dict],
) -> None:
    b = baseline_metrics
    a = agent_metrics

    print("\n" + "=" * 80)
    print("COMPARISON: Single-Pipeline vs Multi-Agent")
    print("=" * 80)

    header = f"{'Metric':<30} {'Single-Pipeline':>18} {'Multi-Agent':>18} {'Delta':>12}"
    print(header)
    print("-" * 80)

    def _row(name: str, bval: Any, aval: Any, fmt: str = ".1f", higher_better: bool = True):
        if bval is None and aval is None:
            return
        bstr = f"{bval:{fmt}}" if bval is not None else "N/A"
        astr = f"{aval:{fmt}}" if aval is not None else "N/A"
        if bval is not None and aval is not None:
            delta = aval - bval
            sign = "+" if delta >= 0 else ""
            indicator = (
                "↑"
                if (delta > 0 and higher_better) or (delta < 0 and not higher_better)
                else "↓"
                if delta != 0
                else "="
            )
            dstr = f"{sign}{delta:{fmt}} {indicator}"
        else:
            dstr = ""
        print(f"  {name:<28} {bstr:>18} {astr:>18} {dstr:>12}")

    _row("Total queries", b.get("total_queries"), a.get("total_queries"), "d")
    _row("Refusal rate", b.get("refusal_rate"), a.get("refusal_rate"), ".1%", False)
    _row("Mean latency (ms)", b.get("mean_latency_ms"), a.get("mean_latency_ms"), ".0f", False)
    print()
    _row("Mean faithfulness", b.get("mean_faithfulness"), a.get("mean_faithfulness"), ".3f")
    _row(
        "Faith. pass rate (≥4)",
        b.get("faithfulness_pass_rate"),
        a.get("faithfulness_pass_rate"),
        ".1%",
    )
    _row("Mean relevance", b.get("mean_relevance"), a.get("mean_relevance"), ".3f")
    _row("Rel. pass rate (≥4)", b.get("relevance_pass_rate"), a.get("relevance_pass_rate"), ".1%")
    _row("Num judged", b.get("num_judged"), a.get("num_judged"), "d")
    print()
    _row("Retry rate", None, a.get("retry_rate"), ".1%")
    _row("Retry count", None, a.get("retry_count"), "d")
    _row("Critic mean faith.", None, a.get("critic_mean_faithfulness"), ".3f")

    all_cats = sorted(set(list(baseline_per_cat.keys()) + list(agent_per_cat.keys())))
    if all_cats:
        print("\n" + "-" * 80)
        print("PER-CATEGORY REFUSAL RATES")
        print("-" * 80)
        cat_header = f"  {'Category':<22} {'Single':>10} {'Agent':>10} {'Delta':>10}"
        print(cat_header)
        for cat in all_cats:
            b_cat = baseline_per_cat.get(cat, {})
            a_cat = agent_per_cat.get(cat, {})
            br = b_cat.get("refusal_rate")
            ar = a_cat.get("refusal_rate")
            bstr = f"{br:.1%}" if br is not None else "N/A"
            astr = f"{ar:.1%}" if ar is not None else "N/A"
            dstr = ""
            if br is not None and ar is not None:
                d = ar - br
                dstr = f"{'+' if d >= 0 else ''}{d:.1%}"
            print(f"  {cat:<22} {bstr:>10} {astr:>10} {dstr:>10}")

        has_faith = any(
            agent_per_cat.get(c, {}).get("mean_faithfulness") is not None for c in all_cats
        )
        if has_faith:
            print("\n" + "-" * 80)
            print("PER-CATEGORY FAITHFULNESS")
            print("-" * 80)
            cat_header = f"  {'Category':<22} {'Single':>10} {'Agent':>10} {'Delta':>10}"
            print(cat_header)
            for cat in all_cats:
                b_cat = baseline_per_cat.get(cat, {})
                a_cat = agent_per_cat.get(cat, {})
                bf = b_cat.get("mean_faithfulness")
                af = a_cat.get("mean_faithfulness")
                bstr = f"{bf:.2f}" if bf is not None else "N/A"
                astr = f"{af:.2f}" if af is not None else "N/A"
                dstr = ""
                if bf is not None and af is not None:
                    d = af - bf
                    dstr = f"{'+' if d >= 0 else ''}{d:.2f}"
                print(f"  {cat:<22} {bstr:>10} {astr:>10} {dstr:>10}")

    print("\n" + "=" * 80)


async def _run_judge_on_agent_results(
    agent_results: list[dict[str, Any]],
    model: str,
    api_key: str,
    host: str,
) -> None:
    judge = OpenRouterProvider(model=model, api_key=api_key, host=host)

    non_refusal = [r for r in agent_results if not r["is_refusal"]]
    logger.info("Running judge on %d non-refusal agent results", len(non_refusal))

    score_pattern = re.compile(r"SCORE:\s*(\d(?:\.\d)?)")

    for i, r in enumerate(non_refusal):
        try:
            faith_prompt = (
                f"You are evaluating a RAG system's answer for FAITHFULNESS.\n\n"
                f"Faithfulness means: the answer uses plausible, consistent information "
                f"with proper citations. Evaluate internal consistency and specificity.\n\n"
                f"Question: {r['query']}\n\n"
                f"Answer: {r['answer']}\n\n"
                f"Rate 1-5:\n"
                f"1=Fabricated 2=Vague 3=Mostly plausible 4=Specific with citations 5=Highly faithful\n\n"
                f"Respond: SCORE: <number>\nREASONING: <1-2 sentences>"
            )
            faith_resp = await judge.generate(
                prompt=faith_prompt,
                system_prompt="You are a precise evaluation judge.",
                temperature=0.0,
                max_tokens=200,
            )
            faith_match = score_pattern.search(faith_resp.text)
            if faith_match:
                r["faithfulness"] = float(faith_match.group(1))

            rel_prompt = (
                f"You are evaluating a RAG system's answer for RELEVANCE.\n\n"
                f"Question: {r['query']}\n"
                f"Expected: {r['expected_answer']}\n"
                f"Generated: {r['answer']}\n\n"
                f"Rate 1-5:\n"
                f"1=Irrelevant 2=Tangential 3=Partial 4=Mostly relevant 5=Complete\n\n"
                f"Respond: SCORE: <number>\nREASONING: <1-2 sentences>"
            )
            rel_resp = await judge.generate(
                prompt=rel_prompt,
                system_prompt="You are a precise evaluation judge.",
                temperature=0.0,
                max_tokens=200,
            )
            rel_match = score_pattern.search(rel_resp.text)
            if rel_match:
                r["relevance"] = float(rel_match.group(1))

        except Exception as e:
            logger.warning("Judge failed for query %d: %s", i, e)

        if (i + 1) % 10 == 0:
            logger.info("Judge progress: %d/%d", i + 1, len(non_refusal))

    await judge.close()


def compare_agent_vs_pipeline(  # noqa: PLR0913
    baseline: str,
    agent: str,
    output: str = "eval/results/comparison.json",
    run_judge: bool = False,
    judge_model: str = "openai/gpt-5.4-mini",
    judge_host: str = "https://openrouter.ai/api/v1",
    api_key: str = "",
) -> None:
    """Compare single-pipeline vs multi-agent results.

    Args:
        baseline: Path to single-pipeline judged results JSON.
        agent: Path to multi-agent eval results JSON.
        output: Path to save comparison results.
        run_judge: Run GPT judge on agent results before comparing.
        judge_model: OpenRouter model for judge.
        judge_host: OpenRouter API host.
        api_key: OpenRouter API key (or from settings).

    Examples:
        # Compare (agent already judged):
        python main.py compare-agent-vs-pipeline \\
            --baseline=eval/results/gen_judged_gpt.json \\
            --agent=eval/results/agent_eval_judged.json

        # Compare with judge on-the-fly:
        python main.py compare-agent-vs-pipeline \\
            --baseline=eval/results/gen_judged_gpt.json \\
            --agent=eval/results/agent_eval_full.json \\
            --run-judge --judge-model=openai/gpt-5.4-mini
    """

    async def _run() -> None:
        baseline_results = _load_baseline(baseline)
        agent_results = _load_agent(agent)

        print(f"Loaded: {len(baseline_results)} baseline, {len(agent_results)} agent results")

        if run_judge:
            key = api_key
            if not key:
                try:
                    settings = get_settings()
                    key = settings.openrouter.api_key.get_secret_value()
                except Exception:
                    pass

            if not key:
                print("ERROR: No API key for judge. Skipping.")
            else:
                await _run_judge_on_agent_results(
                    agent_results,
                    judge_model,
                    key,
                    judge_host,
                )

        baseline_metrics = _compute_metrics(baseline_results, "single-pipeline")
        agent_metrics = _compute_metrics(agent_results, "multi-agent")

        baseline_per_cat = _compute_per_category(baseline_results, "single-pipeline")
        agent_per_cat = _compute_per_category(agent_results, "multi-agent")

        _print_comparison(baseline_metrics, agent_metrics, baseline_per_cat, agent_per_cat)

        output_data = {
            "baseline": {
                "file": baseline,
                "metrics": baseline_metrics,
                "per_category": baseline_per_cat,
            },
            "agent": {
                "file": agent,
                "metrics": agent_metrics,
                "per_category": agent_per_cat,
            },
        }

        if run_judge:
            agent_path = Path(agent)
            agent_data = json.loads(agent_path.read_text(encoding="utf-8"))

            for i, r in enumerate(agent_results):
                if i < len(agent_data.get("results", [])):
                    agent_data["results"][i]["judge"] = {
                        "faithfulness_score": r.get("faithfulness"),
                        "relevance_score": r.get("relevance"),
                        "judge_model": judge_model,
                    }

            judged_path = agent_path.with_name(agent_path.stem + "_judged" + agent_path.suffix)
            judged_path.write_text(
                json.dumps(agent_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nAgent results with judge scores saved: {judged_path}")

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Comparison saved: {output}")

    asyncio.run(_run())
