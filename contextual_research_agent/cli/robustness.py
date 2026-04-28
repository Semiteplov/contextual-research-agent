from __future__ import annotations

import asyncio

from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.generation.robustness_eval import run_robustness_evaluation

logger = get_logger(__name__)


def robustness_eval(  # noqa: PLR0913
    eval_set: str,
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
) -> None:
    """Run robustness evaluation with degraded context.

    Tests the generation pipeline's behavior when context is:
    - empty: no context at all
    - random: irrelevant chunks from the corpus
    - partial: chunks from the correct paper but wrong sections

    Args:
        eval_set: Path to eval set JSON (peft_gold_v3_mapped.json).
        collection: Qdrant collection name.
        scenario: "empty", "random", "partial", or "all".
        output_dir: Directory for result files.
        max_queries: Limit queries per scenario (for quick tests).
        seed: Random seed for reproducibility.
        n_context_chunks: Number of chunks for random/partial context.
        llm_provider: LLM backend.
        llm_model: Model name.
        llm_host: LLM server URL.
        skip_judge: Skip LLM-as-judge evaluation.
        judge_model: Separate model for judge.
        judge_host: Separate host for judge.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Examples:
        # Quick test — 20 queries, empty only, no judge
        python main.py robustness-eval eval/peft_gold_v3_mapped.json \\
            --scenario=empty --max-queries=20 --skip-judge

        # Full run — all scenarios
        python main.py robustness-eval eval/peft_gold_v3_mapped.json \\
            --scenario=all --skip-judge

        # With judge on random scenario
        python main.py robustness-eval eval/peft_gold_v3_mapped.json \\
            --scenario=random \\
            --judge-model=qwen3:8b
    """

    async def _run():
        return await run_robustness_evaluation(
            eval_set_path=eval_set,
            collection=collection,
            scenario=scenario,
            output_dir=output_dir,
            max_queries=max_queries,
            seed=seed,
            n_context_chunks=n_context_chunks,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_host=llm_host,
            skip_judge=skip_judge,
            judge_model=judge_model,
            judge_host=judge_host,
            experiment_name=experiment_name,
            run_name=run_name,
        )

    asyncio.run(_run())
