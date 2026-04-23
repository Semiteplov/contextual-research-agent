from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import mlflow

from contextual_research_agent.agent.llm import LLMProvider, create_llm_provider
from contextual_research_agent.cli.retrieval import _build_pipeline
from contextual_research_agent.common.logging import get_logger
from contextual_research_agent.common.settings import get_settings
from contextual_research_agent.generation.config import (
    CognitiveMode,
    GenerationConfig,
)
from contextual_research_agent.generation.evaluation import (
    GenerationEvaluator,
    GenerationMetrics,
)
from contextual_research_agent.generation.pipeline import GenerationPipeline, RAGResponse
from contextual_research_agent.ingestion.embeddings.hf_embedder import create_hf_embedder
from contextual_research_agent.retrieval.metrics import (
    IRQualityMetrics,
    aggregate_ir_metrics,
    compute_ir_metrics,
)

logger = get_logger(__name__)


def generate(  # noqa: PLR0913
    question: str,
    collection: str = "peft_hybrid",
    mode: str | None = None,
    llm_provider: str = "ollama",
    llm_model: str = "qwen3:8b",
    llm_host: str = "http://localhost:11434",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    rerank: bool = True,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    device: str | None = None,
    channels: str = "dense,sparse,graph_entity,paper_level",
    verbose: bool = False,
) -> None:
    """Run full RAG pipeline: retrieval → generation → print answer.

    Args:
        question: Query text.
        collection: Qdrant collection name.
        mode: Cognitive mode override (factual_qa, summarization, comparison,
              critical_review, methodological_audit, idea_generation).
        llm_provider: LLM backend (ollama | llama_cpp).
        llm_model: Model name/tag.
        llm_host: LLM server URL.
        temperature: Generation temperature.
        max_tokens: Max generation tokens.
        embedding_model: Dense embedding model.
        rerank: Enable cross-encoder reranking.
        rerank_model: Reranker model name.
        device: Device for models (cuda/cpu/None=auto).
        channels: Comma-separated retrieval channels.
        verbose: Show full context and prompts.
    """

    async def _run() -> None:
        channel_list = _parse_channels(channels)

        retrieval_pipeline, retrieval_config = await _build_pipeline(
            collection=collection,
            embedding_model=embedding_model,
            rerank_enabled=rerank,
            rerank_model=rerank_model,
            device=device,
            enabled_channels=channel_list,
        )

        llm = _create_llm(llm_provider, llm_model, llm_host)
        gen_config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            auto_detect_mode=mode is None,
            default_mode=CognitiveMode(mode) if mode else CognitiveMode.FACTUAL_QA,
        )
        gen_pipeline = GenerationPipeline(llm=llm, config=gen_config)

        retrieval_result = await retrieval_pipeline.retrieve(query=question)

        response = await gen_pipeline.generate(
            retrieval_result=retrieval_result,
            mode=mode,
        )

        _print_rag_response(response, verbose=verbose)

        await llm.close()

    asyncio.run(_run())


def evaluate_generation(  # noqa: PLR0913
    eval_set: str,
    collection: str = "peft_hybrid",
    llm_provider: str = "ollama",
    llm_model: str = "qwen3:8b",
    llm_host: str = "http://localhost:11434",
    judge_provider: str | None = None,
    judge_model: str | None = None,
    judge_host: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    rerank: bool = True,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    device: str | None = None,
    channels: str = "dense,sparse,graph_entity,paper_level",
    output: str | None = None,
    experiment_name: str = "generation",
    run_name: str | None = None,
    skip_judge: bool = False,
    max_queries: int | None = None,
) -> None:
    """Evaluate full RAG pipeline on labeled query set.

    Runs retrieval + generation for each query, computes:
    - Semantic similarity (answer vs expected_answer)
    - Faithfulness (LLM-as-judge)
    - Relevance (LLM-as-judge)
    - Per-category breakdown

    Logs all metrics to MLflow.

    Args:
        eval_set: Path to evaluation JSON.
        collection: Qdrant collection name.
        llm_provider: Generation LLM backend.
        llm_model: Generation model.
        llm_host: Generation LLM server URL.
        judge_provider: Judge LLM backend (defaults to same as generation).
        judge_model: Judge model (defaults to same as generation).
        judge_host: Judge LLM server URL (defaults to same as generation).
        temperature: Generation temperature.
        max_tokens: Max generation tokens.
        embedding_model: Dense embedding model.
        rerank: Enable reranking.
        rerank_model: Reranker model.
        device: Device for models.
        channels: Comma-separated retrieval channels.
        output: Path to save detailed results JSON.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        skip_judge: Skip LLM-as-judge evaluations (only compute semantic similarity).
        max_queries: Limit number of queries to evaluate (for quick testing).
    """

    async def _run() -> None:
        # Load eval data
        eval_data = _load_generation_eval_set(eval_set)
        if max_queries:
            eval_data = eval_data[:max_queries]

        channel_list = _parse_channels(channels)

        # Build retrieval pipeline
        retrieval_pipeline, retrieval_config = await _build_pipeline(
            collection=collection,
            embedding_model=embedding_model,
            rerank_enabled=rerank,
            rerank_model=rerank_model,
            device=device,
            enabled_channels=channel_list,
        )

        # Build generation pipeline
        llm = _create_llm(llm_provider, llm_model, llm_host)
        gen_config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            auto_detect_mode=True,
        )
        gen_pipeline = GenerationPipeline(llm=llm, config=gen_config)

        # Build evaluator
        embedder = create_hf_embedder(model=embedding_model, device=device)

        judge_llm: LLMProvider | None = None
        if not skip_judge:
            judge_llm = _create_llm(
                judge_provider or llm_provider,
                judge_model or llm_model,
                judge_host or llm_host,
            )

        evaluator = GenerationEvaluator(
            embedder=embedder,
            judge_llm=judge_llm,
        )

        # Run evaluation loop
        per_query_gen: list[GenerationMetrics] = []
        per_query_ir: list[IRQualityMetrics] = []
        per_query_results: list[dict[str, Any]] = []
        latencies: list[float] = []

        checkpoint_interval = 25
        checkpoint_path = Path(output).with_suffix(".checkpoint.json") if output else None

        start_idx = 0
        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                per_query_results = checkpoint_data.get("per_query", [])
                start_idx = len(per_query_results)
                print(f"  Resuming from checkpoint: {start_idx}/{len(eval_data)} queries done")
            except Exception as e:
                logger.warning("Failed to load checkpoint, starting fresh: %s", e)
                start_idx = 0
                per_query_results = []

        t_total = time.perf_counter()

        for i, item in enumerate(eval_data):
            if i < start_idx:
                continue

            query_text = item["query"]
            expected_answer = item.get("expected_answer", "")
            relevant_ids = set(item.get("relevant_ids", []))
            category = item.get("category", "")

            try:
                # Retrieval
                retrieval_result = await retrieval_pipeline.retrieve(query=query_text)
                latencies.append(retrieval_result.total_latency_ms)

                # IR metrics
                retrieved_ids = [c.chunk_id for c in retrieval_result.candidates]
                ir_metrics = compute_ir_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    k_values=[1, 3, 5, 10],
                    query=query_text,
                )
                per_query_ir.append(ir_metrics)

                # Generation
                response = await gen_pipeline.generate(retrieval_result=retrieval_result)

                # Evaluation
                gen_metrics = await evaluator.evaluate_single(
                    response=response,
                    expected_answer=expected_answer if expected_answer else None,
                    context=retrieval_result.context,
                    category=category,
                )
                per_query_gen.append(gen_metrics)

                per_query_results.append(
                    {
                        "query": query_text,
                        "category": category,
                        "expected_answer": expected_answer,
                        "generated_answer": response.answer,
                        "mode": response.mode.value,
                        "model": response.model,
                        "ir_metrics": {
                            "mrr": ir_metrics.mrr,
                            "recall_at_5": ir_metrics.recall_at_k.get(5, 0),
                            "ndcg_at_10": ir_metrics.ndcg_at_k.get(10, 0),
                        },
                        "gen_metrics": gen_metrics.to_dict(),
                        "chunk_ids_used": response.chunk_ids_used,
                        "llm_latency_ms": response.llm_latency_ms,
                        "retrieval_latency_ms": response.retrieval_latency_ms,
                    }
                )

                sim_str = (
                    f"sim={gen_metrics.semantic_similarity:.3f}"
                    if gen_metrics.semantic_similarity is not None
                    else "sim=N/A"
                )
                faith_str = (
                    f"faith={gen_metrics.faithfulness_score:.0f}"
                    if gen_metrics.faithfulness_score is not None
                    else "faith=N/A"
                )
                print(
                    f"  [{i + 1}/{len(eval_data)}] "
                    f"MRR={ir_metrics.mrr:.3f} "
                    f"{sim_str} "
                    f"{faith_str} "
                    f"({'REFUSAL' if gen_metrics.is_refusal else 'OK'}) "
                    f"({response.llm_latency_ms:.0f}ms)"
                )

            except Exception as e:
                logger.error("Query %d failed: %s — %s", i + 1, query_text[:60], e)
                per_query_results.append(
                    {
                        "query": query_text,
                        "category": category,
                        "expected_answer": expected_answer,
                        "generated_answer": f"ERROR: {e}",
                        "mode": "error",
                        "model": llm_model,
                        "ir_metrics": {"mrr": 0, "recall_at_5": 0, "ndcg_at_10": 0},
                        "gen_metrics": GenerationMetrics(
                            query=query_text, category=category
                        ).to_dict(),
                        "chunk_ids_used": [],
                        "llm_latency_ms": 0,
                        "retrieval_latency_ms": 0,
                    }
                )
                print(f"  [{i + 1}/{len(eval_data)}] ERROR: {e}")

            if checkpoint_path and (i + 1) % checkpoint_interval == 0:
                _save_checkpoint(checkpoint_path, per_query_results)
                print(
                    f"  >>> Checkpoint saved: {len(per_query_results)}/{len(eval_data)} "
                    f"queries → {checkpoint_path}"
                )

        elapsed_total = time.perf_counter() - t_total

        ir_agg = aggregate_ir_metrics(per_query_ir, latencies)
        gen_agg = evaluator.aggregate(per_query_gen)

        print("\n" + "=" * 70)
        print("RETRIEVAL METRICS")
        print("=" * 70)
        for key, value in ir_agg.to_dict().items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("\n" + "=" * 70)
        print("GENERATION METRICS")
        print("=" * 70)
        for key, value in gen_agg.to_dict().items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k2, v2 in value.items():
                    print(f"    {k2}: {v2}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print(f"\nTotal time: {elapsed_total:.1f}s")

        try:
            settings = get_settings()
            mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
            mlflow.set_experiment(experiment_name)

            auto_run_name = run_name or f"{llm_model}_{'rerank' if rerank else 'norerank'}"

            with mlflow.start_run(run_name=auto_run_name):
                # Log retrieval config
                mlflow.log_params(retrieval_config.to_mlflow_params())
                mlflow.log_params(gen_config.to_mlflow_params())
                mlflow.log_params(
                    {
                        "llm/provider": llm_provider,
                        "llm/model": llm_model,
                        "eval/set": eval_set,
                        "eval/num_queries": len(eval_data),
                        "eval/channels": channels,
                    }
                )

                # Log IR metrics
                for key, value in ir_agg.to_dict().items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"ir/{_sanitize_metric_name(key)}", value)

                # Log generation metrics
                for key, value in gen_agg.to_dict().items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"gen/{_sanitize_metric_name(key)}", value)

                # Log category breakdown
                for cat, cat_metrics in gen_agg.category_metrics.items():
                    for metric_name, metric_val in cat_metrics.items():
                        if isinstance(metric_val, (int, float)):
                            mlflow.log_metric(
                                f"gen/{cat}/{_sanitize_metric_name(metric_name)}", metric_val
                            )

                logger.info(
                    "Logged to MLflow: experiment=%s, run=%s", experiment_name, auto_run_name
                )

        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "config": {
                    "retrieval": retrieval_config.to_mlflow_params(),
                    "generation": gen_config.to_mlflow_params(),
                    "llm": {"provider": llm_provider, "model": llm_model},
                },
                "ir_aggregated": ir_agg.to_dict(),
                "gen_aggregated": gen_agg.to_dict(),
                "per_query": per_query_results,
            }
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\nDetailed results saved to: {output}")

        # Cleanup
        await llm.close()
        if judge_llm and judge_llm is not llm:
            await judge_llm.close()

    asyncio.run(_run())


def _parse_channels(channels: str | list | tuple) -> list[str]:
    if isinstance(channels, list):
        return channels
    if isinstance(channels, tuple):
        return list(channels)
    return channels.split(",")


def _create_llm(provider: str, model: str, host: str) -> LLMProvider:
    """Create LLM provider from CLI args."""
    return create_llm_provider(provider=provider, model=model, host=host)


def _load_generation_eval_set(path: str) -> list[dict[str, Any]]:
    """Load evaluation set with expected_answer field."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    for i, item in enumerate(data):
        if "query" not in item:
            raise ValueError(f"Item {i} missing 'query' key")
    return data


def _print_rag_response(response: RAGResponse, verbose: bool = False) -> None:
    """Pretty-print a RAG response."""
    print(f"\nQuery: {response.query}")
    print(f"Mode: {response.mode.value}")
    print(f"Model: {response.model}")
    print(f"Intent: {response.intent}")
    print("=" * 70)

    print(f"\n{response.answer}")

    print("\n" + "-" * 70)
    print("Provenance")
    print("-" * 70)
    print(f"  Chunks used: {response.num_chunks_used}")
    print(f"  Documents: {', '.join(response.document_ids_used[:5])}")
    print(f"  Retrieval: {response.retrieval_latency_ms:.0f}ms")
    print(f"  Generation: {response.llm_latency_ms:.0f}ms")
    print(f"  Total: {response.total_latency_ms:.0f}ms")
    print(f"  Tokens: {response.prompt_tokens} prompt + {response.completion_tokens} completion")

    if verbose:
        print("\n" + "-" * 70)
        print("System Prompt")
        print("-" * 70)
        print(response.system_prompt)
        print("\n" + "-" * 70)
        print("User Prompt")
        print("-" * 70)
        print(response.user_prompt[:3000])
        if len(response.user_prompt) > 3000:
            print(f"\n... [{len(response.user_prompt) - 3000} chars truncated]")

    print()


def _sanitize_metric_name(name: str) -> str:
    return name.replace("@", "_at_")


def _save_checkpoint(path: Path, per_query_results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "per_query": per_query_results,
        "num_completed": len(per_query_results),
    }
    path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
