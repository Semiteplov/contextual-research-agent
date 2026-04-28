from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from contextual_research_agent.agent.service import ResearchAssistantService
from contextual_research_agent.agent.tracing.mlflow_logger import log_agent_batch, log_agent_trace
from contextual_research_agent.agent.tracing.trace import AgentTrace
from contextual_research_agent.common.logging import get_logger, setup_logging

logger = get_logger(__name__)


def agent_query(  # noqa: PLR0913
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
    trace_output: str | None = None,
    log_mlflow: bool = False,
) -> None:
    """Run a single query through the multi-agent system.

    Args:
        question: Query text.
        collection: Qdrant collection name.
        mode: Force cognitive mode (bypasses Router).
        llm_provider: LLM backend.
        llm_model: Model name.
        llm_host: LLM server URL.
        temperature: Generation temperature.
        max_tokens: Max generation tokens.
        embedding_model: Dense embedding model.
        rerank: Enable reranking.
        rerank_model: Reranker model.
        device: Device for models.
        channels: Comma-separated retrieval channels.
        verbose: Show full debug trace (chunks, prompts, latencies).
        trace_output: Path to save trace JSON.
        log_mlflow: Log trace to MLflow.

    Examples:
        python main.py agent-query "How does LoRA reduce parameters?"

        python main.py agent-query "Compare LoRA and QLoRA, and critique their methodology" \\
            --verbose --trace-output=trace.json

        python main.py agent-query "Explain LoRA" --mode=summarization
    """

    async def _run() -> None:
        service = await ResearchAssistantService.create(
            collection=collection,
            embedding_model=embedding_model,
            rerank=rerank,
            rerank_model=rerank_model,
            device=device,
            channels=channels,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_host=llm_host,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            result = await service.query(text=question, mode=mode)

            # Print answer
            print("\n" + "=" * 70)
            print("ANSWER")
            print("=" * 70)
            print(result.answer)

            # Print debug summary
            summary = result.debug_summary
            print("\n" + "-" * 70)
            print("DEBUG SUMMARY")
            print("-" * 70)
            print(f"  Intent:           {summary['intent']}")
            print(f"  Complexity:       {summary['complexity']}")
            print(f"  Mode:             {summary['mode']}")
            print(f"  Chunks retrieved: {summary['chunks_retrieved']}")
            print(f"  Critic verdict:   {summary['critic_verdict']}")
            print(f"  Retry count:      {summary['retry_count']}")
            print(f"  Status:           {summary['status']}")

            # Latency breakdown
            latency = result.latency_breakdown
            print("\n  Latency breakdown:")
            for node, ms in latency.items():
                print(f"    {node:15s}: {ms:>8.0f} ms")
            print(f"    {'TOTAL':15s}: {summary['total_ms']:>8.0f} ms")

            # Token usage
            tokens = summary.get("tokens", {})
            if tokens:
                print(
                    f"\n  Tokens: {tokens.get('prompt', 0)} prompt"
                    f" + {tokens.get('completion', 0)} completion"
                    f" = {tokens.get('total', 0)} total"
                )

            if verbose:
                _print_verbose(result)

            # Save trace
            if trace_output:
                trace_path = Path(trace_output)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(
                    json.dumps(result.trace.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"\n  Trace saved: {trace_output}")

            # MLflow
            if log_mlflow:
                log_agent_trace(result.trace)
                print("  Logged to MLflow")

        finally:
            await service.shutdown()

    asyncio.run(_run())


def agent_eval(  # noqa: PLR0913
    eval_set: str,
    collection: str = "peft_hybrid",
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
    output: str = "eval/results/agent_eval.json",
    max_queries: int | None = None,
    experiment_name: str = "multi_agent",
    run_name: str | None = None,
) -> None:
    """Evaluate multi-agent system on eval set.

    Runs all queries through the multi-agent graph and collects:
    - Per-query traces (routing decisions, critic verdicts, retries)
    - Aggregated metrics (latencies, retry rate, complexity distribution)
    - Comparison data for single-pipeline vs multi-agent

    Args:
        eval_set: Path to eval set JSON.
        output: Path to save results.
        max_queries: Limit queries for quick tests.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
    """

    async def _run() -> None:
        # Load eval set
        eval_data = json.loads(Path(eval_set).read_text(encoding="utf-8"))
        if max_queries:
            eval_data = eval_data[:max_queries]

        print(f"Loaded {len(eval_data)} queries from {eval_set}")

        # Initialize service
        service = await ResearchAssistantService.create(
            collection=collection,
            embedding_model=embedding_model,
            rerank=rerank,
            rerank_model=rerank_model,
            device=device,
            channels=channels,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_host=llm_host,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Checkpoint setup
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path.with_suffix(".checkpoint.json")

        # Load checkpoint
        start_idx = 0
        results: list[dict[str, Any]] = []
        if checkpoint_path.exists():
            try:
                cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                results = cp.get("results", [])
                start_idx = len(results)
                print(f"Resuming from checkpoint: {start_idx}/{len(eval_data)}")
            except Exception:
                start_idx = 0

        try:
            traces: list[AgentTrace] = []

            for i in range(start_idx, len(eval_data)):
                q = eval_data[i]
                query_text = q["query"]
                category = q.get("category", "unknown")
                expected = q.get("expected_answer", "")

                try:
                    result = await service.query(text=query_text)

                    entry = {
                        "query": query_text,
                        "category": category,
                        "expected_answer": expected,
                        "generated_answer": result.answer,
                        "debug": result.debug_summary,
                        "latency_breakdown": result.latency_breakdown,
                        "trace": result.trace.to_dict(),
                    }
                    traces.append(result.trace)

                except Exception as e:
                    logger.error("Query %d failed: %s", i, e)
                    entry = {
                        "query": query_text,
                        "category": category,
                        "expected_answer": expected,
                        "generated_answer": f"[ERROR: {e}]",
                        "debug": {"status": "failed", "error": str(e)},
                    }

                results.append(entry)

                # Progress
                done = len(results)
                if done % 5 == 0 or done == len(eval_data):
                    print(f"  [{done}/{len(eval_data)}] {category}: {query_text[:60]}...")

                # Checkpoint
                if done % 25 == 0:
                    checkpoint_path.write_text(
                        json.dumps({"results": results}, ensure_ascii=False),
                        encoding="utf-8",
                    )

            # Aggregate metrics
            completed = [r for r in results if r.get("debug", {}).get("status") == "completed"]
            failed = [r for r in results if r.get("debug", {}).get("status") == "failed"]
            retried = [r for r in results if r.get("debug", {}).get("retry_count", 0) > 0]

            complexities: dict[str, int] = {}
            for r in results:
                c = r.get("debug", {}).get("complexity", "unknown")
                complexities[c] = complexities.get(c, 0) + 1

            modes: dict[str, int] = {}
            for r in results:
                m = r.get("debug", {}).get("mode", "unknown")
                modes[m] = modes.get(m, 0) + 1

            total_latencies = [
                r.get("debug", {}).get("total_ms", 0)
                for r in results
                if r.get("debug", {}).get("total_ms", 0) > 0
            ]

            summary = {
                "total_queries": len(results),
                "completed": len(completed),
                "failed": len(failed),
                "retried": len(retried),
                "retry_rate": len(retried) / len(results) if results else 0,
                "mean_latency_ms": sum(total_latencies) / len(total_latencies)
                if total_latencies
                else 0,
                "complexity_distribution": complexities,
                "mode_distribution": modes,
            }

            # Print summary
            print("\n" + "=" * 60)
            print("MULTI-AGENT EVALUATION RESULTS")
            print("=" * 60)
            print(f"  Total queries:  {summary['total_queries']}")
            print(f"  Completed:      {summary['completed']}")
            print(f"  Failed:         {summary['failed']}")
            print(f"  Retried:        {summary['retried']} ({summary['retry_rate']:.1%})")
            print(f"  Mean latency:   {summary['mean_latency_ms']:.0f} ms")
            print("\n  Complexity distribution:")
            for c, count in sorted(complexities.items()):
                print(f"    {c:15s}: {count}")
            print("\n  Mode distribution:")
            for m, count in sorted(modes.items()):
                print(f"    {m:20s}: {count}")

            # Save results
            output_data = {
                "summary": summary,
                "config": {
                    "llm_model": llm_model,
                    "collection": collection,
                    "channels": channels,
                },
                "results": results,
            }
            output_path.write_text(
                json.dumps(output_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nResults saved: {output}")

            # Clean checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            # MLflow
            if traces:
                log_agent_batch(
                    traces=traces,
                    experiment_name=experiment_name,
                    run_name=run_name or f"agent_eval_{llm_model}",
                )
                print(f"Logged to MLflow: {experiment_name}")

        finally:
            await service.shutdown()

    asyncio.run(_run())


def agent_chat(  # noqa: PLR0913
    collection: str = "peft_hybrid",
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
    """Interactive chat with the multi-agent system.

    Type queries, get answers with debug info. Type 'quit' to exit.
    Type '/debug' to toggle verbose mode.
    Type '/trace' to save last trace to file.
    """

    async def _run() -> None:
        print("Initializing multi-agent system...")
        service = await ResearchAssistantService.create(
            collection=collection,
            embedding_model=embedding_model,
            rerank=rerank,
            rerank_model=rerank_model,
            device=device,
            channels=channels,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_host=llm_host,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        print("\nMulti-Agent Research Assistant ready.")
        print("Commands: /debug (toggle verbose), /trace (save last trace), quit\n")

        show_verbose = verbose
        last_result = None

        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    break

                if user_input == "/debug":
                    show_verbose = not show_verbose
                    print(f"  Verbose mode: {'ON' if show_verbose else 'OFF'}")
                    continue

                if user_input == "/trace" and last_result:
                    trace_path = Path("last_trace.json")
                    trace_path.write_text(
                        json.dumps(last_result.trace.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(f"  Trace saved: {trace_path}")
                    continue

                result = await service.query(text=user_input)
                last_result = result

                print(f"\nAssistant: {result.answer}")

                # Always show compact debug
                s = result.debug_summary
                print(
                    f"\n  [{s['mode']}] {s['complexity']} | "
                    f"{s['chunks_retrieved']} chunks | "
                    f"critic: {s['critic_verdict']} | "
                    f"{s['total_ms']:.0f}ms"
                )

                if show_verbose:
                    _print_verbose(result)

                print()

        finally:
            await service.shutdown()
            print("\nGoodbye.")

    asyncio.run(_run())


def _print_verbose(result) -> None:
    """Print detailed debug information."""
    trace = result.trace

    # Chunks
    chunks = result.chunks_for_display
    if chunks:
        print(f"\n  Retrieved chunks ({len(chunks)}):")
        for c in chunks[:5]:  # top 5
            print(
                f"    [{c['rank']}] {c['chunk_id'][:30]:30s} "
                f"sect={c['section']:12s} "
                f"score={c['score']:.4f} "
                f"| {c['preview'][:50]}..."
            )
        if len(chunks) > 5:
            print(f"    ... and {len(chunks) - 5} more")

    # Sub-queries (if any)
    if trace.sub_queries:
        print(f"\n  Sub-queries ({len(trace.sub_queries)}):")
        for sq in trace.sub_queries:
            print(f"    [{sq['mode']}] {sq['text'][:60]}...")

    # Critic feedback
    if trace.critic_feedback:
        fb = trace.critic_feedback
        print(f"\n  Critic: {fb.get('verdict', '?')}")
        if fb.get("faithfulness_score"):
            print(
                f"    faithfulness={fb['faithfulness_score']:.0f}"
                f"  completeness={fb.get('completeness_score', '?')}"
            )
        if fb.get("issues"):
            print(f"    issues: {', '.join(fb['issues'])}")
        if fb.get("reasoning"):
            print(f"    reasoning: {fb['reasoning'][:100]}")

    # Event log
    if trace.events:
        print("\n  Event log:")
        for event in trace.events:
            print(f"    {event['node']:15s} {event['status']:25s} {event['latency_ms']:>8.0f}ms")
