from __future__ import annotations

import asyncio
from typing import Any

from qdrant_client import QdrantClient

from contextual_research_agent.agent.agent import create_agent
from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "configs/agents/baseline.yaml"


def query(
    question: str,
    mode: str = "qa",
    document: str | None = None,
    top_k: int = 10,
    config_path: str = DEFAULT_CONFIG,
    verbose: bool = False,
) -> None:
    async def _run() -> None:
        agent = await create_agent(config_path=config_path)

        document_ids = [document] if document else None

        response = await agent.query(
            query=question,
            mode=mode,
            document_ids=document_ids,
            top_k=top_k,
        )

        _print_response(response, verbose=verbose)

    asyncio.run(_run())


def summarize(
    document_id: str,
    top_k: int = 15,
    config_path: str = DEFAULT_CONFIG,
    verbose: bool = False,
) -> None:
    async def _run() -> None:
        agent = await create_agent(config_path=config_path)

        response = await agent.summarize(
            document_id=document_id,
            top_k=top_k,
        )

        _print_response(response, verbose=verbose)

    asyncio.run(_run())


def chat(
    document: str | None = None,
    mode: str = "qa",
    top_k: int = 10,
    config_path: str = DEFAULT_CONFIG,
) -> None:
    async def _run() -> None:
        agent = await create_agent(config_path=config_path)

        current_doc = document
        current_mode = mode
        current_top_k = top_k

        print("\n" + "=" * 60)
        print("Research Assistant - Interactive Mode")
        print("=" * 60)
        print(f"Mode: {current_mode} | Top-K: {current_top_k}")
        if current_doc:
            print(f"Document: {current_doc}")
        print("Type /help for commands, /quit to exit")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd_result = _handle_chat_command(
                    user_input,
                    current_mode,
                    current_doc,
                    current_top_k,
                )

                if cmd_result is None:
                    break

                current_mode, current_doc, current_top_k = cmd_result
                continue

            try:
                document_ids = [current_doc] if current_doc else None

                response = await agent.query(
                    query=user_input,
                    mode=current_mode,
                    document_ids=document_ids,
                    top_k=current_top_k,
                )

                print(f"\nAssistant ({current_mode}):")
                print("-" * 40)
                print(response.answer)

                if response.citations:
                    print(f"\nCitations: {', '.join(response.citations)}")

                print(
                    f"\n[{len(response.retrieval.chunks)} chunks, {response.total_latency_ms:.0f}ms]"
                )
                print()

            except Exception as e:
                print(f"\nError: {e}\n")

    asyncio.run(_run())


def _handle_chat_command(
    cmd: str,
    current_mode: str,
    current_doc: str | None,
    current_top_k: int,
) -> tuple[str, str | None, int] | None:
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if command in ("/quit", "/exit", "/q"):
        print("Goodbye!")
        return None
    if command == "/help":
        print("""
            Commands:
              /mode <qa|summarize>  - Switch cognitive mode
              /doc <id>             - Set document filter
              /doc clear            - Clear document filter
              /top_k <n>            - Set number of chunks to retrieve
              /status               - Show current settings
              /help                 - Show this help
              /quit, /exit, /q      - Exit chat
            """)
        return (current_mode, current_doc, current_top_k)

    if command == "/mode":
        if arg and arg.lower() in ("qa", "summarize"):
            current_mode = arg.lower()
            print(f"Mode set to: {current_mode}")
        else:
            print("Usage: /mode <qa|summarize>")
        return (current_mode, current_doc, current_top_k)

    if command == "/doc":
        if arg == "clear":
            current_doc = None
            print("Document filter cleared")
        elif arg:
            current_doc = arg
            print(f"Document filter set to: {current_doc}")
        else:
            print("Usage: /doc <document_id> or /doc clear")
        return (current_mode, current_doc, current_top_k)

    if command == "/top_k":
        if arg and arg.isdigit():
            current_top_k = int(arg)
            print(f"Top-K set to: {current_top_k}")
        else:
            print("Usage: /top_k <number>")
        return (current_mode, current_doc, current_top_k)

    if command == "/status":
        print("\nCurrent settings:")
        print(f"  Mode: {current_mode}")
        print(f"  Document: {current_doc or '(all)'}")
        print(f"  Top-K: {current_top_k}")
        print()
        return (current_mode, current_doc, current_top_k)

    print(f"Unknown command: {command}. Type /help for commands.")
    return (current_mode, current_doc, current_top_k)


def stats(
    config_path: str = DEFAULT_CONFIG,
) -> None:
    async def _run() -> None:
        agent = await create_agent(config_path=config_path)
        stats_data = await agent.get_stats()

        print("\nVector Store Statistics")
        print("=" * 40)

        for key, value in stats_data.items():
            print(f"  {key}: {value}")

        print()

    asyncio.run(_run())


def retrieve(
    question: str,
    document: str | None = None,
    top_k: int = 10,
    config_path: str = DEFAULT_CONFIG,
) -> None:
    async def _run() -> None:
        agent = await create_agent(config_path=config_path)

        document_ids = [document] if document else None

        result = await agent.retrieve(
            query=question,
            document_ids=document_ids,
            top_k=top_k,
        )

        print(f"\nRetrieved {len(result.chunks)} chunks ({result.latency_ms:.0f}ms)")
        print("=" * 60)

        for i, rc in enumerate(result.chunks, 1):
            print(f"\n[{i}] {rc.chunk.id} (score: {rc.score:.4f})")
            if rc.chunk.section:
                print(f"    Section: {rc.chunk.section}")
            print(f"    Document: {rc.chunk.document_id}")
            print("-" * 40)

            text = rc.chunk.text
            if len(text) > 500:
                text = text[:500] + "..."
            print(text)

        print()

    asyncio.run(_run())


def _print_response(response: Any, verbose: bool = False) -> None:
    print("\n" + "=" * 60)
    print(f"Mode: {response.mode.value}")
    print("=" * 60)

    print("\nAnswer:")
    print("-" * 40)
    print(response.answer)

    if response.citations:
        print(f"\nCitations: {', '.join(response.citations)}")

    print("\n" + "-" * 40)
    print(f"Chunks retrieved: {len(response.retrieval.chunks)}")
    print(
        f"Latency: {response.total_latency_ms:.0f}ms "
        f"(retrieve: {response.latency.get('retrieve_ms', 0):.0f}ms, "
        f"generate: {response.latency.get('generate_ms', 0):.0f}ms)"
    )

    if response.tokens.get("total"):
        print(
            f"Tokens: {response.tokens['total']} "
            f"(prompt: {response.tokens.get('prompt', 0)}, "
            f"completion: {response.tokens.get('completion', 0)})"
        )

    if verbose:
        print("\n" + "=" * 60)
        print("Retrieved Chunks:")
        print("=" * 60)

        for i, rc in enumerate(response.retrieval.chunks, 1):
            print(f"\n[{i}] {rc.chunk.id} (score: {rc.score:.4f})")
            if rc.chunk.section:
                print(f"    Section: {rc.chunk.section}")

            text = rc.chunk.text
            if len(text) > 300:
                text = text[:300] + "..."
            print(text)

    print()


def list_docs(
    config_path: str = DEFAULT_CONFIG,
    limit: int = 100,
) -> None:
    async def _run() -> None:
        client = QdrantClient(host="100.121.65.75", port=6333)

        points, _ = client.scroll(
            collection_name="documents_qwen3_0_6b_1024",
            limit=limit,
            with_payload=["document_id"],
        )

        doc_ids = set()
        for p in points:
            doc_id = p.payload.get("document_id")
            if doc_id:
                doc_ids.add(doc_id)

        print(f"\nFound {len(doc_ids)} documents:")
        for doc_id in sorted(doc_ids):
            print(f"  - {doc_id}")

    asyncio.run(_run())
