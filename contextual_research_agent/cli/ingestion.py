from __future__ import annotations

import asyncio

from contextual_research_agent.agent.agent import create_agent
from contextual_research_agent.common.logging import get_logger

logger = get_logger(__name__)


def ingest_file(
    file_path: str,
    config_path: str = "configs/agents/baseline.yaml",
) -> None:
    async def _run() -> None:
        agent = await create_agent(config_path=config_path)
        result = await agent.ingest(file_path)
        logger.info(
            f"Ingestion result: status={result.status} chunks={result.chunk_count} "
            f"run_id={result.run_id} error={result.error}",
        )

    asyncio.run(_run())


def ingest_files(
    file_paths: list[str],
    continue_on_error: bool = True,
    config_path: str = "configs/agents/baseline.yaml",
) -> None:
    async def _run() -> None:
        agent = await create_agent(config_path=config_path)
        results = await agent.ingest_batch(file_paths, continue_on_error=continue_on_error)

        ok = sum(1 for r in results if r.status == "indexed")
        failed = len(results) - ok
        logger.info(f"Batch ingestion done: ok={ok} failed={failed}")

    asyncio.run(_run())
