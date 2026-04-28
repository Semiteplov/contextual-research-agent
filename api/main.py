from __future__ import annotations

import logging
import sys

import uvicorn
from api.app import create_app
from api.config import get_api_settings


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


app = create_app()


def main() -> None:
    settings = get_api_settings()
    _setup_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting API server on %s:%d (auth=%s)",
        settings.host,
        settings.port,
        "enabled" if settings.auth_enabled else "disabled",
    )

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=settings.workers,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
