from __future__ import annotations

import logging
import sys

from ui.app import create_app
from ui.config import get_ui_settings


def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    settings = get_ui_settings()
    _setup_logging(settings.ui_debug)

    logger = logging.getLogger(__name__)
    logger.info(
        "Starting UI on %s:%d  (API: %s)",
        settings.ui_host,
        settings.ui_port,
        settings.api_url,
    )

    app = create_app(settings)
    app.queue(max_size=10).launch(
        server_name=settings.ui_host,
        server_port=settings.ui_port,
        share=settings.ui_share,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
