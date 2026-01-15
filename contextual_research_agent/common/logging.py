import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_FORMAT = "%(asctime)s | [%(levelname)s] | %(name)s | %(message)s"
JSON_FORMAT = (
    '{"time": "%(asctime)s","level": "%(levelname)s","logger": "%(name)s","message": "%(message)s"}'
)


def setup_logging(
    level: LogLevel = "INFO",
    json_output: bool = False,
) -> None:
    log_format = JSON_FORMAT if json_output else DEFAULT_FORMAT

    logging.basicConfig(
        level=level,
        format=log_format,
        stream=sys.stdout,
        force=True,
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
