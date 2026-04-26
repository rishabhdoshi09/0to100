"""Structured JSON logger used across all modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog
from rich.logging import RichHandler

from config import settings


def _configure_stdlib_logging() -> None:
    log_file = settings.log_dir / "simplequant.log"
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=level, handlers=handlers, format="%(message)s")


def configure_logging() -> None:
    _configure_stdlib_logging()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
