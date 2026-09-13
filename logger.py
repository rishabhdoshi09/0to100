"""Structured logger used across all modules."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import structlog
from rich.logging import RichHandler

from config import settings


# Keep the primary application log bounded.  The previous unbounded FileHandler
# grew simplequant.log into multiple gigabytes on an always-on workstation.
# Five 25 MiB archives plus the active file retain useful recent history while
# preventing one logger from exhausting the runtime volume.
_LOG_MAX_BYTES = 25 * 1024 * 1024
_LOG_BACKUP_COUNT = 5


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log_file = settings.log_dir / "simplequant.log"

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True, show_path=False),
            RotatingFileHandler(
                log_file,
                maxBytes=_LOG_MAX_BYTES,
                backupCount=_LOG_BACKUP_COUNT,
                encoding="utf-8",
            ),
        ],
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


class _QuietHealthAccess(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return '"GET /health ' not in msg and '"GET /api/health ' not in msg


def quiet_uvicorn_health_access() -> None:
    """Watchdogs poll /health every few seconds — keep that off the desk console."""
    log = logging.getLogger("uvicorn.access")
    if any(isinstance(item, _QuietHealthAccess) for item in log.filters):
        return
    log.addFilter(_QuietHealthAccess())