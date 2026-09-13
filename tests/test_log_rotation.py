from __future__ import annotations

from logging.handlers import RotatingFileHandler

import logger as app_logger


def test_application_file_logging_is_bounded(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(app_logger.settings, "log_dir", tmp_path)

    def fake_basic_config(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(app_logger.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(app_logger.structlog, "configure", lambda **_: None)

    app_logger.configure_logging()

    handlers = list(captured.get("handlers") or [])
    rotating = [handler for handler in handlers if isinstance(handler, RotatingFileHandler)]
    assert len(rotating) == 1
    handler = rotating[0]
    assert handler.baseFilename.endswith("simplequant.log")
    assert handler.maxBytes == 25 * 1024 * 1024
    assert handler.backupCount == 5
