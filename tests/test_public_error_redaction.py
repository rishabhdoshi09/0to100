"""Public QuantTerm API errors must not disclose runtime or secret detail."""
from __future__ import annotations

import asyncio
import json

from fastapi import Request

import terminal_api


def _request():
    return Request({
        "type": "http", "method": "GET", "scheme": "http",
        "path": "/api/health", "query_string": b"",
        "headers": [], "server": ("127.0.0.1", 8765),
    })


def test_public_unhandled_errors_hide_paths_tokens_and_sql_details(monkeypatch, capsys):
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    secret = "QT_OPERATOR_TOKEN=EXAMPLE_PRIVATE_SECRET"
    exc = RuntimeError(f"failed SQL against /Users/operator/private.sqlite: {secret}")
    result = asyncio.run(terminal_api._log_unhandled(_request(), exc))
    payload = json.loads(result.body)
    assert result.status_code == 500
    assert payload == {
        "ok": False, "error": "Internal server error",
        "code": "API_INTERNAL_ERROR",
    }
    assert result.headers["cache-control"] == "no-store"
    assert secret not in result.body.decode()
    assert secret not in capsys.readouterr().out


def test_private_local_error_context_is_preserved_for_existing_operator_workflow(monkeypatch):
    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    result = asyncio.run(terminal_api._log_unhandled(
        _request(), RuntimeError("local scanner diagnostic"),
    ))
    payload = json.loads(result.body)
    assert result.status_code == 500
    assert payload["error"] == "local scanner diagnostic"
    assert payload["path"] == "/api/health"
