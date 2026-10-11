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


def test_public_health_never_exposes_runtime_paths_or_exception_text(monkeypatch):
    import product.runtime_lifecycle as lifecycle

    secret = "/Users/operator/.env QT_OPERATOR_TOKEN=EXAMPLE_PRIVATE_SECRET"
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")

    def broken_runtime(*, api_serving):
        raise RuntimeError(secret)

    monkeypatch.setattr(lifecycle, "inspect_runtime", broken_runtime)
    result = terminal_api.health()
    assert result["lifecycle"] == "DEGRADED"
    assert result["live_lock_verified"] is False
    assert result["live_execution_authorized"] is None
    assert secret not in json.dumps(result)
    assert result["reasons"] == []
    assert result["history"] == {}
    assert result["resources"] == {}


def test_public_health_redacts_even_successful_probes_with_private_details(monkeypatch):
    import product.runtime_lifecycle as lifecycle

    secret = "/Volumes/QuantTermStorage/operator-only.db"
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")

    def successful_runtime(*, api_serving):
        return {
            "lifecycle": "READY", "reason": f"Loaded {secret}",
            "reasons": [secret], "components": [{"name": secret, "status": "READY"}],
            "history": {"path": secret}, "resources": {"path": secret},
            "operational_ready": True, "evidence_ready": True,
            "live_locked": True, "live_lock_verified": True,
            "live_execution_authorized": False,
            "live_lock_reason": secret,
        }

    monkeypatch.setattr(lifecycle, "inspect_runtime", successful_runtime)
    result = terminal_api.health()
    assert result["lifecycle"] == "READY"
    assert result["operational_ready"] is True
    assert result["live_locked"] is True
    assert result["live_execution_authorized"] is False
    assert secret not in json.dumps(result)

    # Local/private operator retains the useful diagnostic payload.
    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    private_result = terminal_api.health()
    assert private_result["reason"] == f"Loaded {secret}"
    assert secret in json.dumps(private_result)
