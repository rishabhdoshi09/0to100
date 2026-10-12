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


def test_public_caught_http_500_never_reflects_exception_in_route_detail(monkeypatch):
    import pytest
    from fastapi import HTTPException
    import terminal_product_api as product_api

    secret = "/Users/operator/QuantTerm/runtime/broker.db QT_OPERATOR_TOKEN=do-not-disclose"

    def fail_workspace(symbol):
        raise RuntimeError(secret)

    monkeypatch.setattr(product_api, "build_stock_workspace", fail_workspace)
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    with pytest.raises(HTTPException) as error:
        product_api.stock_intelligence("INFY")
    assert error.value.status_code == 500
    assert secret not in str(error.value.detail)
    assert "Stock intelligence failed" in str(error.value.detail)

    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    with pytest.raises(HTTPException) as local_error:
        product_api.stock_intelligence("INFY")
    assert secret in str(local_error.value.detail)


def test_public_caught_chart_503_error_never_reflects_provider_details(monkeypatch):
    import pytest
    from fastapi import HTTPException
    import data.bhavcopy_runtime as runtime

    secret = "/Volumes/QuantTermStorage/operator-api-secret"

    def broken_history(_symbol):
        raise RuntimeError(secret)

    monkeypatch.setattr(runtime, "get_ohlcv", broken_history)
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    with pytest.raises(HTTPException) as error:
        terminal_api.chart("INFY")
    assert error.value.status_code == 503
    assert secret not in str(error.value.detail)

    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    with pytest.raises(HTTPException) as private_error:
        terminal_api.chart("INFY")
    assert secret in str(private_error.value.detail)


def test_public_projection_fallback_never_discloses_operator_storage(tmp_path, monkeypatch):
    import terminal_product_api as product_api
    import research.intelligence.event_store as event_store

    secret = "/Users/operator/private/journal.sqlite?token=SECRET"
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")

    def broken_store(_path):
        raise RuntimeError(secret)

    monkeypatch.setattr(product_api, "TARGET_EVENT_STORE", path)
    monkeypatch.setattr(event_store, "EventStore", broken_store)

    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    public = product_api.target_portfolio()
    assert public["available"] is False
    assert secret not in json.dumps(public)
    assert public["error"] == "Internal diagnostic unavailable in public read-only mode."

    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    private = product_api.target_portfolio()
    assert private["error"] == secret


def test_public_exception_helpers_hide_secrets_without_weakening_private_diagnostics(monkeypatch):
    from product.public_access import public_exception_text, public_http_error_detail

    secret = "/private/BrokerAPI key=never-show"
    error = OSError(secret)
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    assert secret not in public_exception_text(error)
    assert secret not in public_http_error_detail("Market data failed", error)

    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    assert public_exception_text(error) == secret
    assert secret in public_http_error_detail("Market data failed", error)
