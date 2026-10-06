from __future__ import annotations

import asyncio
import json

from fastapi import Request
from fastapi.responses import JSONResponse

import terminal_api
from product.public_access import access_projection, authorize_mutation


def test_private_operator_mode_preserves_existing_mutation_workflow(monkeypatch):
    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)

    result = authorize_mutation("POST", {})
    assert result.allowed is True
    assert result.code == "PRIVATE_OPERATOR_MODE"


def test_public_mode_without_token_is_strictly_read_only(monkeypatch):
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)

    assert authorize_mutation("GET", {}).allowed is True
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        result = authorize_mutation(method, {})
        assert result.allowed is False
        assert result.code == "PUBLIC_READ_ONLY"

    projection = access_projection()
    assert projection["public_read_only"] is True
    assert projection["mutation_policy"] == "READ_ONLY"
    assert projection["operator_token_configured"] is False
    assert projection["live_money_unlocked"] is False


def test_public_mode_accepts_only_matching_bearer_token(monkeypatch):
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "true")
    monkeypatch.setenv("QT_OPERATOR_TOKEN", "correct-horse-battery-staple")

    missing = authorize_mutation("POST", {})
    wrong = authorize_mutation("POST", {"authorization": "Bearer wrong"})
    correct = authorize_mutation(
        "POST",
        {"authorization": "Bearer correct-horse-battery-staple"},
    )

    assert missing.allowed is False
    assert missing.code == "OPERATOR_AUTH_REQUIRED"
    assert wrong.allowed is False
    assert wrong.code == "OPERATOR_AUTH_REQUIRED"
    assert correct.allowed is True
    assert correct.code == "OPERATOR_AUTHORIZED"

    projection = access_projection()
    assert projection["mutation_policy"] == "OPERATOR_TOKEN_REQUIRED"
    assert projection["operator_token_configured"] is True
    assert "correct-horse-battery-staple" not in json.dumps(projection)


def test_public_boundary_blocks_before_any_route_mutation(monkeypatch):
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)
    downstream_called = False

    async def downstream(_request):
        nonlocal downstream_called
        downstream_called = True
        return JSONResponse({"ok": True})

    request = Request({
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/api/controls/RUN_CYCLE_NOW",
        "raw_path": b"/api/controls/RUN_CYCLE_NOW",
        "query_string": b"",
        "headers": [],
        "client": ("203.0.113.20", 50123),
        "server": ("quantterm.example", 443),
    })

    response = asyncio.run(terminal_api._public_mutation_boundary(request, downstream))
    assert response.status_code == 403
    assert downstream_called is False
    assert response.headers["cache-control"] == "no-store"
    body = json.loads(response.body)
    assert body["code"] == "PUBLIC_READ_ONLY"


def test_public_boundary_does_not_trust_local_proxy_identity(monkeypatch):
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)

    async def downstream(_request):
        raise AssertionError("public proxy request must not reach mutation route")

    request = Request({
        "type": "http",
        "http_version": "1.1",
        "method": "DELETE",
        "scheme": "http",
        "path": "/api/watchlist/1",
        "raw_path": b"/api/watchlist/1",
        "query_string": b"",
        "headers": [(b"x-forwarded-for", b"198.51.100.8")],
        "client": ("127.0.0.1", 43000),
        "server": ("127.0.0.1", 8765),
    })

    response = asyncio.run(terminal_api._public_mutation_boundary(request, downstream))
    assert response.status_code == 403



def test_public_forward_soak_get_never_creates_verification(monkeypatch):
    import api.runtime as runtime
    import product.forward_soak as forward_soak

    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    monkeypatch.setattr(forward_soak, "scoreboard", lambda: {"available": True})
    monkeypatch.setattr(forward_soak, "load_latest_verification", lambda: None)

    def must_not_persist(*_args, **_kwargs):
        raise AssertionError("public GET must not persist forward-soak verification")

    monkeypatch.setattr(forward_soak, "persist_soak_verification", must_not_persist)
    payload = runtime.forward_soak_api()

    assert payload["verification"] == {}
    assert "never create" in payload["verification_note"].lower()


def test_private_forward_soak_get_preserves_existing_lazy_verification(monkeypatch):
    import api.runtime as runtime
    import product.forward_soak as forward_soak

    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    monkeypatch.setattr(forward_soak, "scoreboard", lambda: {"available": True})
    monkeypatch.setattr(forward_soak, "load_latest_verification", lambda: None)
    monkeypatch.setattr(
        forward_soak,
        "persist_soak_verification",
        lambda: {"status": "VERIFIED", "source": "test"},
    )

    payload = runtime.forward_soak_api()
    assert payload["verification"]["status"] == "VERIFIED"


def test_public_market_reports_get_disables_persistence(monkeypatch):
    import api.app as public_api
    import product.recommendations_workspace as workspace

    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    seen = {}

    def fake_builder(**kwargs):
        seen.update(kwargs)
        return {"needs_refresh": False, "load_note": "private default"}

    monkeypatch.setattr(workspace, "build_market_reports_workspace", fake_builder)
    monkeypatch.setattr(public_api._core.core, "_news_payload", lambda: {})
    monkeypatch.setattr(public_api._core.core, "_scan_payload", lambda: {})

    payload = public_api.market_reports_workspace()

    assert seen["persist_today"] is False
    assert "public read-only" in payload["load_note"].lower()


def test_private_market_reports_get_keeps_persistence(monkeypatch):
    import api.app as public_api
    import product.recommendations_workspace as workspace

    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    seen = {}

    def fake_builder(**kwargs):
        seen.update(kwargs)
        return {"needs_refresh": False, "load_note": "private default"}

    monkeypatch.setattr(workspace, "build_market_reports_workspace", fake_builder)
    monkeypatch.setattr(public_api._core.core, "_news_payload", lambda: {})
    monkeypatch.setattr(public_api._core.core, "_scan_payload", lambda: {})

    payload = public_api.market_reports_workspace()

    assert seen["persist_today"] is True
    assert payload["load_note"] == "private default"
