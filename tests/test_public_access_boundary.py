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


def test_malformed_explicit_public_flag_never_grants_private_operator_access(monkeypatch):
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)
    for invalid in ("tru", "enabled", "", "  ", "y", "maybe", "2"):
        monkeypatch.setenv("QT_PUBLIC_READ_ONLY", invalid)
        result = authorize_mutation("POST", {})
        assert result.allowed is False, invalid
        assert result.code == "PUBLIC_READ_ONLY", invalid
        assert access_projection()["public_read_only"] is True


def test_explicit_false_public_flag_preserves_private_local_workflow(monkeypatch):
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)
    for disabled in ("0", "false", "no", "off", " FALSE "):
        monkeypatch.setenv("QT_PUBLIC_READ_ONLY", disabled)
        result = authorize_mutation("POST", {})
        assert result.allowed is True, disabled
        assert result.code == "PRIVATE_OPERATOR_MODE", disabled


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


def test_private_mode_rejects_cross_site_browser_mutation(monkeypatch):
    """CORS only hides the response; a form/text-plain POST from any page the
    operator has open is still delivered to the loopback API and executed."""
    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)

    for headers in (
        {"origin": "https://evil.example"},
        {"origin": "http://rebind.attacker.example:8765"},
        {"origin": "null"},
        {"origin": "http://127.0.0.1:3000", "sec-fetch-site": "same-site"},
        {"sec-fetch-site": "cross-site"},
    ):
        result = authorize_mutation("POST", headers)
        assert result.allowed is False, headers
        assert result.code == "CROSS_SITE_MUTATION"


def test_desk_origin_and_non_browser_clients_keep_private_workflow(monkeypatch):
    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)

    for headers in (
        {},
        {"origin": "http://127.0.0.1:5173", "sec-fetch-site": "same-origin"},
        {"origin": "http://localhost:5173"},
        {"origin": "http://127.0.0.1:5173", "sec-fetch-site": "same-site"},
        {"sec-fetch-site": "none"},
    ):
        result = authorize_mutation("POST", headers)
        assert result.allowed is True, headers
        assert result.code == "PRIVATE_OPERATOR_MODE"
    assert authorize_mutation("GET", {"origin": "https://evil.example"}).allowed is True


def test_operator_token_authorizes_a_non_desk_origin(monkeypatch):
    monkeypatch.setenv("QT_PUBLIC_READ_ONLY", "1")
    monkeypatch.setenv("QT_OPERATOR_TOKEN", "correct-horse-battery-staple")
    ok = authorize_mutation("POST", {
        "origin": "https://desk.operator.example",
        "authorization": "Bearer correct-horse-battery-staple",
    })
    assert ok.allowed is True
    assert ok.code == "OPERATOR_AUTHORIZED"
    assert authorize_mutation(
        "POST", {"origin": "https://desk.operator.example"},
    ).allowed is False


def test_extra_desk_origins_are_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    monkeypatch.setenv("QT_DESK_ORIGINS", "http://127.0.0.1:4173, http://localhost:4173/")
    assert authorize_mutation("POST", {"origin": "http://127.0.0.1:4173"}).allowed is True
    assert authorize_mutation("POST", {"origin": "http://localhost:4173"}).allowed is True
    assert authorize_mutation("POST", {"origin": "https://evil.example"}).allowed is False


def test_cross_site_control_post_never_reaches_route(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    monkeypatch.delenv("QT_OPERATOR_TOKEN", raising=False)
    client = TestClient(terminal_api.app, base_url="http://127.0.0.1:8765")
    response = client.post(
        "/api/controls/RESUME_NEW_PAPER_ENTRIES",
        headers={"Origin": "https://evil.example", "Content-Type": "text/plain"},
        content="x",
    )
    assert response.status_code == 403
    assert response.json()["code"] == "CROSS_SITE_MUTATION"


def test_cross_site_refusal_reports_true_access_mode(monkeypatch):
    from fastapi.testclient import TestClient
    import report_api

    monkeypatch.delenv("QT_PUBLIC_READ_ONLY", raising=False)
    for app, path in ((terminal_api.app, "/api/controls/RUN_CYCLE_NOW"), (report_api.app, "/anything")):
        response = TestClient(app, base_url="http://127.0.0.1:8765").post(
            path, headers={"Origin": "https://evil.example"},
        )
        assert response.status_code == 403
        assert response.json() == {
            "detail": "Mutations are accepted only from the QuantTerm desk origin.",
            "code": "CROSS_SITE_MUTATION",
            "public_read_only": False,
        }
