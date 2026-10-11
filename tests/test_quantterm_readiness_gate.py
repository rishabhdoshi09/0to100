"""A public/live readiness badge must fail closed on incomplete evidence."""
from __future__ import annotations

from scripts.quantterm_readiness_gate import evaluate


def valid_responses():
    return {
        "health": {
            "ok": True, "live_locked": True, "live_lock_verified": True,
            "live_execution_authorized": False,
        },
        "access": {
            "public_read_only": True,
            "operator_token_configured": True,
            "mutation_policy": "OPERATOR_TOKEN_REQUIRED",
        },
        "dashboard": {
            "generated_at": "2026-10-11T00:30:00+00:00",
            "dashboard_cache": {"status": "FRESH"},
            "data": {"history_current": True},
            "scan": {
                "available": True,
                "scanned_at": "2026-10-11T00:20:00+00:00",
                "requested_universe": 2000,
                "universe_size": 1980,
                "coverage": {"checked": 1980, "analysis_errors": 0},
                "desk_overlays": {
                    "recommendations": "saved",
                    "decision_discovery": "saved",
                },
                "provenance": {
                    "data_current": True, "price_data_as_of": "2026-10-09",
                },
            },
            "paper": {"available": True, "last_cycle": {"status": "OBSERVING"}},
            "long_term": {"available": False},
        },
        "decision-simulation-gate": {
            "status_cache": {"status": "FRESH"},
            "scan_fresh": True, "approved": False,
        },
    }


def timings():
    return {key: 0.1 for key in valid_responses()}


def test_local_readiness_requires_current_sources_but_not_paper_approval():
    result = evaluate(valid_responses(), timings_s=timings(), public=True)
    assert result["verdict"] == "LOCAL_READ_GATE_PASS"
    assert result["blockers"] == []



def test_release_readiness_requires_exact_running_build_sha():
    expected = "234e3e30e228e515a38bee994436c0af3dad1b85"
    responses = valid_responses()
    # Healthy, fully populated API evidence without build identity cannot
    # approve deployment of a newer source tree.
    assert evaluate(
        responses, timings_s=timings(), expected_sha=expected,
    )["verdict"] == "HOLD"

    responses["health"]["service"] = "quantterm-terminal-api"
    responses["health"]["build_sha"] = expected[:12]
    assert evaluate(
        responses, timings_s=timings(), expected_sha=expected,
    )["verdict"] == "HOLD"

    responses["health"]["build_sha"] = "f" * 40
    report = evaluate(responses, timings_s=timings(), expected_sha=expected)
    assert report["verdict"] == "HOLD"
    assert any("build SHA" in reason for reason in report["blockers"])

    responses["health"]["build_sha"] = expected
    assert evaluate(
        responses, timings_s=timings(), expected_sha=expected,
    )["verdict"] == "LOCAL_READ_GATE_PASS"

    responses["health"]["service"] = "another-local-process"
    assert evaluate(
        responses, timings_s=timings(), expected_sha=expected,
    )["verdict"] == "HOLD"


def test_stale_dashboard_or_gate_never_produces_pass():
    responses = valid_responses()
    responses["dashboard"]["dashboard_cache"]["status"] = "STALE"
    responses["decision-simulation-gate"]["status_cache"]["status"] = "BOOTSTRAPPING"
    result = evaluate(responses, timings_s=timings())
    assert result["verdict"] == "HOLD"
    assert any("FRESH" in entry for entry in result["blockers"])


def test_unlocked_live_money_is_always_hold():
    responses = valid_responses()
    responses["health"]["live_locked"] = False
    responses["health"]["live_execution_authorized"] = True
    result = evaluate(responses, timings_s=timings())
    assert result["verdict"] == "HOLD"
    assert any("Live" in x for x in result["blockers"])


def test_unverified_coverage_or_missing_provenance_is_hold():
    responses = valid_responses()
    responses["dashboard"]["scan"]["requested_universe"] = 2384
    responses["dashboard"]["scan"]["coverage"]["checked"] = 2179
    responses["dashboard"]["scan"]["provenance"]["data_current"] = False
    result = evaluate(responses, timings_s=timings())
    assert result["verdict"] == "HOLD"
    assert result["metrics"]["scan_coverage_pct"] < 95
    assert any("coverage" in x.lower() for x in result["blockers"])


def test_public_mode_accepts_fully_read_only_or_token_protected_access():
    responses = valid_responses()
    assert evaluate(responses, timings_s=timings(), public=True)["verdict"] == "LOCAL_READ_GATE_PASS"

    responses["access"]["operator_token_configured"] = False
    responses["access"]["mutation_policy"] = "READ_ONLY"
    assert evaluate(responses, timings_s=timings(), public=True)["verdict"] == "LOCAL_READ_GATE_PASS"

    responses["access"]["mutation_policy"] = "OPERATOR_TOKEN_REQUIRED"
    assert evaluate(responses, timings_s=timings(), public=True)["verdict"] == "HOLD"
    responses["access"]["operator_token_configured"] = True
    responses["access"]["mutation_policy"] = "READ_ONLY"
    assert evaluate(responses, timings_s=timings(), public=True)["verdict"] == "HOLD"
    responses["access"]["mutation_policy"] = "PRIVATE_OPERATOR"
    assert evaluate(responses, timings_s=timings(), public=True)["verdict"] == "HOLD"


def test_slow_or_missing_api_endpoint_fails():
    assert evaluate(valid_responses(), timings_s={
        **timings(), "dashboard": 50.0,
    })["verdict"] == "HOLD"
    partial = valid_responses()
    del partial["dashboard"]
    assert evaluate(partial, timings_s=timings())["verdict"] == "HOLD"


def test_completed_scan_without_decision_publication_is_hold():
    responses = valid_responses()
    responses["dashboard"]["scan"]["desk_overlays"]["decision_discovery"] = "error"
    report = evaluate(responses, timings_s=timings())
    assert report["verdict"] == "HOLD"
    assert any("Decision discovery" in x for x in report["blockers"])


def test_readiness_probe_rejects_deceptive_localhost_or_external_origins():
    import pytest
    from scripts.quantterm_readiness_gate import _safe_origin
    for url in (
        "http://127.0.0.1:8765@attacker.example",
        "https://127.0.0.1:8765",
        "http://attacker.example:8765",
        "http://127.0.0.1:8765/something",
        "http://localhost:8765/?token=secret",
        "http://admin:secret@localhost:8765",
        "http://127.0.0.1",
    ):
        with pytest.raises(ValueError):
            _safe_origin(url)
    assert _safe_origin("http://127.0.0.1:8765") == "http://127.0.0.1:8765"


def test_readiness_probe_bypasses_proxy_and_refuses_offhost_redirect_without_socket(monkeypatch):
    from urllib.request import ProxyHandler, Request
    from urllib.error import HTTPError
    import scripts.quantterm_readiness_gate as gate
    import pytest

    handler = next(
        x for x in gate._LOCAL_OPENER.handlers
        if isinstance(x, gate._RejectRedirects)
    )
    assert handler.redirect_request(
        Request("http://127.0.0.1:8765/api/health"),
        None, 302, "Found", {},
        "https://untrusted.example/status",
    ) is None
    assert all(
        not proxy.proxies
        for proxy in gate._LOCAL_OPENER.handlers if isinstance(proxy, ProxyHandler)
    )

    def reject_redirect(request, timeout):
        assert request.full_url == "http://127.0.0.1:8765/api/health"
        raise HTTPError(request.full_url, 302, "Found",
                        {"Location": "https://untrusted.example/"}, None)
    monkeypatch.setattr(gate._LOCAL_OPENER, "open", reject_redirect)
    with pytest.raises(HTTPError):
        gate.fetch_json("http://127.0.0.1:8765", "health", 1.0)


def test_readiness_probe_get_uses_only_loopback_and_handles_json(monkeypatch):
    import json
    import scripts.quantterm_readiness_gate as gate

    class Reply:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, count):
            assert count > 0
            return json.dumps({"ok": True, "live_locked": True}).encode()

    def intercept(request, timeout):
        assert request.get_method() == "GET"
        assert request.full_url == "http://127.0.0.1:8765/api/health"
        assert timeout == 2
        return Reply()

    monkeypatch.setattr(gate._LOCAL_OPENER, "open", intercept)
    payload, elapsed = gate.fetch_json("http://127.0.0.1:8765", "health", 2)
    assert payload["ok"] is True
    assert elapsed >= 0
