"""Read-only soak probe must fail closed on latency, stale reads and live unlock."""
from scripts.quantterm_read_soak import (
    ENDPOINTS, _safe_origin, access_policy_safe, assess_samples, percentile_nearest_rank, sample_one,
)
import pytest


def _cycle(*, dashboard_status="FRESH", latency=0.1, health_ok=True):
    return {
        name: {
            "ok": bool(health_ok) if name == "health" else True,
            "seconds": latency,
            "cache_status": dashboard_status if name in (
                "dashboard", "decision-simulation-gate",
            ) else "",
        } for name in ENDPOINTS
    }


def test_latencies_within_budget_all_20_cycles_can_pass():
    cycles = [_cycle() for _ in range(20)]
    result = assess_samples(cycles)
    assert result["verdict"] == "READ_SOAK_PASS"
    assert result["endpoints"]["dashboard"]["p95_seconds"] == 0.1


def test_p95_exceeding_budget_fails_even_if_last_probe_was_fast():
    cycles = [_cycle(latency=0.2) for _ in range(18)]
    cycles += [_cycle(latency=3.5) for _ in range(2)]
    result = assess_samples(cycles)
    assert result["verdict"] == "HOLD"
    assert result["endpoints"]["dashboard"]["p95_seconds"] == 3.5


def test_stale_cached_read_or_unsafe_live_lock_is_hold():
    cycles = [_cycle() for _ in range(20)]
    cycles[4] = _cycle(dashboard_status="STALE")
    cycles[8] = _cycle(health_ok=False)
    result = assess_samples(cycles)
    assert result["verdict"] == "HOLD"
    assert result["endpoints"]["dashboard"]["cache_status_counts"]["STALE"] == 1


def test_missing_sample_or_timeout_is_hold():
    cycles = [_cycle() for _ in range(19)]
    assert assess_samples(cycles)["verdict"] == "HOLD"
    cycles.append(_cycle())
    cycles[1]["health"] = {"ok": False, "seconds": 3.0, "error_type": "TimeoutError"}
    assert assess_samples(cycles)["verdict"] == "HOLD"


def test_percentile_nearest_rank_semantics():
    assert percentile_nearest_rank([], 95) is None
    assert percentile_nearest_rank([1.0] * 19 + [10.0], 95) == 1.0
    assert percentile_nearest_rank([1.0] * 18 + [10.0] * 2, 95) == 10.0


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:8765",
    "http://attacker.example:8765",
    "http://127.0.0.1:8765/remote/path",
    "http://admin:secret@127.0.0.1:8765",
    "http://127.0.0.1:8765/?foo=bar",
    "http://127.0.0.1",
])
def test_loopback_only_no_credentials_or_remote_paths(url):
    with pytest.raises(ValueError):
        _safe_origin(url)


def test_accepted_loopback_origin():
    assert _safe_origin("http://127.0.0.1:8765") == "http://127.0.0.1:8765"
    assert _safe_origin("http://localhost:8765/") == "http://localhost:8765"


def test_loopback_private_operator_is_safe_for_local_soak_but_not_public_approval():
    assert access_policy_safe({
        "mutation_policy": "PRIVATE_OPERATOR",
        "public_read_only": False,
        "operator_token_configured": False,
    })
    assert access_policy_safe({
        "mutation_policy": "READ_ONLY",
        "public_read_only": True,
        "operator_token_configured": False,
    })
    assert access_policy_safe({
        "mutation_policy": "OPERATOR_TOKEN_REQUIRED",
        "public_read_only": True,
        "operator_token_configured": True,
    })


def test_ambiguous_or_broken_public_access_configuration_fails_soak():
    for config in (
        {"mutation_policy": "PRIVATE_OPERATOR", "public_read_only": True},
        {"mutation_policy": "READ_ONLY", "public_read_only": False},
        {"mutation_policy": "OPERATOR_TOKEN_REQUIRED",
         "public_read_only": True, "operator_token_configured": False},
        {"mutation_policy": "UNKNOWN", "public_read_only": False},
    ):
        assert access_policy_safe(config) is False


def test_local_only_soak_never_follows_a_redirect_to_remote_url(monkeypatch):
    """Hermetic CI prohibits sockets: exercise the actual opener/redirect hook."""
    from urllib.error import HTTPError
    from urllib.request import Request
    import scripts.quantterm_read_soak as soak

    redirects = [
        h for h in soak._LOCAL_OPENER.handlers
        if isinstance(h, soak._RejectRedirects)
    ]
    assert len(redirects) == 1
    req = Request("http://127.0.0.1:8765/api/health", method="GET")
    assert redirects[0].redirect_request(
        req, None, 302, "Found", {"Location": "https://outside.example/health"},
        "https://outside.example/health",
    ) is None

    def redirect_error(request, timeout):
        assert request.full_url == "http://127.0.0.1:8765/api/health"
        raise HTTPError(request.full_url, 302, "Found",
                        {"Location": "https://outside.example/health"}, None)

    monkeypatch.setattr(soak._LOCAL_OPENER, "open", redirect_error)
    result = sample_one("http://127.0.0.1:8765", "health", 2)
    assert result["ok"] is False
    assert result["error_type"] == "HTTPError"


def test_local_soak_ignores_proxy_environment(monkeypatch):
    """Prove proxy handler and GET validation without opening a CI socket."""
    import json
    from urllib.request import ProxyHandler
    import scripts.quantterm_read_soak as soak

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    proxies = [
        h for h in soak._LOCAL_OPENER.handlers if isinstance(h, ProxyHandler)
    ]
    # urllib excludes an empty ProxyHandler from its active handler list
    # (there are no proxy_open methods). Zero installed handlers is the
    # normal safe result; any installed handler must have no proxies.
    assert all(handler.proxies == {} for handler in proxies)

    class LocalResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            assert size > 100
            return json.dumps({
                "ok": True, "live_locked": True,
                "live_lock_verified": True, "live_execution_authorized": False,
            }).encode()

    def local_only(request, timeout):
        assert request.get_method() == "GET"
        assert request.full_url == "http://127.0.0.1:8765/api/health"
        assert timeout == 2
        return LocalResponse()

    monkeypatch.setattr(soak._LOCAL_OPENER, "open", local_only)
    result = sample_one("http://127.0.0.1:8765", "health", 2)
    assert result["ok"] is True


def test_normal_single_flight_cache_ttl_transition_is_accepted():
    cycles = [_cycle() for _ in range(20)]
    # Decision gate TTL is 15s and the 20-cycle read-soak overlaps expiry.
    # A short, fail-closed refresh window is expected, not an outage.
    cycles[15]["decision-simulation-gate"] = {
        "ok": True, "seconds": 0.1, "cache_status": "STALE",
    }
    cycles[16]["decision-simulation-gate"] = {
        "ok": True, "seconds": 0.1, "cache_status": "STALE",
    }
    assert assess_samples(cycles)["verdict"] == "READ_SOAK_PASS"


def test_prolonged_staleness_or_unknown_cache_state_is_a_release_hold():
    cycles = [_cycle() for _ in range(20)]
    for i in (6, 7, 8):
        cycles[i]["decision-simulation-gate"] = {
            "ok": True, "seconds": 0.1, "cache_status": "STALE",
        }
    assert assess_samples(cycles)["verdict"] == "HOLD"
    cycles[8]["decision-simulation-gate"]["cache_status"] = "DEGRADED"
    assert assess_samples(cycles)["verdict"] == "HOLD"


def test_stale_http_dashboard_requires_authority_to_be_fail_closed(monkeypatch):
    import json
    import scripts.quantterm_read_soak as soak

    class Response:
        status = 200
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, _size):
            return json.dumps(self.payload).encode()

    historical = {
        "dashboard_cache": {"status": "STALE"},
        "scan": {"available": False},
        "data": {"ready": False},
        "paper": {"enabled": False},
        "autonomy": {"new_paper_entries": False},
        "fno": {"available": False},
    }
    monkeypatch.setattr(soak._LOCAL_OPENER, "open",
                        lambda *_a, **_k: Response(historical))
    assert sample_one("http://127.0.0.1:8765", "dashboard", 2)["ok"] is True

    historical["paper"]["enabled"] = True
    assert sample_one("http://127.0.0.1:8765", "dashboard", 2)["ok"] is False


def test_stale_decision_gate_must_drop_approval_and_best_trades(monkeypatch):
    import json
    import scripts.quantterm_read_soak as soak

    class Response:
        status = 200
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, _size):
            return json.dumps(self.payload).encode()

    gate = {
        "status_cache": {"status": "STALE"},
        "approved": False, "discovery_ready": False, "scan_fresh": False,
        "live_lock_verified": False, "live_execution_authorized": False,
        "best_trades": [],
    }
    monkeypatch.setattr(soak._LOCAL_OPENER, "open",
                        lambda *_a, **_k: Response(gate))
    assert sample_one("http://127.0.0.1:8765", "decision-simulation-gate", 2)["ok"] is True
    gate["approved"] = True
    assert sample_one("http://127.0.0.1:8765", "decision-simulation-gate", 2)["ok"] is False


def test_direct_read_soak_call_cannot_bypass_cli_localhost_guard(monkeypatch):
    import scripts.quantterm_read_soak as soak

    def network_forbidden(*args, **kwargs):
        raise AssertionError("A remote origin must never reach the transport")
    monkeypatch.setattr(soak._LOCAL_OPENER, "open", network_forbidden)
    result = sample_one("http://127.0.0.1:8765@attacker.example", "health", 2)
    assert result["ok"] is False
    assert result["error_type"] == "ValueError"
