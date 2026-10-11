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


def test_local_only_soak_never_follows_a_redirect_to_remote_url():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "https://outside.example/health")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = sample_one(f"http://127.0.0.1:{server.server_port}", "health", 2)
        assert result["ok"] is False
        assert result["error_type"] == "HTTPError"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_local_soak_ignores_proxy_environment(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    import json

    class Local(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({
                "ok": True, "live_locked": True,
                "live_lock_verified": True, "live_execution_authorized": False,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    # A diagnostic must use a direct loopback socket, not HTTP_PROXY.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Local)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = sample_one(f"http://127.0.0.1:{server.server_port}", "health", 2)
        assert result["ok"] is True
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
