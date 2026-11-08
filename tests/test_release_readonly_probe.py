"""Pure, network-free safety checks for the read-only release evidence probe."""
import pytest

from scripts import quantterm_release_readonly_probe as PROBE
from scripts.quantterm_release_readonly_probe import (
    _origin, _quantile_ms, collect,
)


@pytest.mark.parametrize("bad", [
    "https://127.0.0.1:8765",
    "http://example.com:8765",
    "http://127.0.0.1:8765/api",
    "http://127.0.0.1:8765/?secret=value",
    "http://user:pass@localhost:8765",
    "http://127.0.0.1",
    "file:///tmp/health",
])
def test_probe_never_targets_non_loopback_or_credentialed_origin(bad):
    with pytest.raises(ValueError):
        _origin(bad)


def test_probe_percentile_is_explicit_sample_interpolation():
    assert _quantile_ms([], 0.50) is None
    assert _quantile_ms([1.0], 0.99) == 1.0
    assert _quantile_ms([1.0, 2.0, 3.0, 4.0], 0.95) == 3.85


def test_probe_makes_no_dashboard_requests_without_verified_live_lock():
    requested = []

    def fetch(url, timeout):
        requested.append(url)
        return 1.0, {"live_locked": True, "live_execution_authorized": False}

    rc, report = collect(
        "http://127.0.0.1:8765", samples=3,
        timeout=2, pause=0, fetch=fetch, sleeper=lambda _: None,
    )
    assert rc == 2
    assert report["live_execution_gate"] == "BLOCKED_UNVERIFIED"
    assert report["reason"] == "live-money-lock-not-verified"
    assert requested == ["http://127.0.0.1:8765/api/health"]


def test_probe_serial_get_metrics_never_claim_release_certification():
    requested = []

    def fetch(url, timeout):
        requested.append(url)
        if url.endswith("/api/health"):
            return 2.0, {
                "live_lock_verified": True,
                "live_locked": True,
                "live_execution_authorized": False,
            }
        return 4.0, {}

    rc, report = collect(
        "http://localhost:8765", samples=3,
        timeout=2, pause=0, fetch=fetch, sleeper=lambda _: None,
    )
    assert rc == 0
    assert report["live_execution_gate"] == "LOCKED_VERIFIED"
    assert report["release_certified"] is False
    assert report["endpoints"]["/api/health"]["successes"] == 3
    assert report["endpoints"]["/api/dashboard"]["p99_ms"] == 4.0
    assert len(requested) == 7  # independent safety preflight + three of each GET
    assert all(url.startswith("http://localhost:8765/") for url in requested)


def test_probe_preserves_a_failed_dashboard_measurement():
    def fetch(url, timeout):
        if url.endswith("/api/health"):
            return 2.0, {
                "live_lock_verified": True,
                "live_locked": True,
                "live_execution_authorized": False,
            }
        raise TimeoutError("local API timeout")

    rc, report = collect(
        "http://127.0.0.1:8765", samples=2,
        timeout=2, pause=0, fetch=fetch, sleeper=lambda _: None,
    )
    assert rc == 1
    assert report["result"] == "MEASURED_WITH_FAILURES"
    assert report["endpoints"]["/api/dashboard"]["successes"] == 0
    assert report["endpoints"]["/api/dashboard"]["failures"] == 2
    assert report["endpoints"]["/api/dashboard"]["p95_ms"] is None



def test_probe_http_transport_is_get_only_and_disables_redirects_proxies(monkeypatch):
    seen = []

    class Response:
        status = 200
        def read(self, n):
            assert n == PROBE.MAX_RESPONSE_BYTES + 1
            return b"{}"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None

    class Opener:
        def open(self, request, timeout):
            assert request.get_method() == "GET"
            assert request.full_url == "http://127.0.0.1:8765/api/health"
            seen.append((request.full_url, timeout))
            return Response()

    def build(*handlers):
        assert isinstance(handlers[0], PROBE.ProxyHandler)
        assert handlers[0].proxies == {}
        assert isinstance(handlers[1], PROBE._NoRedirect)
        assert handlers[1].redirect_request(None, None, 302, "redirect", {}, "https://remote") is None
        return Opener()

    monkeypatch.setattr(PROBE, "build_opener", build)
    elapsed, result = PROBE._fetch_json(
        "http://127.0.0.1:8765/api/health", timeout=2.0,
    )
    assert elapsed >= 0
    assert result == {}
    assert seen == [("http://127.0.0.1:8765/api/health", 2.0)]
