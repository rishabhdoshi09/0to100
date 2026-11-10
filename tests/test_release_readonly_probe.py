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


EXACT_RC_SHA = "db42e6ccb17100b75b4844b892089593e29639e6"


def _healthy_host(sha=EXACT_RC_SHA):
    return {
        "healthy": True,
        "heartbeat_fresh": True,
        "all_children_alive": True,
        "all_children_healthy": True,
        "deployment": {"build_sha": sha},
        "supervisor": {"production_sha": sha, "state": "RUNNING"},
        "service": {"manager": "launchd", "returncode": 0},
        "live_execution": {
            "safe": True, "verified": True, "locked": True, "authorized": False,
        },
    }


def test_exact_sha_requires_checkout_manifest_and_running_supervisor():
    ok, report = PROBE._verify_release_identity(
        EXACT_RC_SHA, EXACT_RC_SHA, _healthy_host(),
    )
    assert ok and report["result"] == "VERIFIED"
    assert all(report["checks"].values())
    assert report["release_certified"] is False


@pytest.mark.parametrize("change", [
    ("deployment", "build_sha", "wrong"),
    ("supervisor", "production_sha", "wrong"),
    ("supervisor", "state", "STOPPED"),
    ("service", "returncode", 1),
    ("live_execution", "authorized", True),
    ("live_execution", "safe", False),
])
def test_exact_sha_refuses_untrusted_host_claims(change):
    host = _healthy_host()
    group, field, value = change
    host[group][field] = value
    ok, report = PROBE._verify_release_identity(
        EXACT_RC_SHA, EXACT_RC_SHA, host,
    )
    assert not ok and report["result"] == "BLOCKED"
    assert report["release_certified"] is False


def test_exact_sha_refuses_missing_health_and_mismatched_checkout():
    for checkout, host in [
        (EXACT_RC_SHA, {}),
        ("a" * 40, _healthy_host()),
        (EXACT_RC_SHA, {**_healthy_host(), "heartbeat_fresh": False}),
        (EXACT_RC_SHA, {**_healthy_host(), "healthy": False}),
    ]:
        ok, report = PROBE._verify_release_identity(EXACT_RC_SHA, checkout, host)
        assert not ok and not all(report["checks"].values())


@pytest.mark.parametrize("invalid", ["db42e6cc", "G" * 40, "b" * 41])
def test_expected_sha_rejects_non_full_hashes(invalid):
    with pytest.raises(ValueError):
        PROBE._verify_release_identity(invalid, invalid, _healthy_host())


def test_identity_failure_never_reaches_http_endpoints():
    seen = []
    rc, report = collect(
        "http://127.0.0.1:8765", samples=1, timeout=1, pause=0,
        fetch=lambda url, timeout: seen.append(url),
        expected_sha=EXACT_RC_SHA,
        identity_probe=lambda expected: (False, {"result": "BLOCKED"}),
    )
    assert rc == 2
    assert seen == []
    assert report["reason"] == "release-sha-host-identity-not-verified"
    assert report["release_certified"] is False


def test_identity_drift_during_sampling_blocks_valid_latency_report():
    identity_calls = []
    seen = []

    def identity(expected):
        identity_calls.append(expected)
        return (
            len(identity_calls) == 1,
            {"result": "VERIFIED" if len(identity_calls) == 1 else "BLOCKED"},
        )

    def fetch(url, timeout):
        seen.append(url)
        if url.endswith("/api/health"):
            return 1.0, {
                "live_lock_verified": True,
                "live_locked": True,
                "live_execution_authorized": False,
            }
        return 2.0, {}

    rc, report = collect(
        "http://127.0.0.1:8765", samples=1, timeout=1, pause=0,
        fetch=fetch, expected_sha=EXACT_RC_SHA, identity_probe=identity,
    )
    assert rc == 2
    assert len(identity_calls) == 2
    assert len(seen) == 3
    assert report["result"] == "BLOCKED_SHA_IDENTITY"
    assert report["reason"] == "release-sha-host-identity-changed-during-probe"
    assert report["release_certified"] is False


def test_exact_sha_verified_measurements_remain_uncertified():
    def fetch(url, timeout):
        return 1.0, {
            "live_lock_verified": True,
            "live_locked": True,
            "live_execution_authorized": False,
        } if url.endswith("/api/health") else {}

    rc, report = collect(
        "http://localhost:8765", samples=1, timeout=1, pause=0,
        fetch=fetch, expected_sha=EXACT_RC_SHA,
        identity_probe=lambda expected: (True, {"result": "VERIFIED"}),
    )
    assert rc == 0
    assert report["host_identity_before"]["result"] == "VERIFIED"
    assert report["host_identity_after"]["result"] == "VERIFIED"
    assert report["result"] == "MEASURED_ONLY"
    assert report["release_certified"] is False
