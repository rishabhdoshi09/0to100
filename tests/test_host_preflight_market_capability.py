"""The real-Mac blocker: a working host declared unreachable by a root URL.

NSE answers a bare ``https://www.nseindia.com`` GET with 403 and a bare
``https://nsearchives.nseindia.com`` GET with 404 even when both production
data routes serve perfectly. The old preflight probed exactly those two roots
and treated any status >= 400 as unreachable, so a Mac with working internet,
working Zerodha and working NSE data access was blocked at install.

These tests pin the fix: probes exercise the routes QuantTerm actually reads,
requested the way production requests them, and report transport separately
from capability. Fail-closed is unchanged -- a genuinely unusable data route
still blocks.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import product.host_preflight as HP
from product.host_preflight import (
    BLOCKED,
    CAPABILITY_ENDPOINT_INVALID,
    CAPABILITY_REJECTED,
    CAPABILITY_UNREACHABLE,
    CAPABILITY_USABLE,
    FAIL,
    PASS,
    READY,
    WARN,
    probe_market_access,
    run_host_preflight,
)

NSE_ROOT = "https://www.nseindia.com"
ARCHIVE_ROOT = "https://nsearchives.nseindia.com"
BHAVCOPY = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_"
LIVE_API = "https://www.nseindia.com/api/equity-stockIndices"

BHAV_CSV = b"SYMBOL,SERIES,OPEN_PRICE\n" + b"RELIANCE, EQ, 1400.00\n" * 200


@pytest.fixture(autouse=True)
def _host(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "persistent"))
    for name in ("KITE_API_KEY", "KITE_API_SECRET"):
        monkeypatch.setenv(name, "present-but-never-printed")
    return tmp_path


def _response(status: int, *, content: bytes = b"", json_body=None):
    def _json():
        if json_body is None:
            raise ValueError("no json")
        return json_body
    return SimpleNamespace(status_code=status, content=content, json=_json)


def _install_fake_http(monkeypatch, router):
    """Route every probe request through ``router(url) -> response | Exception``."""
    import requests

    def _get(url, **_kw):
        outcome = router(url)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    class _Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, **kw):
            return _get(url, **kw)

        def close(self):
            pass

    monkeypatch.setattr(requests, "get", _get)
    monkeypatch.setattr(requests, "Session", _Session)


def _kite_probe(state: str):
    """Stub the Kite leg at an exact capability, for composite policy tests."""
    from data.egress import ProbeResult
    from product.host_preflight import CapabilityProbe

    def _runner(url: str, timeout: float = 12.0) -> CapabilityProbe:
        host = "api.kite.trade"
        ok = state != CAPABILITY_UNREACHABLE
        return CapabilityProbe(
            ProbeResult(host, host, ok, "" if ok else "TIMEOUT", state),
            state, {"capability": state, "stubbed": True},
        )
    return _runner


def _live_rows(n: int = 750):
    return {"data": [{"symbol": f"SYM{i}", "lastPrice": 100.0} for i in range(n)]}


# ── the exact reproduced real-Mac case ─────────────────────────────────────
def _real_mac_router(url: str):
    """Internet up, Zerodha up, NSE roots hostile, NSE data routes healthy."""
    if url.startswith(LIVE_API):
        return _response(200, json_body=_live_rows())
    if url.startswith(BHAVCOPY):
        return _response(200, content=BHAV_CSV)
    if url.startswith(NSE_ROOT):                 # cookie priming -- 403 on this Mac
        return _response(403)
    if url.startswith(ARCHIVE_ROOT):             # bare archive root -- permanently 404
        return _response(404)
    if "kite.trade" in url or "pypi.org" in url:
        return _response(200)
    raise AssertionError(f"unexpected probe url: {url}")


def test_real_mac_case_hostile_roots_but_working_data_routes_passes(monkeypatch):
    _install_fake_http(monkeypatch, _real_mac_router)
    monkeypatch.setattr(HP, "_kite_intraday_probe", _kite_probe(CAPABILITY_USABLE))

    checks, environment = probe_market_access()
    by_name = {c.name: c for c in checks}

    assert "market_access" not in by_name, "a working host must not be called egress-blocked"
    assert environment["market_blocked"] is False
    for name in ("nse_live", "nse_archive", "control"):
        assert by_name[name].status == PASS, (name, by_name[name].detail)
        assert by_name[name].evidence["capability"] == CAPABILITY_USABLE

    # The live probe reached the API despite the root refusing it.
    assert by_name["nse_live"].evidence["cookie_priming_status"] == 403
    assert by_name["nse_live"].evidence["rows"] == 750
    assert by_name["nse_archive"].evidence["bytes"] == len(BHAV_CSV)


def test_real_mac_case_reaches_ready(monkeypatch):
    _install_fake_http(monkeypatch, _real_mac_router)
    monkeypatch.setattr(HP, "_kite_intraday_probe", _kite_probe(CAPABILITY_USABLE))
    report = run_host_preflight()
    assert report["verdict"] == READY, report["blockers"]
    assert report["blockers"] == []


# ── the opposite: roots respond, production route does not ─────────────────
def test_responding_roots_do_not_rescue_a_dead_data_route(monkeypatch):
    def router(url: str):
        if url.startswith(LIVE_API):
            return _response(403)                # the route production reads is refused
        if url.startswith(BHAVCOPY):
            return _response(403)
        if url.startswith(NSE_ROOT) or url.startswith(ARCHIVE_ROOT):
            return _response(200)                # roots look fine, and prove nothing
        return _response(200)

    _install_fake_http(monkeypatch, router)
    monkeypatch.setattr(HP, "_kite_intraday_probe", _kite_probe(CAPABILITY_REJECTED))

    checks, environment = probe_market_access()
    by_name = {c.name: c for c in checks}
    assert by_name["nse_archive"].status == FAIL
    assert by_name["nse_archive"].evidence["capability"] == CAPABILITY_REJECTED
    assert by_name["nse_live"].evidence["capability"] == CAPABILITY_REJECTED
    assert environment["market_blocked"] is False   # transport works; capability does not

    report = run_host_preflight()
    assert report["verdict"] == BLOCKED
    assert {b["check"] for b in report["blockers"]} >= {
        "nse_archive", HP.INTRADAY_CAPABILITY,
    }


# ── the four states are distinguished ──────────────────────────────────────
def test_transport_failure_is_unreachable(monkeypatch):
    import requests
    _install_fake_http(monkeypatch, lambda url: requests.exceptions.ConnectionError(
        "Failed to resolve 'nsearchives.nseindia.com'"))
    probe = HP._nse_archive_probe(ARCHIVE_ROOT)
    assert probe.state == CAPABILITY_UNREACHABLE
    assert probe.result.ok is False


def test_provider_refusal_is_rejected_not_unreachable(monkeypatch):
    _install_fake_http(monkeypatch, lambda url: _response(403))
    probe = HP._nse_archive_probe(ARCHIVE_ROOT)
    assert probe.state == CAPABILITY_REJECTED
    # It answered us, so the machine's egress is fine and must not be blamed.
    assert probe.result.ok is True


def test_every_candidate_404_is_endpoint_invalid_not_unreachable(monkeypatch):
    _install_fake_http(monkeypatch, lambda url: _response(404))
    probe = HP._nse_archive_probe(ARCHIVE_ROOT)
    assert probe.state == CAPABILITY_ENDPOINT_INVALID
    assert probe.result.ok is True
    assert len(probe.evidence["attempts"]) == HP.ARCHIVE_PROBE_LOOKBACK_DAYS


def test_a_holiday_404_does_not_stop_the_archive_walk(monkeypatch):
    seen: list[str] = []

    def router(url: str):
        seen.append(url)
        if len(seen) <= 2:                        # weekend, then a holiday
            return _response(404)
        return _response(200, content=BHAV_CSV)

    _install_fake_http(monkeypatch, router)
    probe = HP._nse_archive_probe(ARCHIVE_ROOT)
    assert probe.state == CAPABILITY_USABLE
    assert len(seen) == 3


def test_live_api_200_with_no_rows_is_not_usable(monkeypatch):
    def router(url: str):
        if url.startswith(LIVE_API):
            return _response(200, json_body={"data": []})
        return _response(403)                     # priming refused, as on the Mac
    _install_fake_http(monkeypatch, router)
    probe = HP._nse_live_probe(LIVE_API)
    assert probe.state == CAPABILITY_REJECTED
    assert probe.result.ok is True


# ── the probes must be the production routes, not invented health checks ───
def test_probes_use_the_real_acquisition_routes_and_headers():
    from data.bhavcopy_store import _HEADERS as BHAV_HEADERS, _URL as BHAV_URL
    from data.nse_live import _HEADERS as LIVE_HEADERS

    names = {name: url for name, url, _d in HP.market_endpoints()}
    assert names["nse_live"].startswith(LIVE_API)
    assert names["nse_archive"].startswith(BHAV_URL.split("{d}")[0])
    # No bare root is probed as a verdict any more.
    assert names["nse_live"] != NSE_ROOT
    assert names["nse_archive"] != ARCHIVE_ROOT
    # Headers are taken from the acquisition modules, never re-invented here.
    assert "Mozilla/5.0" in LIVE_HEADERS["User-Agent"]
    assert "Mozilla/5.0" in BHAV_HEADERS["User-Agent"]


def test_archive_probe_sends_production_headers_and_url(monkeypatch):
    from data.bhavcopy_store import _HEADERS as BHAV_HEADERS
    captured: dict = {}
    import requests

    def _get(url, **kw):
        captured["url"] = url
        captured["headers"] = kw.get("headers")
        return _response(200, content=BHAV_CSV)

    monkeypatch.setattr(requests, "get", _get)
    probe = HP._nse_archive_probe(ARCHIVE_ROOT)
    assert probe.state == CAPABILITY_USABLE
    assert captured["url"].startswith(BHAVCOPY)
    assert captured["url"].endswith(".csv")
    assert captured["headers"] == BHAV_HEADERS


def test_live_probe_sends_production_headers(monkeypatch):
    from data.nse_live import _HEADERS as LIVE_HEADERS
    captured: dict = {}
    import requests

    class _Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, **_kw):
            if url.startswith(LIVE_API):
                captured["headers"] = dict(self.headers)
                return _response(200, json_body=_live_rows(120))
            return _response(403)

        def close(self):
            pass

    monkeypatch.setattr(requests, "Session", _Session)
    probe = HP._nse_live_probe(LIVE_API)
    assert probe.state == CAPABILITY_USABLE
    assert captured["headers"] == LIVE_HEADERS
