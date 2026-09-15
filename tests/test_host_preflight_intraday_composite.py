"""Intraday pricing is a chain, so readiness must be a composite.

data.live_quotes sources intraday as Kite FIRST, NSE public API as FALLBACK.
Requiring each leg independently blocked a real Mac whose authenticated Kite
quotes were flowing while NSE's public route answered 404. These tests pin the
composite, the read-only nature of the Kite probe, and the evidence that says
which source actually satisfied intraday.

Bhavcopy stays independently required: it is the primary history source and has
no alternate route.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import product.host_preflight as HP
from data.egress import ProbeResult
from product.host_preflight import (
    BLOCKED,
    CAPABILITY_ENDPOINT_INVALID,
    CAPABILITY_REJECTED,
    CAPABILITY_UNREACHABLE,
    CAPABILITY_USABLE,
    CapabilityProbe,
    FAIL,
    INTRADAY_CAPABILITY,
    PASS,
    READY,
    WARN,
    probe_market_access,
    run_host_preflight,
)

#: Every KiteConnect call that changes broker state. The preflight must never
#: touch one of these, and this list is what test E enforces.
MUTATIONS = (
    "place_order", "modify_order", "cancel_order", "exit_order",
    "place_gtt", "modify_gtt", "delete_gtt", "place_mf_order", "cancel_mf_order",
    "convert_position", "generate_session", "invalidate_access_token", "set_access_token",
)


@pytest.fixture(autouse=True)
def _host(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "persistent"))
    for name in ("KITE_API_KEY", "KITE_API_SECRET"):
        monkeypatch.setenv(name, "present-but-never-printed")
    return tmp_path


def _stub(name: str, state: str):
    host = {"kite_intraday": "api.kite.trade",
            "nse_live": "www.nseindia.com",
            "nse_archive": "nsearchives.nseindia.com"}[name]

    def _runner(url: str, timeout: float = 12.0) -> CapabilityProbe:
        ok = state != CAPABILITY_UNREACHABLE
        return CapabilityProbe(
            ProbeResult(host, host, ok, "" if ok else "TIMEOUT", state),
            state, {"capability": state, "stubbed": True},
        )
    return _runner


def _wire(monkeypatch, *, kite: str, nse_live: str, archive: str = CAPABILITY_USABLE):
    monkeypatch.setattr(HP, "_kite_intraday_probe", _stub("kite_intraday", kite))
    monkeypatch.setattr(HP, "_nse_live_probe", _stub("nse_live", nse_live))
    monkeypatch.setattr(HP, "_nse_archive_probe", _stub("nse_archive", archive))
    monkeypatch.setattr(HP, "_CAPABILITY_RUNNERS", {
        "kite_intraday": HP._kite_intraday_probe,
        "nse_live": HP._nse_live_probe,
        "nse_archive": HP._nse_archive_probe,
    })
    monkeypatch.setattr(HP, "_reachability_probe", _stub("nse_live", CAPABILITY_USABLE))


# ── A: the real-Mac case ───────────────────────────────────────────────────
def test_A_kite_usable_and_nse_live_404_passes(monkeypatch):
    _wire(monkeypatch, kite=CAPABILITY_USABLE, nse_live=CAPABILITY_ENDPOINT_INVALID)

    checks, _env = probe_market_access()
    by_name = {c.name: c for c in checks}

    assert by_name[INTRADAY_CAPABILITY].status == PASS
    assert by_name[INTRADAY_CAPABILITY].required is True
    assert by_name["kite_intraday"].status == PASS
    # The dead fallback is reported, and is degraded rather than fatal.
    assert by_name["nse_live"].status == WARN
    assert by_name["nse_live"].required is False
    assert "degraded only" in by_name["nse_live"].detail

    report = run_host_preflight()
    assert report["verdict"] == READY, report["blockers"]
    assert report["blockers"] == []


# ── B: no broker session, public route carrying intraday ───────────────────
def test_B_kite_rejected_and_nse_live_usable_passes(monkeypatch):
    _wire(monkeypatch, kite=CAPABILITY_REJECTED, nse_live=CAPABILITY_USABLE)

    checks, _env = probe_market_access()
    by_name = {c.name: c for c in checks}

    assert by_name[INTRADAY_CAPABILITY].status == PASS
    assert by_name["nse_live"].status == PASS
    assert by_name["kite_intraday"].status == WARN
    assert by_name["kite_intraday"].required is False

    assert run_host_preflight()["verdict"] == READY


# ── C: both intraday paths gone -- fail closed ─────────────────────────────
def test_C_both_intraday_paths_unusable_blocks(monkeypatch):
    _wire(monkeypatch, kite=CAPABILITY_UNREACHABLE, nse_live=CAPABILITY_ENDPOINT_INVALID)

    checks, _env = probe_market_access()
    by_name = {c.name: c for c in checks}

    assert by_name[INTRADAY_CAPABILITY].status == FAIL
    assert by_name[INTRADAY_CAPABILITY].evidence["satisfied_by"] == ""
    assert by_name[INTRADAY_CAPABILITY].evidence["sources"] == {
        "kite_intraday": CAPABILITY_UNREACHABLE,
        "nse_live": CAPABILITY_ENDPOINT_INVALID,
    }

    report = run_host_preflight()
    assert report["verdict"] == BLOCKED
    assert INTRADAY_CAPABILITY in {b["check"] for b in report["blockers"]}


# ── D: history has no alternate route, so it gates on its own ──────────────
@pytest.mark.parametrize("kite,nse", [
    (CAPABILITY_USABLE, CAPABILITY_USABLE),
    (CAPABILITY_USABLE, CAPABILITY_ENDPOINT_INVALID),
    (CAPABILITY_REJECTED, CAPABILITY_USABLE),
])
def test_D_archive_unusable_blocks_whatever_intraday_does(monkeypatch, kite, nse):
    _wire(monkeypatch, kite=kite, nse_live=nse, archive=CAPABILITY_REJECTED)

    checks, _env = probe_market_access()
    by_name = {c.name: c for c in checks}
    assert by_name["nse_archive"].status == FAIL
    assert by_name["nse_archive"].required is True
    assert by_name[INTRADAY_CAPABILITY].status == PASS   # intraday is fine; history is not

    report = run_host_preflight()
    assert report["verdict"] == BLOCKED
    assert "nse_archive" in {b["check"] for b in report["blockers"]}


# ── E: the probe reads. It must never reach a mutation method. ─────────────
class _RecordingKite:
    """Stands in for KiteClient, recording every attribute the probe touches."""

    def __init__(self, touched: list[str], quotes: dict | None = None,
                 raise_on_quote: BaseException | None = None):
        object.__setattr__(self, "_touched", touched)
        object.__setattr__(self, "_quotes", quotes if quotes is not None else {})
        object.__setattr__(self, "_raise", raise_on_quote)

    def __getattr__(self, name):
        object.__getattribute__(self, "_touched").append(name)
        raise AttributeError(name)

    def batch_quotes(self, symbols):
        object.__getattribute__(self, "_touched").append("batch_quotes")
        exc = object.__getattribute__(self, "_raise")
        if exc is not None:
            raise exc
        return object.__getattribute__(self, "_quotes")

    @property
    def raw(self):
        touched = object.__getattribute__(self, "_touched")
        touched.append("raw")

        class _Raw:
            def quote(self, instruments):
                touched.append("raw.quote")
                raise PermissionError("Incorrect `api_key` or `access_token`.")
        return _Raw()


def _install_kite(monkeypatch, client):
    import data.kite_client as KC
    from config import settings
    monkeypatch.setattr(settings, "kite_access_token", "a-real-token", raising=False)
    monkeypatch.setattr(KC, "KiteClient", lambda *a, **k: client)


def test_E_probe_never_invokes_a_broker_mutation(monkeypatch):
    touched: list[str] = []
    _install_kite(monkeypatch, _RecordingKite(
        touched, quotes={"RELIANCE": {"ltp": 1402.5}}))

    probe = HP._kite_intraday_probe("https://api.kite.trade")

    assert probe.state == CAPABILITY_USABLE
    assert probe.evidence["path"] == "KiteClient.batch_quotes"
    assert touched == ["batch_quotes"]
    for mutation in MUTATIONS:
        assert mutation not in touched


def test_E_refused_session_is_rejected_not_unreachable(monkeypatch):
    touched: list[str] = []
    _install_kite(monkeypatch, _RecordingKite(touched, quotes={}))

    probe = HP._kite_intraday_probe("https://api.kite.trade")

    assert probe.state == CAPABILITY_REJECTED
    assert probe.result.ok is True             # Kite answered; egress is fine
    assert touched == ["batch_quotes", "raw", "raw.quote"]
    for mutation in MUTATIONS:
        assert mutation not in touched


def test_E_transport_failure_is_unreachable(monkeypatch):
    import requests
    touched: list[str] = []
    _install_kite(monkeypatch, _RecordingKite(
        touched, raise_on_quote=requests.exceptions.ConnectTimeout("timed out")))

    probe = HP._kite_intraday_probe("https://api.kite.trade")

    assert probe.state == CAPABILITY_UNREACHABLE
    assert probe.result.ok is False
    for mutation in MUTATIONS:
        assert mutation not in touched


def test_E_no_access_token_is_rejected_and_calls_no_broker_method(monkeypatch):
    import data.kite_client as KC
    from config import settings
    monkeypatch.setattr(settings, "kite_access_token", "", raising=False)

    def _boom(*a, **k):
        raise AssertionError("no broker client may be built without a session")
    monkeypatch.setattr(KC, "KiteClient", _boom)
    monkeypatch.setattr(HP, "_reachability_probe", _stub("kite_intraday", CAPABILITY_USABLE))

    probe = HP._kite_intraday_probe("https://api.kite.trade")
    assert probe.state == CAPABILITY_REJECTED
    assert "main.py login" in probe.evidence["reason"]


# ── F: the evidence names the source that actually served intraday ─────────
def test_F_evidence_identifies_the_satisfying_source(monkeypatch):
    _wire(monkeypatch, kite=CAPABILITY_USABLE, nse_live=CAPABILITY_ENDPOINT_INVALID)
    by_name = {c.name: c for c in probe_market_access()[0]}
    composite = by_name[INTRADAY_CAPABILITY]
    assert composite.evidence["satisfied_by"] == "kite_intraday"
    assert composite.evidence["sources"] == {
        "kite_intraday": CAPABILITY_USABLE,
        "nse_live": CAPABILITY_ENDPOINT_INVALID,
    }
    assert composite.detail == "intraday pricing available via kite_intraday"
    assert by_name["kite_intraday"].evidence["intraday_role"] == "satisfies intraday"
    assert by_name["nse_live"].evidence["intraday_role"] == "alternate intraday source"


def test_F_evidence_names_nse_when_nse_is_the_one_serving(monkeypatch):
    _wire(monkeypatch, kite=CAPABILITY_REJECTED, nse_live=CAPABILITY_USABLE)
    by_name = {c.name: c for c in probe_market_access()[0]}
    composite = by_name[INTRADAY_CAPABILITY]
    assert composite.evidence["satisfied_by"] == "nse_live"
    assert by_name["nse_live"].evidence["intraday_role"] == "satisfies intraday"
    assert by_name["kite_intraday"].evidence["intraday_role"] == "alternate intraday source"


# ── unchanged guarantees ───────────────────────────────────────────────────
def test_skip_network_still_cannot_be_ready():
    report = run_host_preflight(skip_network=True)
    assert report["verdict"] == BLOCKED


def test_kite_quote_is_a_declared_broker_read_not_a_guarded_mutation():
    from data.kite_client import _BROKER_READS, _is_guarded_attribute
    assert "quote" in _BROKER_READS
    assert _is_guarded_attribute("quote", lambda *a: None) is False
    for mutation in ("place_order", "modify_order", "cancel_order", "place_gtt"):
        assert _is_guarded_attribute(mutation, lambda *a: None) is True
