"""A preflight that passes on a host without market access is a trap.

It would let an operator install the desk, walk away, and come back to a week
of NO_TRADE days that look like a quiet market and are actually a firewall. So
market access is REQUIRED, --skip-network can never return READY, and the live
lock is the one failure that must block rather than warn.

Probes are injected: the canonical suite is hermetic and every verdict here is
reachable without a socket.
"""
from __future__ import annotations

import pytest

from data.egress import ENVIRONMENT_EGRESS_BLOCKED, MARKET_EGRESS_BLOCKED, ProbeResult
from product.host_preflight import (
    BLOCKED,
    FAIL,
    PASS,
    READY,
    UNKNOWN,
    WARN,
    check_clock,
    check_live_lock,
    check_python_runtime,
    check_runtime_root,
    check_secrets,
    main,
    probe_market_access,
    render_text,
    run_host_preflight,
)


@pytest.fixture(autouse=True)
def _host(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "persistent"))
    for name in ("KITE_API_KEY", "KITE_API_SECRET"):
        monkeypatch.setenv(name, "present-but-never-printed")
    return tmp_path


def _all_reachable(url: str) -> ProbeResult:
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or url
    return ProbeResult(host, host, True, "", "HTTP 200")


def _market_blocked(url: str) -> ProbeResult:
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or url
    if "pypi" in host:
        return ProbeResult(host, host, True, "", "HTTP 200")
    return ProbeResult(host, host, False, ENVIRONMENT_EGRESS_BLOCKED,
                       "Tunnel connection failed: 403 Forbidden")


def _everything_blocked(url: str) -> ProbeResult:
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or url
    return ProbeResult(host, host, False, ENVIRONMENT_EGRESS_BLOCKED, "tunnel refused")


# ── the verdict ────────────────────────────────────────────────────────────
def test_a_healthy_host_is_ready():
    report = run_host_preflight(probe=_all_reachable)
    assert report["verdict"] == READY
    assert report["blockers"] == []


def test_no_market_access_is_never_ready():
    report = run_host_preflight(probe=_market_blocked)
    assert report["verdict"] == BLOCKED
    assert any(b["check"] == "market_access" for b in report["blockers"])


def test_a_selective_block_is_one_finding_not_three():
    checks, environment = probe_market_access(probe=_market_blocked)
    market = [c for c in checks if c.name == "market_access"]
    assert len(market) == 1
    assert market[0].evidence["failure_class"] == MARKET_EGRESS_BLOCKED
    assert environment["market_blocked"] is True
    assert not [c for c in checks if c.name in ("nse_official", "nse_archive", "zerodha")]


def test_a_total_outage_is_also_one_finding():
    checks, environment = probe_market_access(probe=_everything_blocked)
    assert [c.name for c in checks] == ["market_access"]
    assert environment["failure_class"] == ENVIRONMENT_EGRESS_BLOCKED


def test_individual_providers_are_listed_when_the_failure_is_not_uniform():
    def one_down(url):
        from urllib.parse import urlsplit

        host = urlsplit(url).hostname or url
        if "kite" in host:
            return ProbeResult(host, host, False, "PROVIDER_DOWN", "HTTP 503")
        return ProbeResult(host, host, True, "", "HTTP 200")

    checks, _ = probe_market_access(probe=one_down)
    names = {c.name for c in checks}
    assert "market_access" not in names
    assert "zerodha" in names
    assert next(c for c in checks if c.name == "zerodha").status == FAIL


def test_skipping_the_network_can_never_be_ready():
    report = run_host_preflight(skip_network=True)
    assert report["verdict"] == BLOCKED
    market = next(c for c in report["checks"] if c["name"] == "market_access")
    assert market["status"] == UNKNOWN
    assert "unproven" in market["detail"]


# ── individual checks ──────────────────────────────────────────────────────
def test_the_live_lock_is_confirmed():
    assert check_live_lock().status == PASS


def test_an_unreadable_interlock_blocks_rather_than_warns(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "product.live_execution_interlock", None)
    check = check_live_lock()
    assert check.status == FAIL
    assert check.required is True
    assert "must not start" in check.detail


def test_missing_credentials_block(monkeypatch):
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    check = check_secrets()
    assert check.status == FAIL
    assert "KITE_API_KEY" in check.detail


def test_secret_values_are_never_in_the_report(monkeypatch):
    monkeypatch.setenv("KITE_API_SECRET", "super-secret-value")
    report = run_host_preflight(probe=_all_reachable)
    assert "super-secret-value" not in str(report)
    assert "super-secret-value" not in render_text(report)


def test_a_runtime_root_inside_the_checkout_is_flagged(monkeypatch):
    monkeypatch.delenv("QT_RUNTIME_ROOT", raising=False)
    check = check_runtime_root()
    assert check.status == WARN
    assert "clean checkout" in check.detail
    assert check.evidence["persistent"] is False


def test_a_root_on_an_ephemeral_filesystem_is_flagged(tmp_path, monkeypatch):
    """tmp_path really is under /tmp, which is exactly the case being warned about."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "under-tmp"))
    check = check_runtime_root()
    assert check.status == WARN
    assert "may not survive a reboot" in check.detail


def test_a_configured_persistent_root_passes(tmp_path, monkeypatch):
    import product.host_preflight as HP

    root = tmp_path / "persistent-elsewhere"
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(root))
    # tmp_path is itself ephemeral; point the check at a prefix it is not under
    # so the persistent branch is the one under test.
    monkeypatch.setattr(HP, "EPHEMERAL_PREFIXES", ("/no-such-prefix/",))
    check = HP.check_runtime_root()
    assert check.status == PASS
    assert check.evidence["persistent"] is True


def test_the_clock_is_checked_against_ist():
    check = check_clock()
    assert check.status == PASS
    assert "IST" in check.detail


def test_the_python_version_is_checked():
    assert check_python_runtime().status == PASS


# ── the command ────────────────────────────────────────────────────────────
def test_main_exits_zero_only_when_ready(monkeypatch, capsys):
    import product.host_preflight as HP

    monkeypatch.setattr(HP, "run_host_preflight",
                        lambda *a, **k: {"schema_version": 1, "verdict": READY,
                                         "runtime_root": "/x", "checked_at": "t",
                                         "checks": [], "blockers": [], "warnings": [],
                                         "environment": {}})
    assert main(["--no-write"]) == 0
    assert "READY_FOR_PAPER_OPERATION" in capsys.readouterr().out


def test_main_exits_one_when_blocked(capsys):
    assert main(["--no-write", "--skip-network"]) == 1
    assert "BLOCKED" in capsys.readouterr().out


def test_the_report_persists_under_the_runtime_root(tmp_path):
    main(["--skip-network"])
    assert (tmp_path / "persistent" / "logs" / "product" / "host_preflight.json").exists()
