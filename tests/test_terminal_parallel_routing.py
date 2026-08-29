from __future__ import annotations

import pytest


def test_long_term_legacy_control_keeps_one_market_scan_contract():
    import terminal_api as core
    import terminal_product_api_parallel  # noqa: F401 - applies worker/health patches only

    assert core._OPERATION_CONTROLS["RUN_SCAN_NOW"] == "MARKET_SCAN"
    # One whole-market scan fills all technical setup families. Funds refresh is
    # a separate evidence refresh, not a second hidden stock scanner.
    assert core._OPERATION_CONTROLS["RUN_LONG_TERM_SCAN_NOW"] == "MARKET_SCAN"
    assert core._OPERATION_CONTROLS["REFRESH_LONG_TERM_NOW"] == "LONG_TERM_REFRESH"


def test_parallel_api_recovers_market_ops_through_bounded_base_path(monkeypatch):
    import terminal_api as core
    import terminal_product_api_parallel as parallel

    states = [
        {"running": False, "worker_pid": 15044},
        {"running": True, "worker_pid": 15168},
    ]

    def runtime():
        return states.pop(0) if states else {"running": True, "worker_pid": 15168}

    monkeypatch.setattr(core, "_ops_runtime_payload", runtime)
    monkeypatch.setattr(parallel, "pid_is_alive", lambda pid: int(pid or 0) == 15168)
    monkeypatch.setattr(parallel.time, "sleep", lambda _seconds: None)
    # Never start a real subprocess in the network-free unit suite. This test is
    # about the strict wrapper's hand-off contract, not process creation itself.
    monkeypatch.setattr(
        parallel,
        "_base_ensure_ops_worker",
        lambda wait=True: {"running": True, "worker_pid": 15168},
    )

    observed = parallel._ensure_ops_worker_strict(wait=True)
    assert observed["worker_pid"] == 15168
    assert observed["running"] is True


def test_user_control_fails_loudly_when_worker_recovery_still_unhealthy(monkeypatch):
    import terminal_api as core
    import terminal_product_api_parallel as parallel

    monkeypatch.setattr(core, "_ops_runtime_payload", lambda: {"running": False, "worker_pid": 15044})
    monkeypatch.setattr(parallel, "pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(
        parallel,
        "_base_ensure_ops_worker",
        lambda wait=True: {"running": False, "worker_pid": None},
    )
    monkeypatch.setattr(parallel.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="launcher watchdog owns recovery"):
        parallel._ensure_ops_worker_strict(wait=False)
