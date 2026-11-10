"""QT-PRA-076: the host supervisor must not keep the official-history store resident.

Measured on the owner's 8 GB Mac: the long-lived supervisor (product.host_entrypoint)
held a 337 MB footprint because bootstrap_host_state() unpickled the whole
bhavcopy store in-process (13 MB -> 319 MB; clearing it afterwards only got
back to 135 MB). The supervisor now runs the bootstrap in a child interpreter.
"""
from __future__ import annotations

import json
import sys
import textwrap

import pytest

import product.host_bootstrap as HB
from core.runtime_paths import runtime_path


def _child(code: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(code)]


def test_child_report_is_returned_and_child_logs_are_forwarded(capfd):
    report = HB.bootstrap_host_state_isolated(command=_child(f"""
        import json, sys
        print("bhav_store_loaded_from_cache symbols=3463", flush=True)
        print({HB._REPORT_PREFIX!r} + json.dumps({{"state": "READY", "history_after": {{"symbols": 3463}}}}), flush=True)
    """))

    assert report["state"] == "READY"
    assert report["history_after"] == {"symbols": 3463}
    assert report["isolated"] is True
    out = capfd.readouterr().out
    assert "bhav_store_loaded_from_cache symbols=3463" in out
    assert HB._REPORT_PREFIX not in out


def test_child_that_dies_without_a_report_is_failed_and_persisted():
    report = HB.bootstrap_host_state_isolated(command=_child("""
        import sys
        print("partial work", flush=True)
        sys.exit(3)
    """))

    assert report["state"] == "FAILED"
    assert "rc=3" in report["error"]
    persisted = json.loads(runtime_path(HB.STATUS_PATH).read_text())
    assert persisted["state"] == "FAILED"


def test_stop_request_is_delivered_as_cooperative_cancellation():
    report = HB.bootstrap_host_state_isolated(
        should_stop=lambda: True,
        poll_s=0.05,
        command=_child(f"""
            import json, signal, sys, time
            stop = []
            signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
            print("ready", flush=True)
            while not stop:
                time.sleep(0.01)
            print({HB._REPORT_PREFIX!r} + json.dumps({{"state": "CANCELLED", "error": "history bootstrap cancelled during shutdown"}}), flush=True)
        """),
    )

    assert report["state"] == "CANCELLED"


def test_child_ignoring_stop_is_killed_after_grace_and_reported_cancelled():
    report = HB.bootstrap_host_state_isolated(
        should_stop=lambda: True,
        poll_s=0.05,
        term_grace_s=0.3,
        command=_child("""
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(60)
        """),
    )

    assert report["state"] == "CANCELLED"


def test_child_entrypoint_prints_one_report_line(monkeypatch, capsys):
    monkeypatch.setattr(HB, "bootstrap_host_state", lambda **kwargs: {"state": "DEGRADED", "error": "x"})

    assert HB._child_main() == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith(HB._REPORT_PREFIX)]
    assert len(lines) == 1
    assert json.loads(lines[0][len(HB._REPORT_PREFIX):])["state"] == "DEGRADED"


def test_supervisor_bootstrap_never_runs_the_in_process_history_load(monkeypatch):
    import product.host_supervisor as HS

    monkeypatch.setattr(HB, "bootstrap_host_state", lambda **kwargs: pytest.fail("in-process bootstrap in supervisor"))
    calls = []
    monkeypatch.setattr(HB, "bootstrap_host_state_isolated", lambda **kwargs: calls.append(kwargs) or {"state": "READY"})
    supervisor = HS.HostSupervisor()

    supervisor.bootstrap_host()

    assert supervisor.bootstrap["state"] == "READY"
    assert len(calls) == 1 and callable(calls[0]["should_stop"])


def test_failed_isolated_bootstrap_still_fails_the_supervisor_closed(monkeypatch):
    import product.host_supervisor as HS

    monkeypatch.setattr(HB, "bootstrap_host_state_isolated", lambda **kwargs: {"state": "FAILED", "error": "child died"})
    supervisor = HS.HostSupervisor()

    with pytest.raises(RuntimeError, match="host bootstrap failed: child died"):
        supervisor.bootstrap_host()
