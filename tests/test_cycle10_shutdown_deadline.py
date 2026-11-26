"""Cycle 10 (C8.7-03): the autonomy loop's exit is bounded by a deadline shorter than the
host's stop grace.

The host stops the autonomy child with SIGTERM and SIGKILLs it after STOP_GRACE_S (12 s).
The loop's exit joined its runtime-heartbeat thread for ``max(2, heartbeat_s / 2 + 0.5)``
seconds: 15.5 s at the production heartbeat (30 s), longer than the whole grace. A
heartbeat write stalled on the USB disk when the stop came therefore kept the process
inside that join until the SIGKILL, so ``sup.shutdown()`` (state save, job store close,
scheduler lock release) and the keepers' checkpoints never ran. The thread is a daemon and
its writes are atomic, so the exit no longer waits on it beyond a fixed short deadline.
"""
from __future__ import annotations

import threading
import time

from research.autonomy import console_runtime as CR
from research.autonomy.supervisor import Supervisor
from tests.test_autonomy_console_runtime import _Deps


class _StopAfter(Supervisor):
    def __init__(self, root, stop_at: threading.Event):
        super().__init__(root, deps=_Deps())
        self.stop_at = stop_at

    def tick(self, now_ist=None):
        self.stop_at.wait(timeout=30.0)
        self.stop()
        return None


def test_a_stalled_heartbeat_write_cannot_hold_the_exit_past_the_deadline(tmp_path, monkeypatch):
    stop_at, stalled = threading.Event(), threading.Event()
    real_write = CR._write_runtime_status

    def write(supervisor, **kw):
        if threading.current_thread().name == "quantterm-runtime-heartbeat" and not stalled.is_set():
            stalled.set()
            time.sleep(9.0)                          # the USB disk stalls this one write
        return real_write(supervisor, **kw)

    monkeypatch.setattr(CR, "_write_runtime_status", write)
    sup = _StopAfter(tmp_path / "auto", stop_at)
    assert sup.start()
    done = threading.Event()
    runner = threading.Thread(target=lambda: (CR.run_visible_loop(sup, interval_s=0, heartbeat_s=12.0),
                                              done.set()), daemon=True)
    try:
        runner.start()
        assert stalled.wait(timeout=20.0)          # the heartbeat thread is inside its write
        t = time.monotonic()
        stop_at.set()                              # the stop arrives now
        assert done.wait(timeout=20.0)
        assert time.monotonic() - t < 3.5          # the deadline (2 s) plus the loop's own exit work
    finally:
        stop_at.set()
        runner.join(timeout=20.0)
        sup.shutdown()


def test_the_exit_deadline_is_well_inside_the_host_grace():
    from product.host_supervisor import STOP_GRACE_S

    assert CR.RUNTIME_JOIN_DEADLINE_S <= STOP_GRACE_S / 4
