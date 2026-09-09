"""Supervised child death is detected and only that child is relaunched."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from product.stack_supervision import SupervisedChild, pid_alive, supervise_tick

ROOT = Path(__file__).resolve().parents[1]


def _spawn(label: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "QT_SUPERVISED_CHILD": label},
    )


def _wait_dead(pid: int, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.01)
    raise AssertionError(f"pid {pid} did not exit within {timeout_s}s")


def test_pid_alive_treats_zombie_as_dead():
    proc = _spawn("zombie")
    try:
        assert pid_alive(proc.pid)
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=2)
        assert not pid_alive(proc.pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)


def test_supervise_tick_restarts_only_the_dead_child():
    parent = os.getpid()
    procs: dict[str, subprocess.Popen] = {}

    def _launch(name: str):
        def launcher() -> int:
            procs[name] = _spawn(name)
            return int(procs[name].pid)

        return launcher

    children = {
        "alpha": SupervisedChild("alpha", launcher=_launch("alpha")),
        "beta": SupervisedChild("beta", launcher=_launch("beta")),
        "gamma": SupervisedChild("gamma", launcher=_launch("gamma")),
    }
    launched: list[int] = []
    try:
        first = supervise_tick(children, parent_pid=parent)
        assert first.restarted == ["alpha", "beta", "gamma"]
        assert first.parent_pid == parent
        assert pid_alive(parent)
        for name, pid in first.pids.items():
            assert pid_alive(pid), name
            launched.append(int(pid))

        idle = supervise_tick(children, parent_pid=parent)
        assert idle.restarted == []
        assert idle.pids == first.pids
        assert idle.parent_pid == parent

        killed_name = "beta"
        killed_pid = int(first.pids[killed_name])
        sibling_pids = {name: int(pid) for name, pid in first.pids.items() if name != killed_name}
        os.kill(killed_pid, signal.SIGKILL)
        _wait_dead(killed_pid)
        # Leave the zombie unreaped so detection cannot depend on wait().
        recovered = supervise_tick(children, parent_pid=parent)
        assert recovered.restarted == [killed_name]
        new_pid = int(recovered.pids[killed_name])
        launched.append(new_pid)
        assert new_pid != killed_pid
        assert pid_alive(new_pid)
        assert not pid_alive(killed_pid)
        assert recovered.parent_pid == parent
        assert pid_alive(parent)
        for name, pid in sibling_pids.items():
            assert recovered.pids[name] == pid
            assert pid_alive(pid), name

        second = supervise_tick(children, parent_pid=parent)
        assert second.restarted == []
        assert second.pids[killed_name] == new_pid
        live = {pid for pid in launched if pid_alive(pid)}
        assert live == {int(pid) for pid in recovered.pids.values()}
        assert len(live) == 3
        assert live.isdisjoint({killed_pid})
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


def test_launcher_watch_loop_restarts_dead_market_ops_without_exiting():
    inner = (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    loop = inner.split('while [[ "$STOP" != "1" ]]', 1)[1]
    assert "Market operations is down/stale; restarting." in loop
    assert "start_market_ops" in loop
    assert "Market API is down; restarting." in loop
    assert "Autonomy is down; restarting." in loop
    assert "RecoWealth desk is down; restarting." in loop
    assert "A child crash is restarted; it does not stop the desk." in inner
    assert "exit 1" not in inner.split("QuantTerm is running")[-1]
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    complete_loop = complete.split('while [[ "$STOP" != "1" ]]', 1)[1]
    assert "Report API is down; restarting." in complete_loop
    assert "Inner stack script ended; restarting it." in complete_loop
