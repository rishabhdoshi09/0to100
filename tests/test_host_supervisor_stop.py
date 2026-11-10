"""QT-PRA-080: stopping the host must not orphan a child.

On the owner's Mac (2026-11-10) `launchctl bootout` left `main.py autonomy`
running as a child of PID 1 holding the scheduler lock: stop_all() stopped the
five children one after another (12 s grace each) while launchd SIGKILLed the
supervisor after its exit timeout. stop_all() now signals every child first and
waits once against one shared deadline.
"""
from __future__ import annotations

import subprocess
import sys
import time

import product.host_supervisor as HS

_IGNORE_TERM = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
_EXIT_ON_TERM = "import time; time.sleep(120)"


def _child(name: str, code: str) -> HS.Child:
    spec = next(s for s in HS.child_specs() if s.name == name)
    child = HS.Child(spec)
    child.proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    return child


def _supervisor(monkeypatch, code_by_name):
    sup = HS.HostSupervisor()
    sup.children = {name: _child(name, code) for name, code in code_by_name.items()}
    monkeypatch.setattr(sup, "write_status", lambda **kwargs: None)
    return sup


NAMES = ("frontend", "market_api", "report_api", "market_ops", "autonomy")


def test_stop_all_waits_one_shared_grace_even_if_every_child_ignores_sigterm(monkeypatch):
    monkeypatch.setattr(HS, "STOP_GRACE_S", 1.0)
    sup = _supervisor(monkeypatch, {n: _IGNORE_TERM for n in NAMES})
    time.sleep(0.5)  # let the children install their handlers
    started = time.monotonic()
    sup.stop_all()
    elapsed = time.monotonic() - started
    assert all(c.proc.poll() is not None for c in sup.children.values())
    # Sequential stopping would take >= 5 x grace; a shared deadline ~1 x grace.
    assert elapsed < 3.0, elapsed


def test_stop_all_returns_promptly_when_children_honour_sigterm(monkeypatch):
    sup = _supervisor(monkeypatch, {n: _EXIT_ON_TERM for n in NAMES})
    started = time.monotonic()
    sup.stop_all()
    assert all(c.proc.poll() is not None for c in sup.children.values())
    assert time.monotonic() - started < 3.0


def test_single_child_terminate_keeps_its_own_grace(monkeypatch):
    child = _child("autonomy", _IGNORE_TERM)
    time.sleep(0.5)
    started = time.monotonic()
    child.terminate(grace_s=0.5)
    assert child.proc.poll() is not None
    assert 0.4 <= time.monotonic() - started < 3.0
