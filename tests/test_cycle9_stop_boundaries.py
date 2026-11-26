"""Cycle 9, item 2: the autonomy SIGTERM stop at real boundaries of the REAL tick.

tests/test_cooperative_stop.py replaces Supervisor.tick and signals after READY. These run
the real tick (only the market schedule and the live feed are stubbed) with probe jobs,
and deliver SIGTERM inside the job store's own boundaries: before this tick leases, inside
a running handler that enqueues a follow-up, inside the heartbeat's status write, and
inside start(). Each child is a separate process with bounded waits; a restart then
drains what was left.

Found and fixed: a SIGTERM during the tick's preparation (controls, the Telegram drain,
scheduling: seconds with a slow network) still LEASED and started a new job. Inside the
host's 12 s grace a long job (a scan, a paper cycle) is then SIGKILLed mid-run, and an
interrupted automatic PAPER_CYCLE is retired as failed on restart: that cycle is lost.
Now a tick that sees the stop leases nothing; the job stays queued for the next owner.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_CHILD = r"""
import os, signal, sys
repo, root, mode, counter = sys.argv[1:5]
sys.path.insert(0, repo)
from research import autonomy
from research.autonomy import job_store as JS, jobs as JOBS, supervisor as S
from research.autonomy.safe_singleton_lock import install_supervisor_singleton_lock
from tests.test_autonomy_console_runtime import _Deps

def note(name):
    with open(counter, "a") as fh:
        fh.write(name + "\n")

OWNER = {}

def probe(ctx):
    note("probe")
    if mode == "mid-handler":
        OWNER["sup"].jobs.enqueue("probe_follow", idempotency_key="follow:1")
        signal.raise_signal(signal.SIGTERM)          # the stop arrives while the job runs
        note("probe-finished-after-signal")
    return JOBS.JobResult(status=JS.SUCCEEDED, summary="probe")

def follow(ctx):
    note("follow")
    return JOBS.JobResult(status=JS.SUCCEEDED, summary="follow")

JOBS.HANDLERS["probe_job"] = probe
JOBS.HANDLERS["probe_follow"] = follow

class D(_Deps):
    def drain_telegram_alerts(self, min_interval_s=0):
        if mode == "pre-lease":
            signal.raise_signal(signal.SIGTERM)      # before this tick leases anything

class Local(S.Supervisor):
    def __init__(self, root):
        super().__init__(root, deps=D())
        OWNER["sup"] = self
        if mode != "drain":
            self.jobs.enqueue("probe_job", idempotency_key="probe:1")
    def enqueue_due(self, now_ist=None):
        return None                                   # only the probe jobs exist
    def _manage_live_feed(self, current):
        return None
    def start(self):
        ok = super().start()
        if mode == "startup":
            signal.raise_signal(signal.SIGTERM)       # inside start(): its writes are done, no tick yet
        return ok
    def heartbeat(self):
        out = super().heartbeat()
        if mode == "heartbeat" and not OWNER.get("hb"):
            OWNER["hb"] = True
            real = self._write_status
            def interrupted():
                signal.raise_signal(signal.SIGTERM)   # inside the status write
                return real()
            self._write_status = interrupted
        return out

S.Supervisor = Local
autonomy.install_runtime_handlers = install_supervisor_singleton_lock
rc = autonomy.run_supervisor(root=root, interval_s=0.05, max_iterations=(30 if mode in ("drain", "heartbeat") else None))
print("RC", rc, flush=True)
sys.exit(rc)
"""


def _env(tmp_path: Path) -> dict:
    (tmp_path / "child-runtime").mkdir(exist_ok=True)
    return {**os.environ, "QT_RUNTIME_ROOT": str(tmp_path / "child-runtime"),
            "KITE_ACCESS_TOKEN": "", "TELEGRAM_BOT_TOKEN": "", "DEEPSEEK_API_KEY": ""}


def _run(tmp_path: Path, mode: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", _CHILD, str(REPO), str(tmp_path / "auto"), mode,
                           str(tmp_path / "calls.txt")], cwd=REPO, env=_env(tmp_path),
                          capture_output=True, text=True, timeout=90)


def _calls(tmp_path: Path) -> list[str]:
    p = tmp_path / "calls.txt"
    return p.read_text().split() if p.exists() else []


def _jobs(tmp_path: Path) -> dict[str, tuple[str, int]]:
    from research.autonomy.job_store import JobStore

    store = JobStore(tmp_path / "auto" / "jobs.db")
    try:
        return {j.job_type: (j.status, j.attempt) for j in store.list(limit=100)
                if j.job_type in ("probe_job", "probe_follow")}
    finally:
        store.close()


def _owner_free(tmp_path: Path) -> bool:
    from research.autonomy.supervisor import Supervisor

    again = Supervisor(tmp_path / "auto")
    try:
        return again.start()
    finally:
        again.shutdown()


def test_a_stop_during_the_ticks_preparation_leases_nothing(tmp_path):
    r = _run(tmp_path, "pre-lease")
    assert r.returncode == 0 and "RC 0" in r.stdout, r.stderr[-1200:]
    assert _calls(tmp_path) == []                                # no job was started after the stop
    assert _jobs(tmp_path)["probe_job"][0] == "PENDING"
    assert _owner_free(tmp_path)
    r = _run(tmp_path, "drain")                                  # the next owner runs it, once
    assert r.returncode == 0, r.stderr[-1200:]
    assert _calls(tmp_path) == ["probe"] and _jobs(tmp_path)["probe_job"][0] == "SUCCEEDED"


def test_a_stop_inside_a_running_job_finishes_it_and_leaves_its_follow_up_queued_once(tmp_path):
    r = _run(tmp_path, "mid-handler")
    assert r.returncode == 0 and "RC 0" in r.stdout, r.stderr[-1200:]
    assert _calls(tmp_path) == ["probe", "probe-finished-after-signal"]      # never interrupted
    assert _jobs(tmp_path) == {"probe_job": ("SUCCEEDED", 1), "probe_follow": ("PENDING", 0)}
    r = _run(tmp_path, "drain")
    assert r.returncode == 0, r.stderr[-1200:]
    assert _calls(tmp_path) == ["probe", "probe-finished-after-signal", "follow"]   # each exactly once
    assert _jobs(tmp_path) == {"probe_job": ("SUCCEEDED", 1), "probe_follow": ("SUCCEEDED", 1)}


def test_a_stop_inside_the_status_write_keeps_the_status_whole(tmp_path):
    r = _run(tmp_path, "heartbeat")
    assert r.returncode == 0 and "RC 0" in r.stdout, r.stderr[-1200:]
    status = [p for p in (tmp_path / "auto").rglob("*.json") if "status" in p.name]
    assert status and all(isinstance(json.loads(p.read_text()), dict) for p in status)
    assert _owner_free(tmp_path)


def test_a_stop_inside_start_never_ticks_and_releases_the_owner_lock(tmp_path):
    r = _run(tmp_path, "startup")
    assert r.returncode == 0 and "RC 0" in r.stdout, r.stderr[-1200:]
    assert _calls(tmp_path) == [] and _jobs(tmp_path)["probe_job"][0] == "PENDING"
    assert _owner_free(tmp_path)


def test_the_tick_checks_the_stop_right_before_it_leases():
    import inspect

    from research.autonomy.supervisor import Supervisor

    src = inspect.getsource(Supervisor.tick)
    assert src.index("if self._stop:") < src.index("self.jobs.lease_due(")
