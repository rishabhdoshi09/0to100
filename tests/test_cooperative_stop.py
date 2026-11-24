"""The autonomy process's SIGTERM stop, challenged (GPT relay review, 24 Nov).

The handler must never take a lock (a signal can land while the main thread holds it),
never interrupt a write, survive repeated signals, and a forced kill after the grace period
must leave no scheduler lock behind. Every test runs in its own process with bounded waits.
"""
from __future__ import annotations

import inspect
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _env(tmp_path: Path) -> dict:
    (tmp_path / "child-runtime").mkdir(exist_ok=True)
    return {**os.environ, "QT_RUNTIME_ROOT": str(tmp_path / "child-runtime"),
            "KITE_ACCESS_TOKEN": "", "TELEGRAM_BOT_TOKEN": "", "DEEPSEEK_API_KEY": ""}


_EVENT_DEADLOCK = r"""
import signal, sys, threading
sys.path.insert(0, sys.argv[1])
from research.autonomy import install_sigterm_as_stop
class Sup:
    def stop(self): pass
wake = threading.Event()
install_sigterm_as_stop(Sup(), wake)
with wake._cond:                          # inside the Event's critical section, as every wait() is
    signal.raise_signal(signal.SIGTERM)
print("NO DEADLOCK", flush=True)
"""


def test_an_event_wake_deadlocks_when_the_signal_lands_inside_it(tmp_path):
    """Why the wake is a pipe: the hazard the review predicted, reproduced."""
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, "-c", _EVENT_DEADLOCK, str(REPO)], env=_env(tmp_path),
                       capture_output=True, text=True, timeout=3)


_PIPE_WAKE = r"""
import os, signal, sys, threading, time
sys.path.insert(0, sys.argv[1])
from research.autonomy import PipeWake, install_sigterm_as_stop
class Sup:
    stopped = False
    def stop(self): self.stopped = True
sup, wake = Sup(), PipeWake()
assert install_sigterm_as_stop(sup, wake)
signal.raise_signal(signal.SIGTERM)                       # inline: returns at once
threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
assert wake.wait(30) is True                              # the pending wake
t = time.monotonic()
assert wake.wait(30) is True                              # woken DURING select by the second signal
print("OK", sup.stopped, round(time.monotonic() - t, 1), flush=True)
"""


def test_the_pipe_wake_takes_no_lock_and_wakes_a_sleeping_loop(tmp_path):
    r = subprocess.run([sys.executable, "-c", _PIPE_WAKE, str(REPO)], env=_env(tmp_path),
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and r.stdout.startswith("OK True"), r.stderr[-500:]
    from research.autonomy import PipeWake

    code = "".join(inspect.getsource(getattr(PipeWake, name)) for name in ("__init__", "set", "wait"))
    assert not any(word in code for word in ("Lock(", "Condition(", "Event(", "threading"))


_ENTRY = r"""
import os, signal, sys
repo, root, mode = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo)
from research import autonomy
from research.autonomy import job_store as JS, supervisor as S
from research.autonomy.safe_singleton_lock import install_supervisor_singleton_lock
from tests.test_autonomy_console_runtime import _Deps

class InterruptingCommit:
    # The SIGTERM lands between the job's UPDATE and its COMMIT (GPT C8.7-01's window).
    def __init__(self, db):
        self._db, self.fired = db, False
    def __getattr__(self, name):
        return getattr(self._db, name)
    def commit(self):
        if not self.fired:
            self.fired = True
            signal.raise_signal(signal.SIGTERM)
        return self._db.commit()

class Local(S.Supervisor):
    def __init__(self, root):
        super().__init__(root, deps=_Deps())
        self.done = False
    def tick(self, now_ist=None):
        if not self.done:
            self.done = True
            job = self.jobs.enqueue("probe_job")
            leased = self.jobs.lease_due(self.owner)
            if mode == "mid-commit":
                self.jobs._db = InterruptingCommit(self.jobs._db)
            self.jobs.complete(leased.job_id, JS.SUCCEEDED, result_summary="probe")
            if mode == "mid-commit":
                self.jobs._db = self.jobs._db._db
            print("JOB", leased.job_id, flush=True)
            print("READY", flush=True)
        return None

S.Supervisor = Local
autonomy.install_runtime_handlers = install_supervisor_singleton_lock
rc = autonomy.run_supervisor(root=root, interval_s=60.0)
print("RC", rc, flush=True)
print("SIGTERM_IGNORED", signal.getsignal(signal.SIGTERM) is signal.SIG_IGN, flush=True)
if mode == "late":
    import time
    time.sleep(1.0)                        # a window for SIGTERMs AFTER the clean stop
sys.exit(rc)
"""


class _Child:
    """A child process whose stdout is read through ONE file object. Popen.communicate()
    reads the raw descriptors and silently drops lines already buffered by line reads."""

    def __init__(self, proc):
        self.proc, self.seen = proc, []

    def until(self, wanted: str) -> None:
        for line in self.proc.stdout:
            self.seen.append(line.strip())
            if line.strip() == wanted:
                return
        raise AssertionError(f"the child ended before {wanted!r}: {self.seen}")

    def finish(self, timeout: float = 15.0) -> tuple[str, str]:
        try:
            self.proc.wait(timeout=timeout)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
        rest = self.proc.stdout.read()
        return "\n".join(self.seen) + "\n" + rest, self.proc.stderr.read()


def _start(tmp_path, mode):
    root = tmp_path / "auto"
    proc = subprocess.Popen([sys.executable, "-c", _ENTRY, str(REPO), str(root), mode], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(tmp_path))
    child = _Child(proc)
    child.until("READY")
    job_id = next((line.split()[1] for line in child.seen if line.startswith("JOB ")), None)
    return child, root, job_id


def _job_status(root: Path, job_id: str) -> str:
    from research.autonomy.job_store import JobStore

    store = JobStore(root / "jobs.db")
    try:
        return store.get(job_id).status
    finally:
        store.close()


def test_a_sigterm_between_update_and_commit_never_splits_the_write(tmp_path):
    child, root, job_id = _start(tmp_path, "mid-commit")
    out, err = child.finish()                # the signal already fired inside the commit
    assert child.proc.returncode == 0 and "RC 0" in out, err[-800:]
    assert job_id and _job_status(root, job_id) == "SUCCEEDED"


def test_repeated_sigterms_still_end_in_one_clean_stop(tmp_path):
    child, root, job_id = _start(tmp_path, "late")
    child.proc.send_signal(signal.SIGTERM)
    child.proc.send_signal(signal.SIGTERM)                    # repeated while stopping
    child.until("RC 0")                                       # the clean stop is done
    child.proc.send_signal(signal.SIGTERM)                    # and AFTER it, during the exit
    child.proc.send_signal(signal.SIGTERM)
    out, err = child.finish()
    assert child.proc.returncode == 0, (child.proc.returncode, err[-800:])   # not a kill (-15)
    # The mechanism itself, deterministically: once the stop is done, SIGTERM is ignored, so a
    # signal during interpreter finalization (after Python restores the default handlers, a
    # window no test can hit on demand) cannot kill a clean exit.
    assert "SIGTERM_IGNORED True" in out


def test_the_entry_point_wakes_its_loop_with_the_lock_free_pipe():
    from research import autonomy

    src = inspect.getsource(autonomy.run_supervisor)
    assert "wake = PipeWake()" in src and "threading.Event()" not in src


def test_a_forced_kill_after_the_grace_period_leaks_no_scheduler_lock(tmp_path):
    from research.autonomy.supervisor import Supervisor

    child, root, job_id = _start(tmp_path, "plain")
    child.proc.kill()                        # what the host does after STOP_GRACE_S
    child.proc.wait(timeout=15)
    again = Supervisor(root)
    try:
        assert again.start()
    finally:
        again.shutdown()
