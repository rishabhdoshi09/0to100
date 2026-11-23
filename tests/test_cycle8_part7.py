"""Cycle 8, part 7: a Claude subagent's review of 2787a690..2cf44fc0, re-run against the
current code. Findings 1, 2 and 6 were already fixed by part 6; 7 is a conservative
refusal, kept by design. These pin the rest:

| finding | test |
|---|---|
| 3 test children stripped QT_RUNTIME_ROOT (the checkout's pointer is production's runtime) | their env now carries a temporary root (crash, keeper and runbook tests) |
| 4 a forked child's NORMAL exit closed the parent's keeper: 15 committed rows became 5 | test_a_forked_child_exiting_normally_never_loses_other_commits |
| 5 SIGTERM killed the autonomy process without atexit: a hot WAL the F1 tools refuse | test_sigterm_is_a_cooperative_stop_of_the_real_entry_point |
| 8 re-planning to the same --out erased the apply's record: rollback impossible | test_a_re_plan_never_erases_the_record_of_an_apply |

Part 8 (GPT C8.7-01..07 on these fixes): SIGTERM no longer raises KeyboardInterrupt at an
arbitrary point (it could split a job-store write from its commit and is swallowed by broad
handlers); it asks the loop to stop between ticks. Plus: re-plan after a rollback carries
the backup history (C8.7-06), a concurrent apply during a re-plan is never overwritten
(C8.7-04), the fork hook imports nothing (C8.7-05), the SIGTERM test has no readiness race
and runs the real entry point (C8.7-07).
"""
from __future__ import annotations

import inspect
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from product import sqlite_index_repair as repair
from tests.test_cycle8_review import f1, plan_f1  # noqa: F401  (the shared fixture)

REPO = Path(__file__).resolve().parents[1]


def _env(tmp_path: Path) -> dict:
    (tmp_path / "child-runtime").mkdir(exist_ok=True)      # exists, as QT_RUNTIME_ROOT_REQUIRE_EXISTING needs
    return {**os.environ, "QT_RUNTIME_ROOT": str(tmp_path / "child-runtime"),
            "KITE_ACCESS_TOKEN": "", "TELEGRAM_BOT_TOKEN": "", "DEEPSEEK_API_KEY": ""}


# ── finding 4 ──

_FORK = r'''
import os, shutil, sqlite3, subprocess, sys, tempfile
repo, base = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from product import decision_freeze, sqlite_runtime
db = os.path.join(base, "freeze.db")
rec = lambda tag, i: {"decision_id": f"{tag}{i}", "symbol": "S", "as_of": "2026-11-24",
                      "decision": "WATCH", "reason_code": "T"}

def rows():
    d = tempfile.mkdtemp(dir=base)
    c = os.path.join(d, "c.db")
    shutil.copyfile(db, c)
    if os.path.exists(db + "-wal"):
        shutil.copyfile(db + "-wal", c + "-wal")
    con = sqlite3.connect(c)
    n = con.execute("SELECT count(*) FROM freezes").fetchone()[0]
    con.close()
    return n

for i in range(5):
    decision_freeze.freeze(rec("p", i), path=db)          # this process's keeper holds the store
r, w = os.pipe()
pid = os.fork()
if pid == 0:                                              # the child inherits the keeper ...
    os.close(w)
    os.read(r, 1)                                         # ... waits for the other writer ...
    sys.exit(0)                                           # ... and exits NORMALLY (interpreter shutdown)
os.close(r)
sqlite_runtime.close_keepers()
other = ("import os, sys\nsys.path.insert(0, sys.argv[1])\nfrom product import decision_freeze as f\n"
         "for i in range(10):\n    f.freeze({'decision_id': f'q{i}', 'symbol': 'S', 'as_of': '2026-11-24',"
         " 'decision': 'WATCH', 'reason_code': 'T'}, path=sys.argv[2])\nos._exit(0)\n")
done = subprocess.run([sys.executable, "-c", other, repo, db], capture_output=True, text=True, timeout=120)
assert done.returncode == 0, done.stderr[-500:]
before = rows()
os.write(w, b"x")
_pid, status = os.waitpid(pid, 0)
print(json.dumps({"before": before, "after": rows(), "child": os.waitstatus_to_exitcode(status)}))
'''.replace("import os, shutil", "import json, os, shutil")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
def test_a_forked_child_exiting_normally_never_loses_other_commits(tmp_path):
    r = subprocess.run([sys.executable, "-c", _FORK, str(REPO), str(tmp_path)], capture_output=True, text=True,
                       env=_env(tmp_path), timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"before": 15, "after": 15, "child": 0}


# ── finding 5, as redesigned in part 8: a cooperative stop of the REAL entry point ──

_ENTRY = r"""
import os, sys
repo, root, db = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo)
from research import autonomy
from research.autonomy import supervisor as S
from research.autonomy.safe_singleton_lock import install_supervisor_singleton_lock
from product import decision_freeze
from tests.test_autonomy_console_runtime import _Deps

class Local(S.Supervisor):
    def __init__(self, root):
        super().__init__(root, deps=_Deps())
        self.froze = False

    def tick(self, now_ist=None):
        if not self.froze:
            for i in range(30):
                decision_freeze.freeze({"decision_id": f"d{i}", "symbol": "S", "as_of": "2026-11-24",
                                        "decision": "WATCH", "reason_code": "T"}, path=db)
            self.froze = True
            print("READY", flush=True)            # the handler is installed before start(): no race
        return None

S.Supervisor = Local
autonomy.install_runtime_handlers = install_supervisor_singleton_lock
rc = autonomy.run_supervisor(root=root, interval_s=60.0)
print("RC", rc, flush=True)
sys.exit(rc)
"""


def test_sigterm_is_a_cooperative_stop_of_the_real_entry_point(tmp_path):
    from product.host_supervisor import STOP_GRACE_S
    from research.autonomy.supervisor import Supervisor

    (tmp_path / "live").mkdir()
    db = tmp_path / "live" / "counterfactuals.jsonl.freeze.db"
    root = tmp_path / "auto"
    proc = subprocess.Popen([sys.executable, "-c", _ENTRY, str(REPO), str(root), str(db)], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(tmp_path))
    try:
        lines = []
        for line in proc.stdout:                                  # the banner first, then READY
            lines.append(line.strip())
            if line.strip() == "READY":
                break
        assert "READY" in lines, (lines, proc.stderr.read()[-800:] if proc.poll() is not None else "")
        wal = Path(str(db) + "-wal")
        assert wal.exists() and wal.stat().st_size > 0            # the keeper holds the store open
        proc.send_signal(signal.SIGTERM)                          # what the host supervisor sends
        out, err = proc.communicate(timeout=STOP_GRACE_S)         # within the host's grace period
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0 and "RC 0" in out, err[-800:]
    assert "scheduler lock released" in out
    assert not wal.exists() or wal.stat().st_size == 0            # the keeper checkpointed at exit
    d = repair.check(db, tmp_path / "stage")
    assert d["inconclusive"] is False and d["integrity"] == ["ok"]
    from contextlib import closing
    import sqlite3
    with closing(sqlite3.connect(db)) as con:
        assert con.execute("SELECT count(*) FROM freezes").fetchone()[0] == 30
    again = Supervisor(root)
    try:
        assert again.start()                                      # the scheduler lock is free again
    finally:
        again.shutdown()


def test_the_autonomy_entry_installs_the_stop_before_start():
    from research import autonomy

    src = inspect.getsource(autonomy.run_supervisor)
    assert "install_sigterm_as_stop(sup, wake)" in src and "sleep_fn=wake.wait" in src
    assert src.index("install_sigterm_as_stop(sup, wake)") < src.index("sup.start()")


def test_the_handler_only_stops_and_wakes():
    import threading

    from research.autonomy import install_sigterm_as_stop

    class Sup:
        stopped = False

        def stop(self):
            self.stopped = True

    sup, wake = Sup(), threading.Event()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        assert install_sigterm_as_stop(sup, wake) is True
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)    # no exception: a cooperative stop
        assert sup.stopped and wake.is_set()
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_an_existing_sigterm_handler_is_never_replaced():
    import threading

    from research.autonomy import install_sigterm_as_stop

    def mine(_s, _f):
        pass

    previous = signal.signal(signal.SIGTERM, mine)
    try:
        assert install_sigterm_as_stop(object(), threading.Event()) is False
        assert signal.getsignal(signal.SIGTERM) is mine
    finally:
        signal.signal(signal.SIGTERM, previous)


# ── finding 8 ──

def test_a_re_plan_never_erases_the_record_of_an_apply(f1):
    db, manifest = f1
    plan_f1(db, manifest)
    plan_f1(db, manifest)                                         # before any apply: re-planning is fine
    applied = repair.apply(manifest, "test", target_is_copy=True)
    kept = manifest.read_text()
    with pytest.raises(SystemExit, match="record of an apply"):
        repair.plan(db, manifest.parent / "stage2", manifest)
    assert manifest.read_text() == kept
    done = repair.rollback(manifest, target_is_copy=True)
    assert done["state"] == "rolled_back" and json.loads(manifest.read_text())["backup"] == applied["backup"]


def test_a_rolled_back_attempt_can_be_re_planned_and_keeps_its_history(f1):
    db, manifest = f1
    plan_f1(db, manifest)
    applied = repair.apply(manifest, "test", target_is_copy=True)
    assert repair.rollback(manifest, target_is_copy=True)["state"] == "rolled_back"
    m = plan_f1(db, manifest)                                     # the documented same-name re-plan
    assert m["ready"] and m["state"] == "planned"
    assert applied["backup"] in m["earlier_backups"] and m["replanned_over"]["state"] == "rolled_back"
    assert Path(applied["backup"]).exists()


def test_an_apply_during_a_re_plan_is_never_overwritten(f1, monkeypatch):
    db, manifest = f1
    plan_f1(db, manifest)
    quiet_copy = repair._quiet_copy
    applied = {}

    def copy_then_apply(src, dst):
        copied = quiet_copy(src, dst)
        if not applied:
            applied.update(repair.apply(manifest, "test", target_is_copy=True))
        return copied

    monkeypatch.setattr(repair, "_quiet_copy", copy_then_apply)
    with pytest.raises(SystemExit, match="changed while the plan was made"):
        repair.plan(db, manifest.parent / "stage2", manifest)
    monkeypatch.undo()
    assert json.loads(manifest.read_text())["backup"] == applied["backup"]
    assert repair.rollback(manifest, target_is_copy=True)["state"] == "rolled_back"


_NO_CTYPES = r"""
import os, sqlite3, sys
sys.path.insert(0, sys.argv[1])
from product import sqlite_runtime as rt
db = sys.argv[2]
con = sqlite3.connect(db)
con.execute("PRAGMA journal_mode=WAL")
rt.keep_open(db)
con.close()
assert len(rt._KEEPERS) == 1
sys.modules["ctypes"] = None                     # the fork hook must not need to import anything
child = os.fork()
if child == 0:
    os._exit(0 if not rt._KEEPERS and len(rt._INHERITED) == 1 else 3)
_, status = os.waitpid(child, 0)
sys.exit(os.waitstatus_to_exitcode(status))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
def test_the_fork_hook_pins_without_importing(tmp_path):
    r = subprocess.run([sys.executable, "-c", _NO_CTYPES, str(REPO), str(tmp_path / "keeper.db")],
                       env=_env(tmp_path), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]
