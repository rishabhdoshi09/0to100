"""Cycle 8, part 7: a Claude subagent's review of 2787a690..2cf44fc0, re-run against the
current code. Findings 1, 2 and 6 were already fixed by part 6; 7 is a conservative
refusal, kept by design. These pin the rest:

| finding | test |
|---|---|
| 3 test children stripped QT_RUNTIME_ROOT (the checkout's pointer is production's runtime) | their env now carries a temporary root (crash, keeper and runbook tests) |
| 4 a forked child's NORMAL exit closed the parent's keeper: 15 committed rows became 5 | test_a_forked_child_exiting_normally_never_loses_other_commits |
| 5 SIGTERM killed the autonomy process without atexit: a hot WAL the F1 tools refuse | test_sigterm_stops_like_ctrl_c_and_leaves_no_hot_wal |
| 8 re-planning to the same --out erased the apply's record: rollback impossible | test_a_re_plan_never_erases_the_record_of_an_apply |
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


# ── finding 5 ──

_SERVE = r'''
import sys, time
repo, db = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
from research.autonomy import install_sigterm_as_interrupt
from product import decision_freeze
assert install_sigterm_as_interrupt()
for i in range(30):
    decision_freeze.freeze({"decision_id": f"d{i}", "symbol": "S", "as_of": "2026-11-24",
                            "decision": "WATCH", "reason_code": "T"}, path=db)
print("READY", flush=True)
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    print("STOPPING", flush=True)
'''


def test_sigterm_stops_like_ctrl_c_and_leaves_no_hot_wal(tmp_path):
    (tmp_path / "live").mkdir()
    db = tmp_path / "live" / "counterfactuals.jsonl.freeze.db"
    proc = subprocess.Popen([sys.executable, "-c", _SERVE, str(REPO), str(db)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=_env(tmp_path))
    try:
        assert proc.stdout.readline().strip() == "READY"
        wal = Path(str(db) + "-wal")
        assert wal.exists() and wal.stat().st_size > 0            # the keeper holds the store open
        proc.send_signal(signal.SIGTERM)                          # what the host supervisor sends
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0 and "STOPPING" in out, err[-500:]
    assert not wal.exists() or wal.stat().st_size == 0            # checkpointed at exit
    repair._held_open = lambda *_: []
    d = repair.check(db, tmp_path / "stage")
    assert d["inconclusive"] is False and d["integrity"] == ["ok"]


def test_the_autonomy_entry_installs_it_first():
    from research import autonomy

    src = inspect.getsource(autonomy.run_supervisor)
    assert src.index("install_sigterm_as_interrupt()") < src.index("Supervisor(")


def test_an_existing_sigterm_handler_is_never_replaced():
    from research.autonomy import install_sigterm_as_interrupt

    def mine(_s, _f):
        pass

    previous = signal.signal(signal.SIGTERM, mine)
    try:
        assert install_sigterm_as_interrupt() is False
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
