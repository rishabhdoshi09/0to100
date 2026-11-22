"""Rebuild the bhavcopy store pickle from its verified official archives: plan, apply, rollback.

The store pickle (``logs/bhav/store_cache.pkl``) is a CACHE of the official
daily archives. Before 22 Nov 2026 (GPT cross-review RS-02) it could keep live
intraday overlay rows as history, frames could go out of date order after a late
archive, and names below the 30-row build floor lost their first weeks of
history. The archives themselves are untouched by any of that, so the correct
store is ``data.bhavcopy_store.build_from_local()`` over them: the store's own
parser and builder, no network, no second database.

Three phases, each refusing anything it cannot prove:

* ``plan``: builds a NEW pickle in a staging root that the caller owns (never
  under ``/Volumes``), audits it against the archives
  (research.store_integrity_audit), diffs it row by row against the CURRENT
  pickle and writes a manifest (sha256 of both, counts, every gate). Read-only
  for the target.
* ``apply``: replaces the target pickle with the planned one. It refuses unless
  the caller states the owner's approval in words, every plan gate passed, the
  target pickle still has the sha256 the plan saw, and no process holds it
  open. The old pickle is kept as ``store_cache.pkl.pre-rebuild-<stamp>`` with
  its sha256 verified after the copy; the swap is an atomic rename.
* ``rollback``: restores that backup (sha256 checked before and after).

Live trading and the learning ledgers are not touched. A running desk keeps its
in-memory store until it restarts: apply only with the desk stopped (the
open-file check enforces it for the pickle itself).

    python -m product.bhav_store_rebuild plan --target <runtime root> --staging <dir> --out <manifest.json>
    python -m product.bhav_store_rebuild apply --manifest <manifest.json> --owner-approval "<the owner's words>"
    python -m product.bhav_store_rebuild rollback --manifest <manifest.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PICKLE = Path("logs") / "bhav" / "store_cache.pkl"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _not_live(path: Path, what: str) -> Path:
    p = Path(path).resolve()
    if str(p).startswith("/Volumes/"):
        raise SystemExit(f"refusing {what} under /Volumes: {p} (stage on a disk you own)")
    return p


def _stage(target: Path, staging: Path) -> Path:
    """A staging runtime whose archives are read-only links to the target's archives."""
    src = target / "logs" / "bhav"
    dst = staging / "logs" / "bhav"
    if dst.exists():
        raise SystemExit(f"staging already exists: {dst} (use a fresh directory)")
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.suffix.lower() == ".csv" and f.is_file():
            os.symlink(f, dst / f.name)
    return staging


def _build_in(staging: Path, keep_symbols: list[str]) -> dict[str, Any]:
    """Run the store's own local build in a FRESH interpreter bound to the staging root.

    The full build drops names below its 30-row floor, while appends have always
    admitted a new listing from its first row, so the live store serves young
    listings a bare rebuild would silently remove. A reconstruction changes ROWS
    only: every name the current store holds is promoted from the build's
    pending (below-floor) set with its complete official history.
    """
    keep_file = staging / "keep_symbols.json"
    keep_file.write_text(json.dumps(sorted(keep_symbols)))
    code = (
        "import json, sys\n"
        "from data import bhavcopy_store as bs\n"
        "n = bs.build_from_local()\n"
        f"keep = json.load(open({str(keep_file)!r}))\n"
        "promoted = 0\n"
        "with bs._lock:\n"
        "    for sym in keep:\n"
        "        if sym not in bs._store and sym in bs._pending:\n"
        "            bs._store[sym] = bs._pending.pop(sym)\n"
        "            promoted += 1\n"
        "bs._save_pkl()\n"
        "print(json.dumps({'built_symbols': n, 'promoted_below_floor': promoted, 'symbols': len(bs._store),\n"
        "                  'pending': len(bs._pending), 'last_day': str(bs._store_last_day),\n"
        "                  'sessions': bs._store_sessions}))\n"
    )
    env = dict(os.environ, QT_RUNTIME_ROOT=str(staging), PYTHONDONTWRITEBYTECODE="1")
    for key in _SAFE_ENV:
        env[key] = ""
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=env, capture_output=True,
                         text=True, timeout=3600)
    if out.returncode != 0:
        raise SystemExit(f"staging build failed: {out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


_SAFE_ENV = ("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def _audit_in(staging: Path, pickle_path: Path, out: Path) -> dict[str, Any]:
    """research.store_integrity_audit in a FRESH interpreter bound to the staging root."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("QT_RUNTIME_ROOT", None)
    for key in _SAFE_ENV:
        env[key] = ""
    run = subprocess.run([sys.executable, "-m", "research.store_integrity_audit", "--root", str(staging),
                          "--pickle", str(pickle_path), "--out", str(out)],
                         cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    if run.returncode != 0:
        raise SystemExit(f"audit failed: {run.stderr[-2000:]}")
    return json.loads(out.read_text())


#: Processes of a running desk. The desk keeps its store in memory and saves it
#: again on the next append, which would overwrite a rebuilt pickle with the
#: old store: apply and rollback need the desk stopped, not only the file closed.
DESK_PATTERNS = ("terminal_product_api", "main.py autonomy", "operations/market_ops", "operations.market_ops",
                 "host_supervisor", "run_quantterm_complete")


def _desk_running() -> list[str]:
    found = []
    for pattern in DESK_PATTERNS:
        try:
            out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=30)
        except Exception:
            return ["pgrep unavailable"]
        found += [f"{pattern}:{pid}" for pid in out.stdout.split() if pid.strip() and int(pid) != os.getpid()]
    return found


def _frames(path: Path) -> dict:
    with open(path, "rb") as fh:
        return dict((pickle.load(fh) or {}).get("store") or {})


def _row_diff(old: dict, new: dict) -> dict[str, Any]:
    """Rows removed, added and changed between two stores (normalized dates)."""
    import numpy as np
    import pandas as pd

    removed, added, changed, unsorted_old = [], [], [], []
    for sym in sorted(set(old) | set(new)):
        a, b = old.get(sym), new.get(sym)
        ai = pd.DatetimeIndex([]) if a is None else pd.DatetimeIndex(a.index).normalize()
        bi = pd.DatetimeIndex([]) if b is None else pd.DatetimeIndex(b.index).normalize()
        if a is not None and not ai.is_monotonic_increasing:
            unsorted_old.append(sym)
        removed += [(sym, t.date().isoformat()) for t in ai.difference(bi)]
        added += [(sym, t.date().isoformat()) for t in bi.difference(ai)]
        common = ai.intersection(bi)
        if a is None or b is None or not len(common):
            continue
        a2 = a.copy(); a2.index = ai; a2 = a2[~a2.index.duplicated(keep="last")]
        b2 = b.copy(); b2.index = bi
        for col in sorted(set(a2.columns) & set(b2.columns)):
            x = pd.to_numeric(a2.loc[common, col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(b2.loc[common, col], errors="coerce").to_numpy(dtype=float)
            bad = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            changed += [(sym, t.date().isoformat(), col) for t in common[bad]]
    return {"removed": removed, "added": added, "changed": changed, "unsorted_in_old": unsorted_old}


def plan(target: str | Path, staging: str | Path, out: str | Path) -> dict[str, Any]:
    target = Path(target).resolve()
    staging = _not_live(Path(staging), "the staging root")
    out = _not_live(Path(out), "the manifest")
    current = target / PICKLE
    if not current.exists():
        raise SystemExit(f"no current pickle at {current}")
    before = _sha256(current)
    _stage(target, staging)
    # The current pickle is audited from a copy on the staging disk: the
    # target is only ever read.
    current_copy = staging / "store_cache.current.pkl"
    shutil.copy2(current, current_copy)
    if _sha256(current_copy) != before:
        raise SystemExit("the copy of the current pickle does not match it")
    old_frames = _frames(current_copy)
    built = _build_in(staging, list(old_frames))
    new = staging / PICKLE
    audited = _audit_in(staging, new, staging / "audit_new.json")
    old_audit = _audit_in(staging, current_copy, staging / "audit_current.json")
    new_frames = _frames(new)
    diff = _row_diff(old_frames, new_frames)
    contamination = {k: audited["counts"].get(k, 0) for k in
                     ("ORPHAN_ON_ARCHIVE_DATE", "NO_ARCHIVE_FOR_DATE", "VALUE_MISMATCH", "DUPLICATE_DATE",
                      "MISSING_OFFICIAL_ROW", "MISSING_SYMBOL")}
    gates = {
        "target_unchanged_during_plan": _sha256(current) == before,
        "new_store_has_no_contamination": not any(contamination.values()),
        "new_store_last_day_is_latest_archive": built["last_day"] == audited["archives"]["last"],
        "new_store_sessions_equal_verified_archives": int(built["sessions"]) == audited["archives"]["verified_sessions"],
        "no_value_changed_on_a_kept_row": not diff["changed"],
        "symbol_set_preserved": set(new_frames) == set(old_frames),
        "removed_rows_are_exactly_the_contamination": (
            sorted(map(tuple, diff["removed"])) == sorted(
                map(tuple, old_audit["findings"].get("ORPHAN_ON_ARCHIVE_DATE", [])
                    + old_audit["findings"].get("NO_ARCHIVE_FOR_DATE", [])))),
        "added_rows_are_exactly_the_missing_official_rows": (
            sorted(map(tuple, diff["added"])) == sorted(
                map(tuple, old_audit["findings"].get("MISSING_OFFICIAL_ROW", [])))),
    }
    manifest = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_root": str(target),
        "target_pickle": str(current),
        "target_sha256": before,
        "staging_root": str(staging),
        "new_pickle": str(new),
        "new_sha256": _sha256(new),
        "build": built,
        "audit_of_new": {"counts": audited["counts"], "archives": audited["archives"]},
        "audit_of_current": {"counts": old_audit["counts"], "affected_symbols": old_audit["affected_symbols"],
                             "affected_sessions": old_audit["affected_sessions"]},
        "diff": {"removed": len(diff["removed"]), "added": len(diff["added"]), "changed": len(diff["changed"]),
                 "removed_rows": diff["removed"][:500], "unsorted_in_old": diff["unsorted_in_old"],
                 "added_symbols": sorted({s for s, _ in diff["added"]})},
        "gates": gates,
        "ready": all(gates.values()),
    }
    Path(out).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _held_open(path: Path) -> list[str]:
    try:
        out = subprocess.run(["lsof", "-t", "--", str(path)], capture_output=True, text=True, timeout=30)
    except Exception:
        return ["lsof unavailable"]
    return [p for p in out.stdout.split() if p.strip()]


def _check_desk(target: Path, target_is_copy: bool) -> None:
    if target_is_copy and not str(target.resolve()).startswith("/Volumes/"):
        return
    running = _desk_running()
    if running:
        raise SystemExit(f"refusing: the desk is running ({running}); stop it first")


def apply(manifest_path: str | Path, owner_approval: str, *, target_is_copy: bool = False) -> dict[str, Any]:
    m = json.loads(Path(manifest_path).read_text())
    if not str(owner_approval or "").strip():
        raise SystemExit("refusing: apply needs the owner's approval in words (--owner-approval)")
    if not m.get("ready"):
        raise SystemExit(f"refusing: the plan's gates did not all pass: {m.get('gates')}")
    target, new = Path(m["target_pickle"]), Path(m["new_pickle"])
    if _sha256(target) != m["target_sha256"]:
        raise SystemExit("refusing: the target pickle changed since the plan (re-plan)")
    if _sha256(new) != m["new_sha256"]:
        raise SystemExit("refusing: the planned pickle changed since the plan")
    _check_desk(target, target_is_copy)
    holders = _held_open(target)
    if holders:
        raise SystemExit(f"refusing: the target pickle is held open by {holders} (stop the desk first)")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(f"{target.name}.pre-rebuild-{stamp}")
    shutil.copy2(target, backup)
    if _sha256(backup) != m["target_sha256"]:
        backup.unlink(missing_ok=True)
        raise SystemExit("refusing: the backup copy does not match the target")
    tmp = target.with_name(f"{target.name}.rebuild-tmp")
    shutil.copy2(new, tmp)
    if _sha256(tmp) != m["new_sha256"]:
        tmp.unlink(missing_ok=True)
        raise SystemExit("refusing: the staged copy does not match the plan")
    os.replace(tmp, target)
    ok = _sha256(target) == m["new_sha256"]
    m.update({"applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "backup_pickle": str(backup),
              "owner_approval": owner_approval.strip(), "applied_ok": ok})
    Path(manifest_path).write_text(json.dumps(m, indent=2, default=str))
    if not ok:
        raise SystemExit("the target does not match the planned pickle after the swap: run rollback")
    return m


def rollback(manifest_path: str | Path, *, target_is_copy: bool = False) -> dict[str, Any]:
    m = json.loads(Path(manifest_path).read_text())
    recorded = str(m.get("backup_pickle") or "").strip()
    backup = Path(recorded) if recorded else None
    if backup is None or not backup.is_file():
        raise SystemExit("nothing to roll back: no backup recorded")
    if _sha256(backup) != m["target_sha256"]:
        raise SystemExit("refusing: the backup does not match the pre-rebuild sha256")
    target = Path(m["target_pickle"])
    _check_desk(target, target_is_copy)
    holders = _held_open(target)
    if holders:
        raise SystemExit(f"refusing: the target pickle is held open by {holders} (stop the desk first)")
    tmp = target.with_name(f"{target.name}.rollback-tmp")
    shutil.copy2(backup, tmp)
    os.replace(tmp, target)
    ok = _sha256(target) == m["target_sha256"]
    m.update({"rolled_back_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rollback_ok": ok})
    Path(manifest_path).write_text(json.dumps(m, indent=2, default=str))
    if not ok:
        raise SystemExit("rollback did not restore the pre-rebuild pickle")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--target", required=True)
    p.add_argument("--staging", required=True); p.add_argument("--out", required=True)
    a = sub.add_parser("apply"); a.add_argument("--manifest", required=True)
    a.add_argument("--owner-approval", required=True)
    a.add_argument("--target-is-copy", action="store_true",
                   help="the target is an isolated copy (never honoured under /Volumes)")
    r = sub.add_parser("rollback"); r.add_argument("--manifest", required=True)
    r.add_argument("--target-is-copy", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        out = plan(args.target, args.staging, args.out)
        print(json.dumps({k: out[k] for k in ("ready", "gates", "build", "diff")}, indent=1, default=str)[:4000])
    elif args.cmd == "apply":
        out = apply(args.manifest, args.owner_approval, target_is_copy=args.target_is_copy)
        print(json.dumps({k: out.get(k) for k in ("applied_ok", "backup_pickle", "new_sha256")}, indent=1))
    else:
        out = rollback(args.manifest, target_is_copy=args.target_is_copy)
        print(json.dumps({k: out.get(k) for k in ("rollback_ok", "target_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
