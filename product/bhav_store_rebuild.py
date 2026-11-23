"""Rebuild the bhavcopy store pickle from its verified official archives: plan, apply, rollback.

The store pickle (``logs/bhav/store_cache.pkl``) is a CACHE of the official
daily archives. Before 22 Nov 2026 (GPT cross-review RS-02) it could keep live
intraday overlay rows as history, frames could go out of date order after a late
archive, and names below the 30-row build floor lost their first weeks of
history. The archives themselves are untouched by any of that, so the correct
store is ``data.bhavcopy_store.build_from_local()`` over them: the store's own
parser and builder, no network, no second database.

Three phases, each refusing anything it cannot prove:

* ``plan`` copies an IMMUTABLE snapshot of the target's archives and runtime
  calendar into a staging root the caller owns (never under ``/Volumes``) and
  records a source manifest (name, size, mtime, sha256 per archive). It builds
  a NEW pickle there in a fresh interpreter, keeps every name and the key ORDER
  the current store serves (the deployed walk-forward samples by that order),
  audits the new and the current pickle against the snapshot
  (research.store_integrity_audit, complete finding lists) and diffs them row
  by row. Every gate is written to the manifest. Read-only for the target.
* ``apply`` refuses unless the caller states the owner's approval in words,
  every gate passed, the target pickle still has the planned sha256, the
  target's archives and calendar still match the source manifest, no desk
  process or launchd service runs (the desk saves its in-memory store again on
  the next append) and the probe itself succeeded. It takes the store's
  maintenance lock (``data.bhavcopy_store.maintenance_lock_path``: no store
  writer saves while it exists), writes a transaction record, keeps a
  sha-verified ``store_cache.pkl.pre-rebuild-<stamp>`` backup, swaps by an
  atomic same-directory rename of a verified, fsynced temp file, and releases
  the lock.
* ``rollback`` restores that backup only while the target still holds exactly
  the applied pickle: a cache the desk saved after the apply is never discarded.

Live trading and the learning ledgers are not touched.

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
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PUBLISH_LOCK = ".store_publish.lock"    # data.bhavcopy_store.PUBLISH_LOCK
_SAFE_ENV = ("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")

#: Command lines of every process that can hold or save the store: the launchd
#: entrypoints and supervisor, the startup writer, the API (both stacks), the
#: report API, autonomy and market ops. Matched with ``pgrep -f``.
DESK_PATTERNS = (
    "product.host_entrypoint", "product.host_launchd_entrypoint", "product.host_bootstrap",
    "host_supervisor", "run_quantterm_complete", "run_quantterm.sh",
    "terminal_product_api", "api.app:app", "report_api", "main.py autonomy",
    "operations.market_ops", "operations/market_ops",
)
LAUNCHD_LABEL = "com.quantterm.desk"


def _logs(root: Path, *parts: str) -> Path:
    """``logs/<parts>`` of the runtime at ``root``, resolved by core.runtime_paths.

    This tool addresses explicit runtimes (the target and a staging copy), not
    the process's own, so the runtime root is bound per call and restored.
    """
    from core.runtime_paths import ENV_VAR, logs_path

    previous = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = str(root)
    try:
        return logs_path(*parts)
    finally:
        if previous is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = previous


def _pickle(root: Path) -> Path:
    return _logs(Path(root), "bhav", "store_cache.pkl")


def _calendar(root: Path) -> Path:
    return _logs(Path(root), "nse_holidays.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _on_live_volume(path: str | Path) -> bool:
    """Under /Volumes, compared case-insensitively: the Mac's volumes are
    case-insensitive and realpath keeps the spelling given, so '/volumes/...'
    reaches the same production files (cycle-7 review)."""
    text = str(path).lower().rstrip("/")
    return text == "/volumes" or text.startswith("/volumes/")      # the mount root too (GPT V2-04)


def _not_live(path: Path, what: str) -> Path:
    p = Path(path).resolve()
    if _on_live_volume(p):
        raise SystemExit(f"refusing {what} under /Volumes: {p} (stage on a disk you own)")
    return p


def _archive_files(root: Path) -> list[Path]:
    return sorted(f for f in _logs(root, "bhav").iterdir() if f.suffix.lower() == ".csv" and f.is_file())


def _source_manifest(root: Path, *, hashes: bool) -> dict[str, Any]:
    """Identity of the archives and runtime calendar the store would be built from."""
    files = {}
    for f in _archive_files(root):
        st = f.stat()
        files[f.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                         **({"sha256": _sha256(f)} if hashes else {})}
    cal = _calendar(root)
    calendar = {"present": cal.exists(), **({"sha256": _sha256(cal)} if cal.exists() else {})}
    return {"archives": files, "calendar": calendar}


def _stage(target: Path, staging: Path) -> dict[str, Any]:
    """An immutable snapshot of the target's archives and calendar in a fresh staging root."""
    dst = _logs(staging, "bhav")
    if dst.exists():
        raise SystemExit(f"staging already exists: {dst} (use a fresh directory)")
    dst.mkdir(parents=True)
    archives = {}
    for f in _archive_files(target):
        before = f.stat()
        shutil.copy2(f, dst / f.name)
        after = f.stat()
        ident = (before.st_size, before.st_mtime_ns, before.st_ino)
        if ident != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise SystemExit(f"refusing: archive {f.name} changed while it was being staged (re-plan)")
        copied = dst / f.name
        if copied.stat().st_size != before.st_size:
            raise SystemExit(f"refusing: the staged copy of {f.name} is incomplete")
        archives[f.name] = {"size": before.st_size, "mtime_ns": before.st_mtime_ns, "ino": before.st_ino,
                            "sha256": _sha256(copied)}
    if {f.name for f in _archive_files(target)} != set(archives):
        raise SystemExit("refusing: the target's archive set changed while it was being staged (re-plan)")
    if _calendar(target).exists():
        shutil.copy2(_calendar(target), _calendar(staging))
    staged_cal = _calendar(staging)
    calendar = {"present": staged_cal.exists(), **({"sha256": _sha256(staged_cal)} if staged_cal.exists() else {})}
    return {"archives": archives, "calendar": calendar}


def _env(**extra) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **extra)
    for key in _SAFE_ENV:
        env[key] = ""
    return env


def _build_in(staging: Path, keep_order: list[str]) -> dict[str, Any]:
    """The store's own local build, in a FRESH interpreter bound to the staging root.

    The full build drops names below its 30-row floor, while appends admit a new
    listing from its first row, so the live store serves young listings a bare
    rebuild would silently remove. A reconstruction changes ROWS only: every
    name the current store holds is promoted from the build's pending set with
    its complete history, and the served keys keep the current store's order.
    """
    keep_file = staging / "keep_order.json"
    keep_file.write_text(json.dumps(list(keep_order)))
    code = (
        "import json\n"
        "from data import bhavcopy_store as bs\n"
        "n = bs.build_from_local()\n"
        f"keep = json.load(open({str(keep_file)!r}))\n"
        "promoted = 0\n"
        "with bs._lock:\n"
        "    for sym in keep:\n"
        "        if sym not in bs._store and sym in bs._pending:\n"
        "            bs._store[sym] = bs._pending.pop(sym)\n"
        "            promoted += 1\n"
        "    held = set(keep)\n"
        "    ordered = {s: bs._store[s] for s in keep if s in bs._store}\n"
        "    ordered.update({s: bs._store[s] for s in sorted(bs._store) if s not in held})\n"
        "    bs._store = ordered\n"
        "bs._save_pkl()\n"
        "print(json.dumps({'built_symbols': n, 'promoted_below_floor': promoted, 'symbols': len(bs._store),\n"
        "                  'pending': len(bs._pending), 'last_day': str(bs._store_last_day),\n"
        "                  'sessions': bs._store_sessions}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), env=_env(QT_RUNTIME_ROOT=str(staging)),
                         capture_output=True, text=True, timeout=3600)
    if out.returncode != 0:
        raise SystemExit(f"staging build failed: {out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _audit_in(staging: Path, pickle_path: Path, out: Path) -> dict[str, Any]:
    """research.store_integrity_audit in a FRESH interpreter bound to the staging root."""
    env = _env()
    env.pop("QT_RUNTIME_ROOT", None)
    run = subprocess.run([sys.executable, "-m", "research.store_integrity_audit", "--root", str(staging),
                          "--pickle", str(pickle_path), "--out", str(out)],
                         cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    if run.returncode != 0:
        raise SystemExit(f"audit failed: {run.stderr[-2000:]}")
    return json.loads(out.read_text())


def _frames(path: Path) -> dict:
    with open(path, "rb") as fh:
        return dict((pickle.load(fh) or {}).get("store") or {})


def _row_diff(old: dict, new: dict) -> dict[str, Any]:
    """Every row removed, added and value-changed between two stores (complete multisets)."""
    import numpy as np
    import pandas as pd

    removed, added, changed, unsorted_old = Counter(), Counter(), Counter(), []
    for sym in sorted(set(old) | set(new)):
        a, b = old.get(sym), new.get(sym)
        ai = pd.DatetimeIndex([]) if a is None else pd.DatetimeIndex(a.index).normalize()
        bi = pd.DatetimeIndex([]) if b is None else pd.DatetimeIndex(b.index).normalize()
        if a is not None and not ai.is_monotonic_increasing:
            unsorted_old.append(sym)
        ca, cb = Counter(ai), Counter(bi)
        for t, n in (ca - cb).items():
            removed[(sym, t.date().isoformat())] += n
        for t, n in (cb - ca).items():
            added[(sym, t.date().isoformat())] += n
        common = ai.unique().intersection(bi.unique())
        if a is None or b is None or not len(common):
            continue
        a2 = a.copy(); a2.index = ai; a2 = a2[~a2.index.duplicated(keep="last")]
        b2 = b.copy(); b2.index = bi
        for col in sorted(set(a2.columns) & set(b2.columns)):
            x = pd.to_numeric(a2.loc[common, col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(b2.loc[common, col], errors="coerce").to_numpy(dtype=float)
            bad = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            for t in common[bad]:
                changed[(sym, t.date().isoformat(), col)] += 1
    return {"removed": removed, "added": added, "changed": changed, "unsorted_in_old": unsorted_old}


def _multiset(rows, width: int) -> Counter:
    return Counter(tuple(r[:width]) for r in rows)


def plan(target: str | Path, staging: str | Path, out: str | Path) -> dict[str, Any]:
    target = Path(target).resolve()
    staging = _not_live(Path(staging), "the staging root")
    out = _not_live(Path(out), "the manifest")
    current = _pickle(target)
    if not current.exists():
        raise SystemExit(f"no current pickle at {current}")
    before = _sha256(current)
    source = _stage(target, staging)
    # The current pickle is audited from a copy on the staging disk: the
    # target is only ever read.
    current_copy = staging / "store_cache.current.pkl"
    shutil.copy2(current, current_copy)
    if _sha256(current_copy) != before:
        raise SystemExit("the copy of the current pickle does not match it")
    old_frames = _frames(current_copy)
    built = _build_in(staging, list(old_frames))
    new = _pickle(staging)
    audited = _audit_in(staging, new, staging / "audit_new.json")
    old_audit = _audit_in(staging, current_copy, staging / "audit_current.json")
    new_frames = _frames(new)
    diff = _row_diff(old_frames, new_frames)
    f_old = old_audit["findings"]
    truncated = sorted(set(old_audit.get("findings_truncated") or []) | set(audited.get("findings_truncated") or []))
    wrong_rows = _multiset(f_old.get("ORPHAN_ON_ARCHIVE_DATE", []) + f_old.get("NO_ARCHIVE_FOR_DATE", []), 2)
    for row in f_old.get("DUPLICATE_DATE", []):
        sym, day = row[0], row[1]
        extra = int(row[2]) if len(row) > 2 else 1
        wrong_rows[(sym, day)] += extra                  # the store held that many rows too many
    gates = {
        "target_unchanged_during_plan": _sha256(current) == before,
        "finding_lists_complete": not truncated,
        "new_store_has_no_contamination": not audited["contamination_kinds"],
        "new_store_is_complete": not audited["incomplete_kinds"],
        "new_store_last_day_is_latest_archive": built["last_day"] == audited["archives"]["last"],
        "new_store_sessions_equal_verified_archives": int(built["sessions"]) == audited["archives"]["verified_sessions"],
        "symbol_set_preserved": set(new_frames) == set(old_frames),
        "symbol_order_preserved": list(new_frames) == list(old_frames),
        "removed_rows_are_exactly_the_contamination": diff["removed"] == wrong_rows,
        "added_rows_are_exactly_the_missing_official_rows": diff["added"] == _multiset(
            f_old.get("MISSING_OFFICIAL_ROW", []), 2),
        "changed_values_are_exactly_the_audited_mismatches": diff["changed"] == _multiset(
            f_old.get("VALUE_MISMATCH", []), 3),
    }
    manifest = {
        "schema": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_root": str(target),
        "target_pickle": str(current),
        "target_sha256": before,
        "source": source,
        "staging_root": str(staging),
        "new_pickle": str(new),
        "new_sha256": _sha256(new),
        "build": built,
        "audit_of_new": {"counts": audited["counts"], "archives": audited["archives"]},
        "audit_of_current": {"counts": old_audit["counts"], "affected_symbols": old_audit["affected_symbols"],
                             "affected_sessions": old_audit["affected_sessions"]},
        "diff": {"removed": sum(diff["removed"].values()), "added": sum(diff["added"].values()),
                 "changed": sum(diff["changed"].values()),
                 "removed_rows": sorted(diff["removed"])[:500], "unsorted_in_old": diff["unsorted_in_old"],
                 "added_symbols": sorted({s for s, _ in diff["added"]})},
        "gates": gates,
        "ready": all(gates.values()),
        "state": "planned",
    }
    Path(out).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


# ── apply / rollback ───────────────────────────────────────────────────────────

def _desk_running() -> list[str]:
    """Live desk processes and services; a failed probe is reported as running (fail closed)."""
    found = []
    for pattern in DESK_PATTERNS:
        try:
            out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=30)
        except Exception as exc:
            return [f"pgrep failed: {type(exc).__name__}"]
        if out.returncode not in (0, 1):
            return [f"pgrep failed: rc={out.returncode}"]
        found += [f"{pattern}:{pid}" for pid in out.stdout.split() if pid.strip() and int(pid) != os.getpid()]
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=30)
        except Exception as exc:
            return found + [f"launchctl failed: {type(exc).__name__}"]
        if out.returncode != 0:
            return found + [f"launchctl failed: rc={out.returncode}"]
        for line in out.stdout.splitlines():
            parts = line.split()
            # Loaded is active: between a crash and its KeepAlive restart the
            # PID column reads "-" (verification QT-05).
            if len(parts) >= 3 and parts[-1] == LAUNCHD_LABEL:
                found.append(f"launchd:{LAUNCHD_LABEL}:{parts[0]}")
    return found


def _check_desk(target: Path, target_is_copy: bool) -> None:
    # Copy mode only when NEITHER the entry apply/rollback replaces (parent
    # resolved, final name kept: os.replace writes that entry) NOR its referent
    # is on a live volume (GPT C7V-05, cycle 7).
    entry = Path(target).parent.resolve() / Path(target).name
    if target_is_copy and not _on_live_volume(entry) and not _on_live_volume(Path(target).resolve()):
        return
    running = _desk_running()
    if running:
        raise SystemExit(f"refusing: the desk is running ({running}); stop it first")


def _held_open(path: Path) -> list[str]:
    try:
        out = subprocess.run(["lsof", "-t", "--", str(path)], capture_output=True, text=True, timeout=30)
    except Exception:
        return ["lsof unavailable"]
    return [p for p in out.stdout.split() if p.strip()]


def _full_fsync(fd: int) -> None:
    """fsync that reaches the medium on macOS (F_FULLFSYNC), else a plain fsync."""
    try:
        import fcntl

        if hasattr(fcntl, "F_FULLFSYNC"):
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            return
    except OSError:
        pass
    os.fsync(fd)


def _write_manifest(path: Path, m: dict) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(m, indent=2, default=str))
        fh.flush()
        _full_fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)                 # the rename itself is durable (GPT QT-06)


def _durable_copy(src: Path, directory: Path, prefix: str, expect_sha: str) -> Path:
    """Copy into a unique, fsynced temp file in ``directory``; verify its sha256 before any rename."""
    fd, name = tempfile.mkstemp(prefix=prefix, dir=str(directory))
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out, 1 << 20)
            out.flush()
            _full_fsync(out.fileno())
        if _sha256(tmp) != expect_sha:
            raise SystemExit(f"refusing: the copy of {src.name} does not match its expected sha256")
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fsync_dir(directory: Path) -> None:
    """Make renames in ``directory`` durable. A filesystem that cannot sync a
    directory (EINVAL/ENOTSUP) is tolerated; any other error is raised, never
    swallowed (GPT QT-06)."""
    import errno

    fd = os.open(str(directory), os.O_RDONLY)
    try:
        try:
            _full_fsync(fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)):
                raise
    finally:
        os.close(fd)


class _MaintenanceLock:
    """The store's publish lock, held from the checks to the swap.

    Every store save takes the same exclusive OS lock (data.bhavcopy_store
    ``PUBLISH_LOCK``), so a save prepared before the repair cannot publish after
    it, and a desk that starts meanwhile defers its save. The OS releases the
    lock if this process dies: it can never go stale (verification QT-05/QT-09).
    """

    def __init__(self, bhav_dir: Path, purpose: str, wait_s: float = 30.0):
        self.path = bhav_dir / PUBLISH_LOCK
        self.purpose = purpose
        self.wait_s = wait_s
        self.fd = None

    def __enter__(self):
        import fcntl

        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise SystemExit("refusing: a store writer holds the publish lock (stop the desk first)")
                time.sleep(0.1)
        self.fd = fd
        return self

    def __exit__(self, *exc):
        import fcntl

        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
        return False


def _verify_source(m: dict) -> None:
    target = Path(m["target_root"])
    now = _source_manifest(target, hashes=False)
    planned = m["source"]
    if set(now["archives"]) != set(planned["archives"]):
        raise SystemExit("refusing: the target's archive set changed since the plan (re-plan)")
    bhav = _logs(target, "bhav")
    for name, meta in now["archives"].items():
        was = planned["archives"][name]
        if (meta["size"], meta["mtime_ns"]) != (was["size"], was["mtime_ns"]):
            raise SystemExit(f"refusing: archive {name} changed since the plan (re-plan)")
        if _sha256(bhav / name) != was["sha256"]:
            raise SystemExit(f"refusing: archive {name} content differs from the plan (re-plan)")
    cal = _calendar(target)
    if cal.exists() != bool(planned["calendar"].get("present")) or (
            cal.exists() and _sha256(cal) != planned["calendar"].get("sha256")):
        raise SystemExit("refusing: the target's runtime calendar changed since the plan (re-plan)")


def apply(manifest_path: str | Path, owner_approval: str, *, target_is_copy: bool = False) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if not str(owner_approval or "").strip():
        raise SystemExit("refusing: apply needs the owner's approval in words (--owner-approval)")
    if not m.get("ready"):
        raise SystemExit(f"refusing: the plan's gates did not all pass: {m.get('gates')}")
    if m.get("state") != "planned":
        raise SystemExit(f"refusing: the manifest is in state {m.get('state')!r}, not 'planned'")
    target, new = Path(m["target_pickle"]), Path(m["new_pickle"])
    _check_desk(target, target_is_copy)
    with _MaintenanceLock(target.parent, "bhav_store_rebuild apply"):
        _check_desk(target, target_is_copy)                       # nothing started meanwhile
        if _sha256(target) != m["target_sha256"]:
            raise SystemExit("refusing: the target pickle changed since the plan (re-plan)")
        if _sha256(new) != m["new_sha256"]:
            raise SystemExit("refusing: the planned pickle changed since the plan")
        _verify_source(m)
        holders = _held_open(target)
        if holders:
            raise SystemExit(f"refusing: the target pickle is held open by {holders} (stop the desk first)")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = target.with_name(f"{target.name}.pre-rebuild-{stamp}")
        btmp = _durable_copy(target, target.parent, ".backup-", m["target_sha256"])
        os.replace(btmp, backup)
        _fsync_dir(target.parent)              # the backup is durable BEFORE anything is replaced
        candidate = _durable_copy(new, target.parent, ".rebuild-", m["new_sha256"])
        # Transaction record BEFORE the swap: an interruption after this point
        # leaves a manifest that can drive the rollback.
        m.update({"state": "swapping", "backup_pickle": str(backup), "owner_approval": owner_approval.strip(),
                  "swap_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        _write_manifest(manifest_path, m)
        os.replace(candidate, target)
        _fsync_dir(target.parent)
        ok = _sha256(target) == m["new_sha256"]
        m.update({"state": "applied" if ok else "swap_unverified", "applied_ok": ok,
                  "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        _write_manifest(manifest_path, m)
    if not ok:
        raise SystemExit("the target does not match the planned pickle after the swap: run rollback")
    return m


def rollback(manifest_path: str | Path, *, target_is_copy: bool = False,
             backup_path: str | Path | None = None) -> dict[str, Any]:
    """Restore the pre-rebuild pickle.

    From a verified apply (``applied``) it runs only while the target still holds
    exactly the applied pickle, so a cache the desk saved later is never
    discarded. From an INTERRUPTED or unverified swap (``swapping``,
    ``swap_unverified``) the target's bytes are unknown by definition, so it
    restores whatever is there, including a missing or damaged file (GPT QT-06).
    ``backup_path`` names another verified copy (the runbook's SSD copy) when
    the recorded backup is gone; it must match the pre-rebuild sha256.
    """
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    recorded = str(backup_path or m.get("backup_pickle") or "").strip()
    backup = Path(recorded) if recorded else None
    if backup is None or not backup.is_file():
        raise SystemExit("nothing to roll back: no backup recorded or found (pass --backup <verified copy>)")
    if _sha256(backup) != m["target_sha256"]:
        raise SystemExit("refusing: the backup does not match the pre-rebuild sha256")
    # An interrupted swap, and an interrupted or unverified ROLLBACK, leave the
    # target's bytes unknown: recovery from a verified backup is always allowed
    # (GPT QT-06, R7, cycle 6).
    interrupted = m.get("state") in {"swapping", "swap_unverified", "rolling_back", "rollback_unverified"}
    if not interrupted and m.get("state") not in {"applied", "rolled_back"}:
        raise SystemExit(f"nothing to roll back: the manifest is in state {m.get('state')!r}")
    target = Path(m["target_pickle"])
    _check_desk(target, target_is_copy)
    with _MaintenanceLock(target.parent, "bhav_store_rebuild rollback"):
        _check_desk(target, target_is_copy)
        current = _sha256(target) if target.is_file() else None
        if current == m["target_sha256"]:
            m.update({"state": "rolled_back", "rollback_ok": True,
                      "rolled_back_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            _write_manifest(manifest_path, m)
            return m                                              # already the pre-rebuild pickle
        if current != m["new_sha256"] and not interrupted:
            raise SystemExit("refusing: the target changed after the apply (the desk saved a newer cache); "
                             "rolling back would discard it")
        holders = _held_open(target) if target.exists() else []
        if holders:
            raise SystemExit(f"refusing: the target pickle is held open by {holders} (stop the desk first)")
        candidate = _durable_copy(backup, target.parent, ".rollback-", m["target_sha256"])
        m.update({"state": "rolling_back", "rollback_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        _write_manifest(manifest_path, m)
        os.replace(candidate, target)
        _fsync_dir(target.parent)
        ok = _sha256(target) == m["target_sha256"]
        m.update({"state": "rolled_back" if ok else "rollback_unverified", "rollback_ok": ok,
                  "rolled_back_from": "interrupted swap" if interrupted else "verified apply",
                  "rolled_back_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        _write_manifest(manifest_path, m)
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
    r.add_argument("--backup", default=None, help="a verified copy to restore when the recorded backup is gone")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        out = plan(args.target, args.staging, args.out)
        print(json.dumps({k: out[k] for k in ("ready", "gates", "build", "diff")}, indent=1, default=str)[:4000])
    elif args.cmd == "apply":
        out = apply(args.manifest, args.owner_approval, target_is_copy=args.target_is_copy)
        print(json.dumps({k: out.get(k) for k in ("applied_ok", "backup_pickle", "new_sha256")}, indent=1))
    else:
        out = rollback(args.manifest, target_is_copy=args.target_is_copy, backup_path=args.backup)
        print(json.dumps({k: out.get(k) for k in ("rollback_ok", "target_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
