"""One-time, auditable NSE bhavcopy quarantine/cache repair.

Default is READ-ONLY audit.  --apply requires the launchd host AND API to be
stopped, preserves original bad CSV/pickle/artifacts in a dated quarantine, and
removes fake trading-day bars from the saved canonical pickle.  It never touches
a live broker or PAPER journal.  Run only after code review.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

from data.bhavcopy_integrity import validate_archive_path
from research.intelligence.data.nse_calendar import is_session, load_holidays
from core.runtime_paths import logs_dir


def _host_is_stopped() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("--apply is restricted to the stopped canonical macOS host")
    proc = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/com.quantterm.desk"],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode == 0 and ("state = running" in proc.stdout or "pid = " in proc.stdout):
        raise RuntimeError("Stop com.quantterm.desk before repairing persistent data")
    for port in (8765, 8766, 5173):
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Local QuantTerm service still listens on {port}")


def audit(bhav_dir: Path) -> tuple[list[tuple[Path, date, str]], list[date]]:
    bad = []
    good = []
    holidays = load_holidays()
    for path in sorted(bhav_dir.glob("*.csv")):
        try:
            day = datetime.strptime(path.stem, "%d%m%Y").date()
        except ValueError:
            continue
        if not is_session(day, holidays):
            bad.append((path, day, "EXCHANGE_HOLIDAY_OR_WEEKEND"))
            continue
        valid, reason = validate_archive_path(path, day, full=True)
        if valid:
            good.append(day)
        else:
            bad.append((path, day, reason))
    return bad, good


def _repair_pickle(original: Path, bad_dates: set[date], good: list[date]) -> tuple[dict, int, date]:
    import pandas as pd

    with original.open("rb") as fh:
        data = pickle.load(fh)  # trusted, user-owned QuantTerm cache only
    store = data.get("store")
    if not isinstance(store, dict) or not store:
        raise RuntimeError("Canonical pickle is invalid; refusing a destructive repair")
    removed_dates = set()
    removed_rows = 0
    blocked = pd.to_datetime(sorted(bad_dates))
    newest = None

    for sym, frame in list(store.items()):
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            raise RuntimeError(f"Unexpected cached frame for {sym}; do not rewrite")
        if len(frame) == 0:
            continue
        normalized = frame.index.normalize()
        mask = normalized.isin(blocked)
        if mask.any():
            removed_rows += int(mask.sum())
            removed_dates.update(ts.date() for ts in normalized[mask])
            frame = frame.loc[~mask].copy()
            store[sym] = frame
        if len(frame):
            last = frame.index.max().date()
            if newest is None or last > newest:
                newest = last

    if newest is None or not good or newest > max(good):
        raise RuntimeError("Repaired cache still claims a session beyond verified archives")
    data["last_day"] = newest
    data["sessions"] = max(0, int(data.get("sessions") or 0) - len(removed_dates))
    return data, removed_rows, newest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write repair (host MUST be stopped)")
    args = parser.parse_args()

    root = logs_dir()
    bhav = root / "bhav"
    cache = bhav / "store_cache.pkl"
    bad, good = audit(bhav)
    print("BHAV_DIR:", bhav)
    print("Verified sessions:", len(good), "latest:", max(good) if good else "NONE")
    print("Rejected archives:", len(bad))
    for path, day, reason in bad[:80]:
        print("REJECT", path.name, day, reason)
    if len(bad) > 80:
        print("... more rejected files:", len(bad) - 80)

    # A prior operator might already have removed the bad file while the
    # pickle and trade projections still claim its synthetic session.
    dates_to_remove = {d for _, d, _ in bad}
    scan = root / "product" / "latest_momentum_scan.json"
    try:
        saved = json.loads(scan.read_text(encoding="utf-8"))
        claimed = date.fromisoformat(str(saved.get("as_of_session") or "")[:10])
        if not is_session(claimed, load_holidays()):
            dates_to_remove.add(claimed)
            print("REJECT saved scan non-session:", claimed)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    if cache.exists():
        # An old pickle can itself advertise a date not supported by verified CSVs.
        with cache.open("rb") as fh:
            header = pickle.load(fh)
        cached = header.get("last_day")
        if isinstance(cached, date) and (not good or cached > max(good)
                                         or not is_session(cached, load_holidays())):
            dates_to_remove.add(cached)
            print("REJECT cached session:", cached)

    if not args.apply:
        print("READ ONLY. Pass --apply ONLY after the QuantTerm host is stopped.")
        return 0

    _host_is_stopped()
    if not dates_to_remove:
        print("Nothing to repair; no files changed.")
        return 0
    if not good:
        raise RuntimeError("No verified history remains; refusing to modify data")

    amended = None
    if cache.exists():
        amended, n_rows, new_last = _repair_pickle(cache, dates_to_remove, good)
        print("Prepared cache: removed rows:", n_rows, "new last day:", new_last)

    dest = bhav / "quarantine" / ("date_integrity_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    dest.mkdir(parents=True, exist_ok=False)

    if cache.exists():
        shutil.copy2(cache, dest / cache.name)
    for path, _day, _reason in bad:
        shutil.move(str(path), str(dest / path.name))

    if amended is not None:
        temp = cache.with_suffix(".pkl.date_repair.tmp")
        try:
            with temp.open("wb") as fh:
                pickle.dump(amended, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp, cache)
        finally:
            temp.unlink(missing_ok=True)

    # Remove decision products tied to the contaminated scan while preserving
    # originals for investigation. New scan must republish these atomically.
    for filename in (
        "product/latest_momentum_scan.json",
        "product/best_setups.json",
        "product/latest_recommendations.json",
        "product/startup_trade_discovery.json",
        "product/latest_scan_audit.json",
        "product/latest_long_term_scan.json",
    ):
        src = root / filename
        if src.exists():
            dst = dest / src.name
            if dst.exists():
                dst = dest / src.name.replace(".json", ".backup.json")
            shutil.move(str(src), str(dst))
    (dest / "repair_manifest.json").write_text(
        json.dumps({"removed_dates": sorted(map(str, dates_to_remove)),
                    "rejected": [{"file": p.name, "reason": reason}
                                 for p, _, reason in bad],
                    "verified_latest": str(max(good))}, indent=2),
        encoding="utf-8",
    )
    print("REPAIR COMPLETE. Originals backed up at:", dest)
    print("Restart ONLY after reviewing the backup. Re-run NSE scan before entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
