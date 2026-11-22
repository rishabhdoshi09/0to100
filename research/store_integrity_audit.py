"""Does the saved OHLCV store equal its verified official archives? (read-only, on a COPY)

The bhavcopy store (``logs/bhav/store_cache.pkl``) is a cache that should be
exactly the official daily archives parsed by ``data.bhavcopy_store._read_day``.
Before 22 Nov 2026 (GPT cross-review RS-02) ``_incremental_append`` updated only
the symbols an archive lists, so a live intraday overlay row of a symbol that
archive did NOT list was kept and persisted as history. This audit compares
every (symbol, session) row of a store pickle with the archives and classifies
each difference:

* ``ORPHAN_ON_ARCHIVE_DATE``: the store has a row for a session whose archive is
  verified but does not list that symbol (EQ series). It can only be a live
  overlay (or a row from an older archive version): not official history.
* ``NO_ARCHIVE_FOR_DATE``: the store has a row for a session with no verified
  archive on disk at all (a missed archive, so the row came from an overlay).
* ``VALUE_MISMATCH``: both have the row and a price, volume or delivery value
  differs.
* ``MISSING_OFFICIAL_ROW``: an archive lists the symbol but the store has no
  row. ``BELOW_BUILD_FLOOR`` marks a symbol absent from the store entirely with
  fewer than 30 archive rows (``_full_build`` keeps names with 30+ rows only):
  expected, not contamination.

Never run against the live runtime: any path under ``/Volumes`` is refused. The
copy's pickle is read directly (``_load_pkl`` deletes a pickle it cannot read).

    nice -n 19 venv/bin/python -m research.store_integrity_audit --root <runtime copy> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

#: Columns compared, as the store keeps them.
VALUE_COLUMNS = ("open", "high", "low", "close", "volume", "deliv_per")
BUILD_FLOOR = 30           # data.bhavcopy_store._full_build: len(g) >= 30


def _refuse_live(path: str | Path) -> Path:
    p = Path(path).resolve()
    if str(p).lower().startswith("/volumes/"):         # case-insensitive volumes (cycle-7 review)
        raise SystemExit(f"refusing a live runtime path: {p}")
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(root: str | Path, *, pickle_path: str | Path | None = None) -> dict[str, Any]:
    """The report for the runtime COPY at ``root`` (QT_RUNTIME_ROOT is pointed at it, then restored)."""
    root = _refuse_live(root)
    previous = os.environ.get("QT_RUNTIME_ROOT")
    os.environ["QT_RUNTIME_ROOT"] = str(root)
    try:
        return _audit(Path(pickle_path).resolve() if pickle_path else None)
    finally:
        if previous is None:
            os.environ.pop("QT_RUNTIME_ROOT", None)
        else:
            os.environ["QT_RUNTIME_ROOT"] = previous


def _audit(pickle_path: Path | None) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from core.runtime_paths import logs_path
    from data import bhavcopy_store as bs

    pkl = pickle_path or logs_path("bhav", "store_cache.pkl")
    _refuse_live(pkl)
    with open(pkl, "rb") as fh:
        payload = pickle.load(fh)
    store: dict = dict(payload.get("store") or {})

    # data.bhavcopy_store binds its archive directory when first imported; bind
    # it to THIS copy for the audit (and drop the listing caches), then restore,
    # so an audit inside a process that imported the store earlier lists the
    # right archives.
    bhav_dir = _refuse_live(logs_path("bhav"))
    saved = (bs._BHAV_DIR, bs._SESSION_LISTING, bs._DATES_CACHE)
    bs._BHAV_DIR, bs._SESSION_LISTING, bs._DATES_CACHE = bhav_dir, None, None
    try:
        return _compare(store, payload, pkl, bs)
    finally:
        bs._BHAV_DIR, bs._SESSION_LISTING, bs._DATES_CACHE = saved


#: Kinds that mean the store holds something the archives do not (wrong data).
CONTAMINATION = ("ORPHAN_ON_ARCHIVE_DATE", "NO_ARCHIVE_FOR_DATE", "VALUE_MISMATCH", "DUPLICATE_DATE",
                 "UNSORTED", "NON_DATETIME_INDEX", "TZ_AWARE_INDEX", "NON_MIDNIGHT_INDEX", "MISSING_COLUMN",
                 "PERSISTED_OVERLAY_MARKER", "PENDING_OVERLAPS_STORE")
#: Kinds that mean the store lacks official history it should hold (served).
INCOMPLETE = ("MISSING_OFFICIAL_ROW", "MISSING_SYMBOL", "ELIGIBLE_NAME_ONLY_PENDING", "BELOW_FLOOR_NOT_PENDING")
#: Findings kept per kind in the report; counts are always exact. A truncated
#: list is flagged, and a gate that needs the whole list must refuse.
FINDINGS_CAP = 200_000


def _check_frame(sym: str, frame, ref, archive_set, findings, prefix: str, per_symbol, per_date) -> int:
    """Classify one frame (served or pending) against its official rows; returns rows checked."""
    import numpy as np
    import pandas as pd

    def add(kind, *row):
        findings[prefix + kind].append((sym, *row))
        per_symbol[sym] += 1
        if row and isinstance(row[0], str):
            per_date[row[0]] += 1

    if not isinstance(frame.index, pd.DatetimeIndex):
        # Readers call .normalize() on the stored index: string or object
        # labels serve no history even when they parse (verification QT-10).
        add("NON_DATETIME_INDEX", type(frame.index).__name__)
    raw = pd.DatetimeIndex(frame.index)
    if raw.tz is not None:
        add("TZ_AWARE_INDEX", str(raw.tz))
        raw = raw.tz_convert("Asia/Kolkata").tz_localize(None)
    if len(raw) and bool((raw != raw.normalize()).any()):
        add("NON_MIDNIGHT_INDEX", str(raw[raw != raw.normalize()][0]))
    if not raw.is_monotonic_increasing:
        add("UNSORTED", int((raw[1:] < raw[:-1]).sum()))
    marker = getattr(frame, "attrs", {}).get("quantterm_live_overlay_date")
    if marker:
        add("PERSISTED_OVERLAY_MARKER", str(marker)[:10])
    frame = frame.copy()
    frame.index = raw.normalize()
    counts = frame.index.value_counts()
    for ts, n in counts[counts > 1].sort_index().items():
        add("DUPLICATE_DATE", ts.date().isoformat(), int(n) - 1)     # rows too many on that date
    frame = frame[~frame.index.duplicated(keep="last")]
    for ts in frame.index:
        if ts not in archive_set:
            add("NO_ARCHIVE_FOR_DATE", ts.date().isoformat())
        elif ref is None or ts not in ref.index:
            add("ORPHAN_ON_ARCHIVE_DATE", ts.date().isoformat())
    if ref is None:
        return len(frame)
    for col in VALUE_COLUMNS:
        if col not in ref.columns:
            continue
        if col not in frame.columns:
            if pd.to_numeric(ref[col], errors="coerce").notna().any():
                add("MISSING_COLUMN", col)
            continue
        common = frame.index.intersection(ref.index)
        a = pd.to_numeric(frame.loc[common, col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(ref.loc[common, col], errors="coerce").to_numpy(dtype=float)
        differ = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        for ts, va, vb in zip(common[differ], a[differ], b[differ]):
            add("VALUE_MISMATCH", ts.date().isoformat(), col, va, vb)
    for ts in ref.index.difference(frame.index):
        add("MISSING_OFFICIAL_ROW", ts.date().isoformat())
    return len(frame)


def _compare(store: dict, payload: dict, pkl: Path, bs) -> dict[str, Any]:
    import pandas as pd

    archive_days = list(bs._dates_on_disk())
    archive_set = {pd.Timestamp(d) for d in archive_days}

    # Long official table, parsed exactly as the store parses it.
    parts, unreadable = [], []
    for d in archive_days:
        f = bs._read_day(d)
        if f is None:
            unreadable.append(d.isoformat())
            continue
        parts.append(f)
    official = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["symbol", "date"])
    official["date"] = pd.to_datetime(official["date"]).dt.normalize()
    official = official.drop_duplicates(["symbol", "date"], keep="last")
    official_rows = {s: g.set_index("date") for s, g in official.groupby("symbol", sort=False)}

    findings: dict[str, list] = defaultdict(list)
    per_symbol: Counter = Counter()
    per_date: Counter = Counter()
    checked_rows = 0
    for sym, frame in store.items():
        checked_rows += _check_frame(sym, frame, official_rows.get(sym), archive_set, findings, "",
                                     per_symbol, per_date)

    has_pending = "pending" in payload
    pending: dict = dict(payload.get("pending") or {})
    for sym, frame in pending.items():
        if sym in store:
            findings["PENDING_OVERLAPS_STORE"].append((sym,))
        _check_frame(sym, frame, official_rows.get(sym), archive_set, findings, "PENDING_",
                     per_symbol, per_date)

    for sym, g in official_rows.items():
        if sym in store:
            continue
        if len(g) >= BUILD_FLOOR:
            # Only the served map is read (get_ohlcv): an eligible name kept
            # only in pending serves no history (verification QT-10).
            findings["ELIGIBLE_NAME_ONLY_PENDING" if sym in pending else "MISSING_SYMBOL"].append((sym, len(g)))
        elif sym in pending:
            continue
        elif has_pending:
            findings["BELOW_FLOOR_NOT_PENDING"].append((sym, len(g)))
        else:
            findings["BELOW_BUILD_FLOOR"].append((sym, len(g)))

    def kinds(names):
        return [k for k in sorted(findings) if findings[k] and (k in names or k.removeprefix("PENDING_") in names)]

    contamination = kinds(CONTAMINATION)
    incomplete = kinds(INCOMPLETE)
    affected = sorted({row[0] for k in contamination for row in findings[k]})
    sessions = sorted({row[1] for k in contamination for row in findings[k]
                       if len(row) > 1 and isinstance(row[1], str) and row[1][:4].isdigit()})
    return {
        "pickle": {"path": str(pkl), "sha256": _sha256(pkl), "last_day": str(payload.get("last_day")),
                   "sessions": payload.get("sessions"), "symbols": len(store), "pending": len(pending),
                   "has_pending_map": has_pending,
                   "rows_checked": checked_rows},
        "archives": {"verified_sessions": len(archive_days),
                     "first": archive_days[0].isoformat() if archive_days else None,
                     "last": archive_days[-1].isoformat() if archive_days else None,
                     "unreadable": unreadable, "official_rows": int(len(official))},
        "counts": {k: len(v) for k, v in sorted(findings.items())},
        "contamination_kinds": contamination,
        "incomplete_kinds": incomplete,
        "affected_symbols": affected,
        "affected_sessions": sessions,
        "rows_per_session": dict(sorted(per_date.items())),
        "rows_per_symbol_top": per_symbol.most_common(25),
        "findings": {k: v[:FINDINGS_CAP] for k, v in sorted(findings.items())},
        "findings_truncated": sorted(k for k, v in findings.items() if len(v) > FINDINGS_CAP),
        "method": "every served and pending row vs data.bhavcopy_store._read_day of the verified archives "
                  "(_dates_on_disk), plus raw index structure and live markers",
        "read_only": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--pickle", default=None, help="audit this pickle instead of the copy's own")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _refuse_live(a.out)                # the report is written to a copy, never the live runtime (C7V-05)
    rep = audit(a.root, pickle_path=a.pickle)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("pickle", "archives", "counts", "contamination_kinds",
                                          "incomplete_kinds", "affected_sessions")},
                     indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
