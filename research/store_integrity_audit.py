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
    if str(p).startswith("/Volumes/"):
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


def _compare(store: dict, payload: dict, pkl: Path, bs) -> dict[str, Any]:
    import numpy as np
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
    official = pd.concat(parts, ignore_index=True)
    official["date"] = pd.to_datetime(official["date"]).dt.normalize()
    official = official.drop_duplicates(["symbol", "date"], keep="last")
    official_rows = {s: g.set_index("date") for s, g in official.groupby("symbol", sort=False)}
    listed_on = official.groupby("date")["symbol"].apply(set).to_dict()

    findings: dict[str, list] = defaultdict(list)
    per_symbol: Counter = Counter()
    per_date: Counter = Counter()
    max_rel: dict[str, float] = {}
    checked_rows = 0
    for sym, frame in store.items():
        idx = pd.DatetimeIndex(frame.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        frame = frame.copy()
        frame.index = idx.normalize()
        ref = official_rows.get(sym)
        for ts in frame.index[frame.index.duplicated(keep=False)].unique():
            findings["DUPLICATE_DATE"].append((sym, ts.date().isoformat()))
        frame = frame[~frame.index.duplicated(keep="last")]
        checked_rows += len(frame)
        for ts in frame.index:
            if ts not in archive_set:
                findings["NO_ARCHIVE_FOR_DATE"].append((sym, ts.date().isoformat()))
                per_symbol[sym] += 1
                per_date[ts.date().isoformat()] += 1
            elif ref is None or ts not in ref.index:
                findings["ORPHAN_ON_ARCHIVE_DATE"].append((sym, ts.date().isoformat()))
                per_symbol[sym] += 1
                per_date[ts.date().isoformat()] += 1
        if ref is None:
            continue
        common = frame.index.intersection(ref.index)
        for col in VALUE_COLUMNS:
            if col not in frame.columns or col not in ref.columns:
                continue
            a = pd.to_numeric(frame.loc[common, col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(ref.loc[common, col], errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(a) & np.isnan(b)
            differ = ~both_nan & ~np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=False)
            if differ.any():
                for ts, va, vb in zip(common[differ], a[differ], b[differ]):
                    findings["VALUE_MISMATCH"].append((sym, ts.date().isoformat(), col, va, vb))
                    per_symbol[sym] += 1
                    per_date[ts.date().isoformat()] += 1
                    if col == "close" and vb == vb and vb != 0 and va == va:
                        max_rel[f"{sym}:{ts.date().isoformat()}"] = abs(va / vb - 1.0)
        missing = ref.index.difference(frame.index)
        for ts in missing:
            findings["MISSING_OFFICIAL_ROW"].append((sym, ts.date().isoformat()))

    for sym, g in official_rows.items():
        if sym not in store:
            kind = "BELOW_BUILD_FLOOR" if len(g) < BUILD_FLOOR else "MISSING_SYMBOL"
            findings[kind].append((sym, len(g)))

    contamination = [k for k in ("ORPHAN_ON_ARCHIVE_DATE", "NO_ARCHIVE_FOR_DATE", "VALUE_MISMATCH",
                                 "DUPLICATE_DATE") if findings.get(k)]
    affected = sorted({row[0] for k in contamination for row in findings[k]})
    sessions = sorted({row[1] for k in contamination for row in findings[k]})
    return {
        "pickle": {"path": str(pkl), "sha256": _sha256(pkl), "last_day": str(payload.get("last_day")),
                   "sessions": payload.get("sessions"), "symbols": len(store), "rows_checked": checked_rows},
        "archives": {"verified_sessions": len(archive_days),
                     "first": archive_days[0].isoformat() if archive_days else None,
                     "last": archive_days[-1].isoformat() if archive_days else None,
                     "unreadable": unreadable, "official_rows": int(len(official))},
        "counts": {k: len(v) for k, v in sorted(findings.items())},
        "contamination_kinds": contamination,
        "affected_symbols": affected,
        "affected_sessions": sessions,
        "rows_per_session": dict(sorted(per_date.items())),
        "rows_per_symbol_top": per_symbol.most_common(25),
        "close_mismatch_max_rel": max(max_rel.values()) if max_rel else None,
        "findings": {k: v[:2000] for k, v in sorted(findings.items())},
        "listed_on_sessions": len(listed_on),
        "method": "every store row vs data.bhavcopy_store._read_day of the verified archives (_dates_on_disk)",
        "read_only": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--pickle", default=None, help="audit this pickle instead of the copy's own")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = audit(a.root, pickle_path=a.pickle)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("pickle", "archives", "counts", "contamination_kinds",
                                          "affected_sessions", "close_mismatch_max_rel")},
                     indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
