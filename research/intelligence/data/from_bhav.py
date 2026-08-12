"""
🔗 Bridge: canonical bhavcopy files (from Historical Data Setup) → an immutable snapshot.

Reads the DDMMYYYY.csv files that `research/momentum_breakout/data_setup` already normalizes,
validates each bar (OHLC sanity, positive prices, no duplicate symbol/date, no future date),
and commits ONE immutable snapshot via the existing `SnapshotStore`. It adds no new ingestion
system — it connects the two existing ones so real user files reach the PAPER_AUTO loop.
"""
from __future__ import annotations

import csv
from datetime import datetime, date
from pathlib import Path

from research.intelligence.data.snapshot_store import SnapshotStore

# canonical bhav columns data_setup writes (with a few tolerant aliases)
_COLS = {"symbol": ("SYMBOL",), "series": ("SERIES",), "open": ("OPEN_PRICE", "OPEN"),
         "high": ("HIGH_PRICE", "HIGH"), "low": ("LOW_PRICE", "LOW"),
         "close": ("CLOSE_PRICE", "CLOSE"), "volume": ("TTL_TRD_QNTY", "VOLUME")}


def _pick(row: dict, keys) -> str:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return ""


def _iso_from_stem(stem: str) -> str | None:
    try:
        return datetime.strptime(stem, "%d%m%Y").date().isoformat()
    except Exception:
        return None


def bhav_to_rows(bhav_dir) -> tuple:
    """Return (equity_rows, report). Rows: (symbol, iso_date, o, h, l, c, vol, series).
    Quarantines defective rows rather than failing the whole file; blocks on systemic issues."""
    bhav_dir = Path(bhav_dir)
    rows, report = [], {"files": 0, "accepted": 0, "quarantined": 0, "duplicates": 0,
                        "future_dated": 0, "warnings": []}
    seen = set()
    today = date.today()
    for f in sorted(bhav_dir.glob("*.csv")):
        iso = _iso_from_stem(f.stem)
        if iso is None:
            report["warnings"].append(f"{f.name}: not named DDMMYYYY.csv — skipped"); continue
        report["files"] += 1
        if date.fromisoformat(iso) > today:
            report["future_dated"] += 1
            report["warnings"].append(f"{f.name}: future-dated session — skipped"); continue
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                sym = str(_pick(r, _COLS["symbol"])).strip().upper()
                series = str(_pick(r, _COLS["series"]) or "EQ").strip().upper()
                try:
                    o, h, l, c = (float(_pick(r, _COLS["open"])), float(_pick(r, _COLS["high"])),
                                  float(_pick(r, _COLS["low"])), float(_pick(r, _COLS["close"])))
                    vol = int(float(_pick(r, _COLS["volume"]) or 0))
                except Exception:
                    report["quarantined"] += 1; continue
                if not sym or min(o, h, l, c) <= 0 or not (l <= o <= h and l <= c <= h):
                    report["quarantined"] += 1; continue          # invalid OHLC → quarantine row
                key = (sym, iso)
                if key in seen:
                    report["duplicates"] += 1; continue           # duplicate symbol/date → reject
                seen.add(key)
                rows.append((sym, iso, o, h, l, c, vol, series))
                report["accepted"] += 1
    return rows, report


def snapshot_from_bhav_dir(bhav_dir, store: SnapshotStore | None = None, *, index_dir=None,
                           activate: bool = False, extra_manifest: dict | None = None):
    """Build (and optionally activate) an immutable snapshot from canonical bhav files.
    Returns (snapshot_id or None, report). Empty/all-defective input ⇒ (None, report) — never a
    fabricated snapshot."""
    store = store or SnapshotStore()
    rows, report = bhav_to_rows(bhav_dir)
    index_rows = _index_rows(index_dir) if index_dir else []
    if not rows:
        report["result"] = "no valid equity rows — nothing committed"
        return None, report
    sid = store.commit_snapshot(rows, index_rows=index_rows,
                                extra_manifest={"source": "bhav_import", **(extra_manifest or {})})
    report["snapshot_id"] = sid
    if activate:
        store.activate_snapshot(sid, actor="user", reason="bhav import")
        report["activated"] = True
    report["result"] = "committed"
    return sid, report


def rows_from_bhav_store(*, max_symbols: int | None = None) -> tuple:
    """Materialise equity rows from the in-memory official bhav store."""
    from data.bhavcopy_runtime import ensure_loaded
    from data import bhavcopy_store as BS

    ensure_loaded(rebuild_from_local=False)
    rows: list = []
    report = {
        "symbols_seen": 0,
        "accepted": 0,
        "quarantined": 0,
        "source": "bhav_store",
    }
    today = date.today()
    for sym, df in BS.iter_raw_frames():
        report["symbols_seen"] += 1
        if max_symbols is not None and report["symbols_seen"] > max_symbols:
            break
        try:
            frame = df.sort_index()
            for ts, bar in frame.iterrows():
                try:
                    d = getattr(ts, "date", lambda: ts)()
                    iso = d.isoformat() if hasattr(d, "isoformat") else str(d)[:10]
                    if date.fromisoformat(iso) > today:
                        report["quarantined"] += 1
                        continue
                    o = float(bar.get("open"))
                    h = float(bar.get("high"))
                    l = float(bar.get("low"))
                    c = float(bar.get("close"))
                    vol = int(float(bar.get("volume") or 0))
                except Exception:
                    report["quarantined"] += 1
                    continue
                if min(o, h, l, c) <= 0 or not (l <= o <= h and l <= c <= h):
                    report["quarantined"] += 1
                    continue
                rows.append((sym, iso, o, h, l, c, vol, "EQ"))
                report["accepted"] += 1
        except Exception:
            report["quarantined"] += 1
            continue
    report["symbols"] = len({r[0] for r in rows})
    return rows, report


def snapshot_from_bhav_store(
    store: SnapshotStore | None = None,
    *,
    activate: bool = True,
    actor: str = "system",
    reason: str = "bhav store certification",
    extra_manifest: dict | None = None,
):
    """Commit (+ optionally activate) a verified snapshot from the live bhav store.

    Used when Kite auth is deferred/unavailable so research still has a pinned
    ``snapshot_id`` sourced from official NSE bhavcopy.
    """
    store = store or SnapshotStore()
    rows, report = rows_from_bhav_store()
    if not rows:
        report["result"] = "bhav store empty — nothing committed"
        report["snapshot_id"] = None
        report["activated"] = False
        return None, report
    sid = store.commit_snapshot(
        rows,
        extra_manifest={
            "source": "bhav_store",
            "has_universe_history": False,
            **(extra_manifest or {}),
        },
    )
    report["snapshot_id"] = sid
    report["result"] = "committed"
    if activate:
        store.activate_snapshot(sid, actor=actor, reason=reason)
        report["activated"] = True
    else:
        report["activated"] = False
    return sid, report


def _index_rows(index_dir) -> list:
    out = []
    for f in sorted(Path(index_dir).glob("*.csv")):
        iso = _iso_from_stem(f.stem)
        if not iso:
            continue
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    out.append(("NIFTY", iso, float(_pick(r, _COLS["open"])),
                                float(_pick(r, _COLS["high"])), float(_pick(r, _COLS["low"])),
                                float(_pick(r, _COLS["close"]))))
                except Exception:
                    continue
    return out
