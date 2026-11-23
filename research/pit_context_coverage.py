"""Coverage and sanity of the point-in-time market context on a runtime COPY.

Re-measures what product.pit_features would freeze for every DECISION
observation already in a feature-store copy: how many get delivery_pct, vix,
index_trend and breadth_pct_above_50dma, and whether the values are plausible.
The stored observations are never changed (the database is opened read-only).

Never run against the live runtime: any path under ``/Volumes`` is refused.

The copy's store pickle and index pickle are read directly, never through
data.bhavcopy_store._load_pkl, which deletes a pickle it cannot read and
rejects one whose daily archives are absent. The copy holds no daily archives,
so the "official session" gate uses the store's own session list (every
session in the pickle came from an official bhavcopy archive). That is stated
in the report as a limit.

    nice -n 19 venv/bin/python -m research.pit_context_coverage \\
        --root <runtime copy> --features-db <feature_store.db copy> --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core.live_volume import on_live_volume

_INDEX_NAMES = {"^NSEI": ("NIFTY 50", "Nifty 50"), "^INDIAVIX": ("INDIA VIX", "India VIX")}


def _refuse_live(path: str | Path) -> Path:
    if on_live_volume(path):                           # as spelled, before resolving (GPT C8.5-04)
        raise SystemExit(f"refusing a live runtime path: {path}")
    p = Path(path).resolve()
    if on_live_volume(p):
        raise SystemExit(f"refusing a live runtime path: {p}")
    return p


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    vals = sorted(values)
    return {"n": len(vals), "min": round(vals[0], 3), "p10": round(vals[len(vals) // 10], 3),
            "median": round(statistics.median(vals), 3), "p90": round(vals[(len(vals) * 9) // 10], 3),
            "max": round(vals[-1], 3)}


def run(root: str | Path, features_db: str | Path) -> dict[str, Any]:
    """The report for the runtime COPY at ``root`` (QT_RUNTIME_ROOT is pointed at it, then restored)."""
    root = _refuse_live(root)
    features_db = _refuse_live(features_db)
    previous = os.environ.get("QT_RUNTIME_ROOT")
    os.environ["QT_RUNTIME_ROOT"] = str(root)
    try:
        return _run(features_db)
    finally:
        if previous is None:
            os.environ.pop("QT_RUNTIME_ROOT", None)
        else:
            os.environ["QT_RUNTIME_ROOT"] = previous


def _run(features_db: Path) -> dict[str, Any]:
    from core.runtime_paths import logs_path
    from data.corporate_actions import adjust_frame, load_events
    from product import pit_features as PF

    with open(logs_path("bhav", "store_cache.pkl"), "rb") as fh:
        payload = pickle.load(fh)
    store = dict(payload.get("store") or {})
    events = load_events() or {}            # read-only: the copy's CA table
    with open(logs_path("indices", "index_store.pkl"), "rb") as fh:
        index_store = dict((pickle.load(fh) or {}).get("store") or {})

    def ohlcv(sym: str):
        df = store.get(str(sym).upper())
        if df is None:
            return None
        ev = events.get(str(sym).upper())
        return adjust_frame(df.copy(), ev, copy=False) if ev else df.copy()

    def index_fn(ticker: str, as_of: str):
        for name in _INDEX_NAMES.get(ticker, ()):
            if name in index_store:
                return index_store[name]
        return None

    all_days = sorted({ts.date() for df in store.values() for ts in df.index})
    symbols = list(store)
    days_set = sorted(set(all_days))
    # One identity per source for the whole run: the breadth cache is keyed on it.
    symbols_fn, archive_fn = (lambda: symbols), (lambda: days_set)

    con = sqlite3.connect(f"file:{features_db}?mode=ro", uri=True)
    rows = con.execute("SELECT symbol, ts, features FROM observations WHERE kind='DECISION'").fetchall()
    con.close()

    PF.reset_breadth_cache()
    context_by_session: dict[Any, dict[str, Any]] = {}
    filled = Counter()
    by_year: dict[str, Counter] = defaultdict(Counter)
    values: dict[str, list[float]] = defaultdict(list)
    trend_breadth: dict[str, list[float]] = defaultdict(list)
    stored_present = Counter()
    no_session = 0
    lag_days: list[int] = []
    for symbol, ts, feats in rows:
        stored = json.loads(feats)
        for name in PF.CONTEXT_PRODUCED:
            v = stored.get(name)
            if v is not None and not (name == "index_trend" and str(v).upper() == "UNKNOWN"):
                stored_present[name] += 1
        year = str(ts)[:4]
        by_year[year]["observations"] += 1
        session = PF.published_session(ts, archive_fn=archive_fn)
        if session is None:
            no_session += 1
            continue
        cut = PF.cutoff_session(ts)
        if cut is not None:
            lag_days.append((cut - session).days)
        if session not in context_by_session:
            ctx = PF.market_context(session, index_fn=index_fn)
            breadth = PF.breadth_pct_above_50dma(session, symbols_fn=symbols_fn, ohlcv_fn=ohlcv,
                                                 archive_fn=archive_fn)
            if breadth is not None:
                ctx["breadth_pct_above_50dma"] = breadth
            context_by_session[session] = ctx
        ctx = dict(context_by_session[session])
        deliv = PF.delivery_on(store.get(str(symbol or "").upper()), session)   # delivery is CA-invariant
        if deliv is not None:
            ctx["delivery_pct"] = deliv
        for name in PF.CONTEXT_PRODUCED:
            if ctx.get(name) is not None:
                filled[name] += 1
                by_year[year][name] += 1
                if name != "index_trend":
                    values[name].append(float(ctx[name]))
        if ctx.get("index_trend") and ctx.get("breadth_pct_above_50dma") is not None:
            trend_breadth[ctx["index_trend"]].append(float(ctx["breadth_pct_above_50dma"]))

    n = len(rows)
    trends = Counter(c.get("index_trend") for c in context_by_session.values() if c.get("index_trend"))
    return {
        "observations": n,
        "distinct_source_sessions": len(context_by_session),
        "no_published_session": no_session,
        "cutoff_minus_source_days": dict(Counter(lag_days)),
        "stored_present_before": dict(stored_present),
        "would_fill": {k: {"n": filled[k], "pct": round(100.0 * filled[k] / n, 2) if n else None}
                       for k in PF.CONTEXT_PRODUCED},
        "by_year": {y: dict(c) for y, c in sorted(by_year.items())},
        "value_summary": {k: _summary(v) for k, v in values.items()},
        "index_trend_sessions": dict(trends),
        "breadth_by_index_trend": {k: _summary(v) for k, v in trend_breadth.items()},
        "store": {"symbols": len(store), "sessions": len(all_days), "first": all_days[0] if all_days else None,
                  "last": all_days[-1] if all_days else None, "ca_event_symbols": len(events)},
        "limits": [
            "verified-archive gate = the store pickle's own session list (the copy has no daily archives)",
            "the copy dates from its snapshot; observations after it get no context",
            "read-only: no observation was changed",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--features-db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = run(a.root, a.features_db)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("observations", "distinct_source_sessions", "would_fill",
                                          "value_summary", "breadth_by_index_trend")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
