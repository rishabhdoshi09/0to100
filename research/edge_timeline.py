"""
🗂️ Edge Timeline — a signal's drift HISTORY is itself evidence.

One `assess_drift` call is a snapshot; the timeline is the movie. Persisting
every *state transition* (STABLE→DECAYING→RECOVERING→…) per signal lets the
system answer questions a single read never can — and this longitudinal memory
of how each edge behaves over time is exactly the compounding, hard-to-copy
asset the moat thesis rests on:

    • CYCLICAL — decays and recovers repeatedly (a regime-sensitive edge you
      should size to the cycle, NOT kill on the first dip).
    • DEAD — decayed and never came back over many trades (retire it).
    • DURABLE — long STABLE runs, drift is rare (trust it, lean in).
    • how long recoveries take — so you know how patient to be before quitting.

Design mirrors the rest of the Research OS: a pure, unit-tested classifier
(`_classify_profile`) with a thin, fail-open SQLite layer on a monkeypatchable
path. Events are TRANSITIONS, not samples — we append only when a signal's
status changes, so the table is a compact ledger of turning points, not a firehose.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from research import drift as _drift

_DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "edge_timeline.db"

# A DECAYING signal with no recovery event and this many trades elapsed since
# the decay began is treated as DEAD (the edge did not come back).
_DEAD_TRADES = int(os.getenv("QT_EDGE_DEAD_TRADES", "50") or 50)

_DDL = """
CREATE TABLE IF NOT EXISTS edge_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence TEXT,
    baseline_r REAL,
    recent_r REAL,
    delta_r REAL,
    n INTEGER,
    n_since_change INTEGER,
    variance_ratio REAL,
    risk_profile_changed INTEGER,
    insight TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_events_sig ON edge_events(signal, observed_at);
"""

# Recovery = a return to health after a decay. These statuses END a decay.
_RECOVERY_STATES = {"RECOVERING", "STRENGTHENING", "STABLE"}


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    c = sqlite3.connect(_DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        if stmt.strip():
            c.execute(stmt)
    c.commit()
    return c


# ══════════════════════════════════════════════════════════════════════════════
# Pure classification — the movie, not the snapshot
# ══════════════════════════════════════════════════════════════════════════════

def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return float(s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0)


def _classify_profile(events: list[dict], current_n: int) -> dict:
    """Turn a signal's ordered transition history into a durable character
    judgement. `events` = [{status, n}, …] oldest→newest (n is the stream length
    at each transition, so it grows monotonically); `current_n` is the signal's
    latest outcome count. Pure — no I/O.

    Returns {profile, n_decays, n_recoveries, median_recovery_trades, rationale}.
    Profiles: CYCLICAL | DEAD | DECAYING | RECOVERING | STRENGTHENING |
              DURABLE | EMERGING | STABLE.
    """
    if not events:
        return {"profile": "EMERGING" if current_n else "UNKNOWN",
                "n_decays": 0, "n_recoveries": 0, "median_recovery_trades": 0.0,
                "rationale": "No recorded transitions yet."}

    n_decays = sum(1 for e in events if e["status"] == "DECAYING")
    n_recoveries = sum(1 for e in events if e["status"] == "RECOVERING")

    # pair each decay with the next recovery-state transition → trades to recover
    recovery_gaps: list[float] = []
    open_decay_n: int | None = None
    for e in events:
        st = e["status"]
        if st == "DECAYING":
            open_decay_n = int(e.get("n") or 0)
        elif st in _RECOVERY_STATES and open_decay_n is not None:
            recovery_gaps.append(max(0, int(e.get("n") or 0) - open_decay_n))
            open_decay_n = None
    med_recovery = round(_median(recovery_gaps), 1)

    last = events[-1]["status"]
    last_n = int(events[-1].get("n") or 0)
    since_last = max(0, current_n - last_n)

    # a currently-open decay that has run long with no recovery → DEAD
    decay_unrecovered = open_decay_n is not None and (current_n - open_decay_n) >= _DEAD_TRADES

    if n_decays >= 2 and (n_recoveries >= 1 or recovery_gaps):
        prof = "CYCLICAL"
        rat = (f"Decayed {n_decays}× and recovered {len(recovery_gaps)}× "
               f"(typ. ~{med_recovery:.0f} trades to come back) — a cyclical edge; "
               f"size to the cycle, don't kill on a dip.")
    elif decay_unrecovered or (last == "DECAYING" and since_last >= _DEAD_TRADES):
        prof = "DEAD"
        rat = (f"Decayed and never recovered across ≥{_DEAD_TRADES} trades — "
               f"treat as retired unless it proves otherwise.")
    elif last == "DECAYING":
        prof = "DECAYING"
        rat = "Currently in a confirmed decay — too soon to call it dead; size down."
    elif last == "RECOVERING":
        prof = "RECOVERING"
        rat = (f"Bounced back after a dip"
               + (f" (recoveries typ. ~{med_recovery:.0f} trades)." if recovery_gaps
                  else ".") + " Trust can be rebuilt gradually.")
    elif last == "STRENGTHENING":
        prof = "STRENGTHENING"
        rat = "Edge improving on the latest read — lean in within risk limits."
    elif last == "STABLE":
        if len(events) <= 1 and current_n >= 2 * _DEAD_TRADES:
            prof = "DURABLE"
            rat = f"Stable across {current_n} trades with no drift — a durable edge."
        else:
            prof = "STABLE"
            rat = f"Back to stable; {n_decays} past decay(s) on record."
    else:
        prof = "STABLE"
        rat = "No notable pattern."

    return {"profile": prof, "n_decays": n_decays, "n_recoveries": n_recoveries,
            "median_recovery_trades": med_recovery, "rationale": rat}


# ══════════════════════════════════════════════════════════════════════════════
# I/O — append transitions, read history, profile signals (all fail-open)
# ══════════════════════════════════════════════════════════════════════════════

def _last_status(c: sqlite3.Connection, signal: str) -> str | None:
    row = c.execute("SELECT status FROM edge_events WHERE signal=? "
                    "ORDER BY id DESC LIMIT 1", (signal,)).fetchone()
    return row["status"] if row else None


def record_snapshot(streams: dict[str, list[float]] | None = None,
                    now: str | None = None) -> list[dict]:
    """Assess every signal and append an event ONLY where the status has changed
    since the last recorded one (a transition ledger, not a sample log). Returns
    the list of newly-recorded transitions. Fail-open: any error → []."""
    try:
        streams = streams if streams is not None else _drift._signal_r_streams()
        now = now or time.strftime("%Y-%m-%dT%H:%M:%S")
        recorded: list[dict] = []
        c = _conn()
        try:
            for sig, rs in streams.items():
                if len(rs) < _drift._MIN_N:
                    continue
                d = _drift.assess_drift(rs)
                prev = _last_status(c, sig)
                if prev == d.status:
                    continue                          # no transition → nothing new
                c.execute(
                    "INSERT INTO edge_events (signal, observed_at, status, "
                    "confidence, baseline_r, recent_r, delta_r, n, n_since_change, "
                    "variance_ratio, risk_profile_changed, insight) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sig, now, d.status, d.confidence, d.baseline_r, d.recent_r,
                     d.delta_r, d.n, d.n_since_change, d.variance_ratio,
                     int(d.risk_profile_changed), d.insight))
                recorded.append({"signal": sig, "status": d.status,
                                 "from": prev, "n": d.n})
            c.commit()
        finally:
            c.close()
        return recorded
    except Exception:
        return []


def signal_history(signal: str) -> list[dict]:
    """Ordered transition history for one signal (oldest→newest). Fail-open → []."""
    try:
        c = _conn()
        try:
            rows = c.execute("SELECT * FROM edge_events WHERE signal=? "
                             "ORDER BY id ASC", (signal,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            c.close()
    except Exception:
        return []


def _current_n(signal: str, streams: dict[str, list[float]] | None) -> int:
    if streams is not None:
        return len(streams.get(signal, []))
    try:
        return len(_drift._signal_r_streams().get(signal, []))
    except Exception:
        hist = signal_history(signal)
        return int(hist[-1].get("n") or 0) if hist else 0


def signal_profile(signal: str,
                   streams: dict[str, list[float]] | None = None) -> dict:
    """The durable character of one signal from its recorded history + current
    stream length. Fail-open → an EMERGING/UNKNOWN stub."""
    hist = signal_history(signal)
    prof = _classify_profile(hist, _current_n(signal, streams))
    prof["signal"] = signal
    prof["events"] = len(hist)
    return prof


def timeline_report() -> list[dict]:
    """Every signal that has any recorded history, profiled. Actionable
    characters (CYCLICAL / DEAD / DECAYING / RECOVERING) surface first. Fail-open."""
    try:
        streams = _drift._signal_r_streams()
        c = _conn()
        try:
            sigs = [r["signal"] for r in c.execute(
                "SELECT DISTINCT signal FROM edge_events").fetchall()]
        finally:
            c.close()
    except Exception:
        return []
    out = [signal_profile(s, streams) for s in sigs]
    rank = {"DEAD": 0, "DECAYING": 1, "CYCLICAL": 2, "RECOVERING": 3,
            "STRENGTHENING": 4, "DURABLE": 5, "STABLE": 6, "EMERGING": 7}
    return sorted(out, key=lambda p: rank.get(p["profile"], 9))


def timeline_directives(max_items: int = 3) -> list[dict]:
    """Brain-ready directives from durable signal character (distinct from the
    momentary drift read): retire the dead, respect the cyclical, don't panic on
    a signal that always comes back. Fail-open → []."""
    try:
        from scan.unified_scanner import SIGNAL_META
    except Exception:
        SIGNAL_META = {}
    dirs: list[dict] = []
    for p in timeline_report():
        prof = p["profile"]
        if prof not in ("DEAD", "CYCLICAL"):
            continue                                  # the two durable, actionable calls
        label = SIGNAL_META.get(p["signal"], (p["signal"],))[0]
        if prof == "DEAD":
            dirs.append({"severity": "warn",
                         "text": f"⚰️ {label} — {p['rationale']}"})
        elif prof == "CYCLICAL":
            dirs.append({"severity": "info",
                         "text": f"🔁 {label} — {p['rationale']}"})
        if len(dirs) >= max_items:
            break
    return dirs
