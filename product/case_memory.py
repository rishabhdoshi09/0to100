"""Case memory — every market idea QuantTerm shows becomes a remembered Case.

A Case is not a new buy list. It is the object the desk already had in pieces:

  idea → why now → what proves it wrong → similar settled cases → did it pay?

Scanner / reco cards OPEN cases.
Bhavcopy (same first-touch rule as signal_outcome_tracker) SETTLES them at night.
live_edge + this ledger REMEMBER the setup type.

Honesty:
  • n < 30 → Promising, but not proven. No “made money” claim.
  • n = 0 → not remembered yet. Never invent 18 similar cases.
  • After-cost language only when live_edge already netted costs AND n ≥ 30.
  • Places no orders.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
CASES_DB = ROOT / "logs" / "product" / "cases" / "cases.db"
PROVEN_N = 30
_HORIZON_SESSIONS = int(os.getenv("QT_OUTCOME_HORIZON", "15") or 15)

_DDL = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    setup TEXT NOT NULL,
    category_id TEXT,
    idea TEXT,
    why_now TEXT,
    invalidation TEXT,
    opened_at TEXT NOT NULL,
    entry REAL,
    stop REAL,
    target REAL,
    regime TEXT,
    source TEXT,
    settled_at TEXT,
    outcome_pct REAL,
    worked INTEGER
);
CREATE INDEX IF NOT EXISTS idx_cases_setup ON cases(setup);
CREATE INDEX IF NOT EXISTS idx_cases_symbol ON cases(symbol);
CREATE INDEX IF NOT EXISTS idx_cases_settled ON cases(settled_at);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    try:
        from core.market_clock import now_ist
        return now_ist().strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _conn(path: Path | None = None) -> sqlite3.Connection:
    db = Path(path or CASES_DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    return conn


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def primary_setup(row: Mapping[str, Any] | None, category_id: str = "") -> str:
    """One setup name per case — first real signal, else the bucket."""
    for item in (row or {}).get("signals") or []:
        key = str(item or "").strip().upper()
        if key:
            return key
    for tag in (row or {}).get("evidence_tags") or []:
        key = str(tag or "").strip().upper()
        if key and "_" in key:
            return key
    cat = str(category_id or (row or {}).get("category_id") or "").strip()
    return cat.upper() or "UNTYPED"


def _plain_setup(setup: str) -> str:
    return str(setup or "setup").replace("_", " ").strip().lower()


def _idea_line(symbol: str, setup: str, setup_label: str = "") -> str:
    label = str(setup_label or "").strip()
    if "breakout" in (label + " " + setup).lower():
        return f"{symbol} looks like a strong breakout."
    if "recover" in (label + " " + setup).lower() or "DOUBLE_BOTTOM" in setup:
        return f"{symbol} looks like a recovery setup."
    if setup_label:
        return f"{symbol} is a {label} case."
    return f"{symbol} is a {_plain_setup(setup)} case."


def _regime_plain(name: str) -> str:
    u = str(name or "").upper()
    if any(tok in u for tok in ("HEALTHY", "BULL", "RISK_ON", "STRONG", "RISKON")):
        return "strong markets"
    if any(tok in u for tok in ("NARROW", "BEAR", "RISK_OFF", "WEAK", "RISKOFF")):
        return "weak markets"
    return str(name or "").replace("_", " ").strip().lower() or "unlabelled tape"


def _agg_rows(pairs: Sequence[tuple[float, int]]) -> dict[str, Any]:
    n = len(pairs)
    if not n:
        return {"n": 0, "wins": 0, "win_rate": None, "expectancy_r": None}
    wins = sum(1 for _, w in pairs if int(w) == 1)
    rs = [r for r, _ in pairs]
    return {
        "n": n,
        "wins": wins,
        "win_rate": round(wins / n * 100.0, 1),
        "expectancy_r": round(sum(rs) / n, 3) if rs else None,
    }


def _desk_pairs(setup: str, *, db_path: Path | None = None) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    try:
        conn = _conn(db_path)
        rows = conn.execute(
            """SELECT entry, stop, outcome_pct, worked
               FROM cases WHERE setup=? AND worked IS NOT NULL
                 AND outcome_pct IS NOT NULL AND entry > 0 AND stop > 0""",
            (setup,),
        ).fetchall()
        conn.close()
    except Exception:
        return out
    for row in rows:
        entry, stop = _f(row["entry"]), _f(row["stop"])
        if entry <= stop:
            continue
        risk = (entry - stop) / entry
        if risk <= 0:
            continue
        r = (_f(row["outcome_pct"]) / 100.0) / risk
        try:
            from core.costs import cost_in_r
            r -= cost_in_r(risk, "CNC")
        except Exception:
            pass
        out.append((max(-1.5, min(4.0, r)), int(row["worked"] or 0)))
    return out


def _live_stats(setup: str) -> dict[str, Any]:
    try:
        from scan.live_edge import profile_edge
        prof = profile_edge() or {}
    except Exception:
        return {"n": 0, "regimes": {}}
    sig = (prof.get("signals") or {}).get(setup) or {}
    regimes = {}
    for name, stats in (prof.get("regimes") or {}).items():
        if int((stats or {}).get("n") or 0) > 0:
            regimes[str(name)] = dict(stats)
    return {
        "n": int(sig.get("n") or 0),
        "wins": int(sig.get("wins") or 0),
        "win_rate": sig.get("win_rate"),
        "expectancy_r": sig.get("expectancy_r"),
        "regimes": regimes,
    }


def setup_memory(setup: str, *, db_path: Path | None = None) -> dict[str, Any]:
    """Union live tracked signals + desk-settled cases for one setup type."""
    live = _live_stats(setup)
    desk = _agg_rows(_desk_pairs(setup, db_path=db_path))
    n = int(live.get("n") or 0) + int(desk.get("n") or 0)
    wins = int(live.get("wins") or 0) + int(desk.get("wins") or 0)
    # Prefer live_edge expectancy when it already has a sample; else desk R.
    exp = live.get("expectancy_r") if int(live.get("n") or 0) else desk.get("expectancy_r")
    if int(live.get("n") or 0) and int(desk.get("n") or 0) and desk.get("expectancy_r") is not None:
        # Blend sample-weighted when both exist.
        ln, dn = int(live["n"]), int(desk["n"])
        le = float(live.get("expectancy_r") or 0.0)
        de = float(desk.get("expectancy_r") or 0.0)
        exp = round((le * ln + de * dn) / (ln + dn), 3)
    win_rate = round(wins / n * 100.0, 1) if n else None
    return {
        "setup": setup,
        "n": n,
        "wins": wins,
        "win_rate": win_rate,
        "expectancy_r": exp if n else None,
        "regimes": live.get("regimes") or {},
        "live_n": int(live.get("n") or 0),
        "desk_n": int(desk.get("n") or 0),
    }


def _memory_line(idea: str, mem: Mapping[str, Any]) -> str:
    n = int(mem.get("n") or 0)
    setup = _plain_setup(str(mem.get("setup") or "setup"))
    if n <= 0:
        return (
            f"{idea} QuantTerm has not remembered similar {setup} cases yet — "
            "it will check what happens and learn."
        )
    if n < PROVEN_N:
        return (
            f"{idea} QuantTerm has only seen {n} similar case{'s' if n != 1 else ''}. "
            "Promising, but not proven yet."
        )
    exp = mem.get("expectancy_r")
    paid = None
    if exp is not None:
        paid = "made money after costs" if float(exp) > 0 else "did not pay after costs"
    regimes = mem.get("regimes") or {}
    strong = weak = None
    for name, stats in regimes.items():
        label = _regime_plain(str(name))
        rn = int((stats or {}).get("n") or 0)
        wr = (stats or {}).get("win_rate")
        if rn < 5 or wr is None:
            continue
        if label == "strong markets":
            strong = (rn, float(wr))
        elif label == "weak markets":
            weak = (rn, float(wr))
    bits = [f"We saw this {setup} {n} times before."]
    if strong and weak:
        bits.append(
            f"It worked more often in strong markets ({strong[1]:.0f}% of {strong[0]}) "
            f"and failed more often in weak markets ({weak[1]:.0f}% of {weak[0]})."
        )
    elif regimes:
        parts = []
        for name, stats in list(regimes.items())[:3]:
            rn = int((stats or {}).get("n") or 0)
            wr = (stats or {}).get("win_rate")
            if rn and wr is not None:
                parts.append(f"{_regime_plain(str(name))} {wr:.0f}% (n={rn})")
        if parts:
            bits.append("By tape: " + "; ".join(parts) + ".")
    if paid:
        bits.append(f"Across those cases it {paid}.")
    return " ".join(bits)


def remember_case(
    card: Mapping[str, Any],
    *,
    row: Mapping[str, Any] | None = None,
    db_path: Path | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Project a customer Case onto a reco card. Optionally open it in the ledger."""
    symbol = str(card.get("symbol") or (row or {}).get("symbol") or "").upper()
    category_id = str(card.get("category_id") or "")
    setup = primary_setup(row or card, category_id)
    idea = _idea_line(symbol or "This name", setup, str(card.get("setup_label") or ""))
    why = list(card.get("why_now") or card.get("key_points") or [])
    invalidation = list(card.get("what_changes_mind") or [])
    mem = setup_memory(setup, db_path=db_path)
    n = int(mem.get("n") or 0)
    if n <= 0:
        verdict = "unmeasured"
    elif n < PROVEN_N:
        verdict = "unproven"
    elif float(mem.get("expectancy_r") or 0) > 0:
        verdict = "remembered_positive"
    else:
        verdict = "remembered_negative"
    case = {
        "schema_version": 1,
        "case_id": "",
        "symbol": symbol,
        "setup": setup,
        "idea": idea,
        "why_now": why,
        "invalidation": invalidation,
        "n_similar": n,
        "proven": n >= PROVEN_N,
        "verdict": verdict,
        "memory_line": _memory_line(idea, {**mem, "setup": setup}),
        "win_rate": mem.get("win_rate"),
        "expectancy_r": mem.get("expectancy_r") if n >= PROVEN_N else None,
        "places_orders": False,
    }
    if persist and symbol:
        case["case_id"] = open_case(card, row=row, db_path=db_path)
    return case


def open_case(
    card: Mapping[str, Any],
    *,
    row: Mapping[str, Any] | None = None,
    db_path: Path | None = None,
) -> str:
    """Remember that the desk showed this idea today. Deduped per symbol+setup+day."""
    symbol = str(card.get("symbol") or "").upper()
    setup = primary_setup(row or card, str(card.get("category_id") or ""))
    day = _today()
    case_id = f"{symbol}|{setup}|{day}"
    if not symbol:
        return ""
    src = row or card
    why = json.dumps(list(card.get("why_now") or [])[:6], ensure_ascii=False)
    inv = json.dumps(list(card.get("what_changes_mind") or [])[:6], ensure_ascii=False)
    try:
        conn = _conn(db_path)
        existing = conn.execute("SELECT case_id FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if existing:
            conn.close()
            return case_id
        conn.execute(
            """INSERT INTO cases
               (case_id, symbol, setup, category_id, idea, why_now, invalidation,
                opened_at, entry, stop, target, regime, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                case_id, symbol, setup, str(card.get("category_id") or ""),
                _idea_line(symbol, setup, str(card.get("setup_label") or "")),
                why, inv, f"{day}T00:00:00",
                _f(src.get("entry") or src.get("entry_price") or card.get("entry")) or None,
                _f(src.get("stop") or src.get("stop_price") or card.get("stop")) or None,
                _f(src.get("target") or src.get("target_price") or card.get("target")) or None,
                str(src.get("regime") or src.get("market_regime") or ""),
                "desk",
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        return case_id
    return case_id


def _resolve_path(row: Mapping[str, Any]) -> tuple[float, float, int] | None:
    """First-touch target vs stop on official bhavcopy. None = still open / no path."""
    entry, stop, target = _f(row.get("entry")), _f(row.get("stop")), _f(row.get("target"))
    if entry <= 0 or stop <= 0 or target <= entry or stop >= entry:
        return None
    try:
        import pandas as pd
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(str(row["symbol"]))
    except Exception:
        return None
    if df is None or getattr(df, "empty", True) or not {"high", "low", "close"} <= set(df.columns):
        return None
    opened = str(row.get("opened_at") or "")[:10]
    try:
        since = df[df.index >= pd.Timestamp(opened)]
    except Exception:
        return None
    if since is None or since.empty:
        return None
    highs = since["high"].to_numpy(dtype=float)
    lows = since["low"].to_numpy(dtype=float)
    closes = since["close"].to_numpy(dtype=float)
    n = len(highs)
    horizon = min(n, _HORIZON_SESSIONS)
    filled = False
    for i in range(horizon):
        if not filled:
            if highs[i] >= entry:
                filled = True
            else:
                continue
        if lows[i] <= stop:
            return (stop, (stop - entry) / entry * 100.0, 0)
        if highs[i] >= target:
            return (target, (target - entry) / entry * 100.0, 1)
    if not filled:
        return (0.0, 0.0, -1) if n >= _HORIZON_SESSIONS else None
    if n >= _HORIZON_SESSIONS:
        last = float(closes[horizon - 1])
        return (last, (last - entry) / entry * 100.0, 1 if last >= entry else 0)
    return None


def settle_due_cases(*, db_path: Path | None = None, limit: int = 400) -> int:
    """Night path: resolve open cases from official bars. Returns settled count."""
    settled = 0
    try:
        conn = _conn(db_path)
        rows = conn.execute(
            """SELECT case_id, symbol, opened_at, entry, stop, target
               FROM cases WHERE worked IS NULL
               ORDER BY opened_at ASC LIMIT ?""",
            (limit,),
        ).fetchall()
        for row in rows:
            resolved = _resolve_path(dict(row))
            if resolved is None:
                continue
            _price, pct, worked = resolved
            conn.execute(
                """UPDATE cases SET settled_at=?, outcome_pct=?, worked=?
                   WHERE case_id=?""",
                (_now_iso(), float(pct), int(worked), row["case_id"]),
            )
            settled += 1
        conn.commit()
        conn.close()
    except Exception:
        return settled
    return settled


def attach_case(card: dict[str, Any], *, row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    card["case"] = remember_case(card, row=row, persist=True)
    return card


def morning_digest(*, db_path: Path | None = None, limit: int = 6) -> dict[str, Any]:
    """Morning wrap: a few remembered setup types, never invented counts."""
    settle_due_cases(db_path=db_path)
    setups: list[str] = []
    try:
        conn = _conn(db_path)
        for row in conn.execute(
            "SELECT setup, COUNT(*) AS n FROM cases GROUP BY setup ORDER BY n DESC LIMIT 12"
        ):
            setups.append(str(row["setup"]))
        conn.close()
    except Exception:
        setups = []
    if not setups:
        try:
            from scan.live_edge import profile_edge
            ranked = sorted(
                ((k, int((v or {}).get("n") or 0)) for k, v in (profile_edge().get("signals") or {}).items()),
                key=lambda pair: pair[1],
                reverse=True,
            )
            setups = [k for k, n in ranked if n][:8]
        except Exception:
            setups = []
    lines = []
    for setup in setups[:limit]:
        mem = setup_memory(setup, db_path=db_path)
        if int(mem.get("n") or 0) <= 0:
            continue
        lines.append({
            "setup": setup,
            "n_similar": mem["n"],
            "proven": mem["n"] >= PROVEN_N,
            "memory_line": _memory_line(f"This {_plain_setup(setup)} setup.", {**mem, "setup": setup}),
        })
    return {
        "title": "What QuantTerm remembers this morning",
        "blurb": (
            "Cases the desk has already observed. Fewer than 30 similar outcomes "
            "stays unproven. Empty means nothing has been remembered yet."
        ),
        "setups": lines,
        "open_count": _open_count(db_path),
        "settled_count": _settled_count(db_path),
        "places_orders": False,
    }


def _open_count(db_path: Path | None = None) -> int:
    try:
        conn = _conn(db_path)
        n = conn.execute("SELECT COUNT(*) FROM cases WHERE worked IS NULL").fetchone()[0]
        conn.close()
        return int(n or 0)
    except Exception:
        return 0


def _settled_count(db_path: Path | None = None) -> int:
    try:
        conn = _conn(db_path)
        n = conn.execute("SELECT COUNT(*) FROM cases WHERE worked IS NOT NULL").fetchone()[0]
        conn.close()
        return int(n or 0)
    except Exception:
        return 0
