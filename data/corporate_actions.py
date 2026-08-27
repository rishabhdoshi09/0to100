"""
🔧 Corporate-Action Adjustment — make historical prices continuous & trustworthy.

The professor's disqualifier, fixed: NSE bhavcopy is UNADJUSTED, so a 1:1 bonus
or a 1→5 split reads as a phantom −50% / −80% crash. `core.data_integrity`
DETECTS those gaps; this module REMOVES them, so a backtest measures price
action, not accounting.

Design (deliberately minimal, and honest about data):

  • An "event" is {symbol, ex_date, factor, type}. `factor` = the multiple by
    which the SHARE COUNT rose on the ex-date: a 1:1 bonus → 2.0, a 1→5 split
    → 5.0, a 3:1 bonus (3 new per 1 held) → 4.0. Prices BEFORE the ex-date are
    divided by the cumulative factor; volumes are multiplied — the standard
    back-adjustment that pins history to today's share base.
  • adjust_frame() is a PURE function over a datetime-indexed OHLCV frame.
  • load_events() reads a real CA table from `logs/ca_events.json`. If that file
    is ABSENT it returns {} and the whole system behaves exactly as before —
    there is NO synthesised or guessed adjustment (invariant #1: no fake data).
    The events themselves must come from NSE corporate-actions archives; this
    module cannot and will not invent them.

Adjustment is applied ON READ (data.bhavcopy_store.get_ohlcv), so the on-disk
store stays raw, there is no double-adjustment across rebuilds, and an updated
CA table takes effect with no re-download.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_CA_FILE = Path(__file__).resolve().parent.parent / "logs" / "ca_events.json"
_VALID_TYPES = {"split", "bonus", "consolidation", "dividend"}


def _events_path() -> Path:
    override = os.getenv("QT_CA_EVENTS_FILE")
    return Path(override) if override else _CA_FILE


def load_events(path=None) -> dict:
    """Read the corporate-action table → {SYMBOL: [event, ...]}. Returns {} when
    the file is absent or unreadable — NEVER a fabricated table (no-file ⇒ the
    system runs exactly as it did before, un-adjusted but flagged by the
    integrity guard). Each event: {ex_date, factor>0, type}. Malformed rows are
    dropped, not guessed."""
    import pandas as pd
    p = Path(path) if path else _events_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {}
    out: dict[str, list[dict]] = {}
    for row in raw if isinstance(raw, list) else []:
        try:
            sym = str(row["symbol"]).strip().upper()
            factor = float(row["factor"])
            ex = pd.Timestamp(row["ex_date"])
            typ = str(row.get("type", "split")).lower()
            if not sym or factor <= 0 or factor == 1.0 or pd.isna(ex):
                continue
            if typ not in _VALID_TYPES:
                continue
            out.setdefault(sym, []).append(
                {"ex_date": ex, "factor": factor, "type": typ})
        except Exception:
            continue
    for sym in out:
        out[sym].sort(key=lambda e: e["ex_date"])
    return out


def adjust_frame(df, events, copy: bool = True):
    """Back-adjust one symbol's OHLCV frame for its corporate actions. PURE by
    default.

    `df` is datetime-indexed with any of open/high/low/close/volume (deliv_per, a
    percentage, is left untouched). `events` is that symbol's event list. Bars
    STRICTLY BEFORE an ex-date have prices divided by the event's factor and
    volume multiplied by it, applied cumulatively across all events. Returns a new
    frame; the input is not mutated. Empty/absent events → the frame is returned
    unchanged (a copy). `copy=False` mutates `df` in place — for callers that
    already own a private copy (e.g. bhavcopy_store.get_ohlcv) and want to skip a
    redundant second copy."""
    if df is None or getattr(df, "empty", True) or not events:
        return (df.copy() if copy else df) if df is not None else df
    import numpy as np
    import pandas as pd
    out = df.copy() if copy else df
    idx = pd.DatetimeIndex(out.index)
    # divisor[i] = product of factors of every event whose ex_date is AFTER bar i.
    # A DatetimeIndex comparison already yields a plain ndarray mask.
    divisor = np.ones(len(out), dtype=float)
    for e in events:
        divisor[idx < e["ex_date"]] *= e["factor"]
    price_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
    for c in price_cols:
        out[c] = out[c].to_numpy(dtype=float) / divisor
    if "volume" in out.columns:
        out["volume"] = out["volume"].to_numpy(dtype=float) * divisor
    return out


def is_continuous(df, threshold_pct: float = None) -> bool:
    """True when the (adjusted) close series has NO phantom gap — the acceptance
    test for a correct adjustment. Delegates to the same detector the integrity
    guard uses, so 'adjusted' means exactly 'the guard now passes'."""
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return True
    from core.data_integrity import phantom_gaps, _GAP_PCT
    thr = threshold_pct if threshold_pct is not None else _GAP_PCT
    return len(phantom_gaps(df["close"].to_numpy(dtype=float), thr)) == 0


def events_path():
    return _events_path()


_NSE_CA_URL = "https://www.nseindia.com/api/corporates-corporateActions"
_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
}
_BONUS_RE = re.compile(r"\bbonus\s+(\d+)\s*:\s*(\d+)\b", re.I)
_SPLIT_RE = re.compile(
    r"from\s+r[se]\.?\s*([\d.]+).{0,40}?to\s+r[se]\.?\s*([\d.]+)",
    re.I,
)
_STALE_S = 20 * 60 * 60
_EQ_SERIES = {"EQ", "BE", "SM"}


def parse_action_subject(subject: str) -> tuple[str, float] | None:
    """Map an official NSE subject to a share-count event. Dividends/rights are skipped."""
    text = str(subject or "").strip()
    if not text:
        return None
    low = text.lower()
    if "ncrps" in low or "ncd" in low:
        return None
    if "right" in low:
        return None
    bonus = _BONUS_RE.search(text)
    if bonus and "bonus" in low:
        issued, held = int(bonus.group(1)), int(bonus.group(2))
        if issued <= 0 or held <= 0:
            return None
        return "bonus", 1.0 + (issued / held)
    split = _SPLIT_RE.search(text)
    if split and ("split" in low or "sub-division" in low or "sub division" in low or "consolidat" in low):
        old_fv, new_fv = float(split.group(1)), float(split.group(2))
        if old_fv <= 0 or new_fv <= 0 or old_fv == new_fv:
            return None
        kind = "consolidation" if new_fv > old_fv else "split"
        return kind, old_fv / new_fv
    return None


def _event_key(row: dict) -> tuple:
    return (
        str(row.get("symbol") or "").upper(),
        str(row.get("ex_date") or "")[:10],
        str(row.get("type") or ""),
        round(float(row.get("factor") or 0), 6),
    )


def _rows_from_nse(payload) -> list[dict]:
    import pandas as pd
    rows = payload if isinstance(payload, list) else []
    if isinstance(payload, dict):
        rows = list(payload.get("data") or payload.get("corporateActions") or [])
    out: list[dict] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        series = str(raw.get("series") or "").upper()
        if series and series not in _EQ_SERIES:
            continue
        parsed = parse_action_subject(str(raw.get("subject") or ""))
        if parsed is None:
            continue
        kind, factor = parsed
        if factor <= 0 or factor == 1.0:
            continue
        symbol = str(raw.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        try:
            ex = pd.Timestamp(raw.get("exDate") or raw.get("ex_date") or "")
        except Exception:
            continue
        if pd.isna(ex):
            continue
        out.append({
            "symbol": symbol,
            "ex_date": ex.strftime("%Y-%m-%d"),
            "factor": round(float(factor), 6),
            "type": kind,
            "subject": str(raw.get("subject") or ""),
            "source": "nse_corporate_actions",
        })
    return out


def _nse_session():
    import requests
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com/", timeout=12)
    except Exception:
        pass
    return session


def _fetch_nse_window(session, start, end) -> list[dict]:
    from datetime import date as date_cls
    if not isinstance(start, date_cls):
        return []
    params = {
        "index": "equities",
        "from_date": start.strftime("%d-%m-%Y"),
        "to_date": end.strftime("%d-%m-%Y"),
    }
    resp = session.get(_NSE_CA_URL, params=params, timeout=18)
    if resp.status_code in {401, 403}:
        try:
            session.get("https://www.nseindia.com/", timeout=12)
        except Exception:
            pass
        resp = session.get(_NSE_CA_URL, params=params, timeout=18)
    if resp.status_code != 200:
        raise RuntimeError(f"NSE corporate-actions HTTP {resp.status_code}")
    return _rows_from_nse(resp.json())


def _existing_rows(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [dict(row) for row in raw] if isinstance(raw, list) else []


def _atomic_write_rows(path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def ledger_status(path=None) -> dict:
    p = Path(path) if path else _events_path()
    events = load_events(p)
    n_events = sum(len(v) for v in events.values())
    return {
        "available": p.exists() and n_events > 0,
        "n_symbols": len(events),
        "n_events": n_events,
        "path": str(p),
        "source": "nse_corporate_actions",
        "adjustment_verified": n_events > 0,
    }


def refresh_events(*, force: bool = False, years: int = 5, budget_s: float = 25.0,
                   session=None) -> dict:
    """Download official NSE bonus/split/consolidation rows into ``logs/ca_events.json``.

    Dividends and rights are ignored: this table only stores share-count factors used
    to back-adjust bhavcopy. Existing rows are kept when a window fetch fails.
    """
    import time
    from datetime import date, timedelta

    path = _events_path()
    existing = _existing_rows(path)
    age_s = None
    if path.exists():
        try:
            age_s = max(0.0, time.time() - path.stat().st_mtime)
        except OSError:
            age_s = None
    if not force and existing and age_s is not None and age_s < _STALE_S:
        events = load_events(path)
        return {
            "available": True,
            "symbols": len(events),
            "events": sum(len(v) for v in events.values()),
            "path": str(path),
            "source": "nse_corporate_actions_cached",
            "fetched": 0,
        }

    merged = {_event_key(row): row for row in existing if _event_key(row)[0]}
    fetched = 0
    errors: list[str] = []
    started = time.monotonic()
    end = date.today()
    window = timedelta(days=180)
    horizon = date.today() - timedelta(days=max(365, int(years) * 365))
    sess = session
    try:
        while end >= horizon and (time.monotonic() - started) < max(4.0, float(budget_s)):
            start = max(horizon, end - window)
            try:
                if sess is None:
                    sess = _nse_session()
                rows = _fetch_nse_window(sess, start, end)
                fetched += len(rows)
                for row in rows:
                    merged[_event_key(row)] = row
            except Exception as exc:
                errors.append(str(exc)[:180])
                break
            if start <= horizon:
                break
            end = start - timedelta(days=1)
    except Exception as exc:
        errors.append(str(exc)[:180])

    rows = sorted(merged.values(), key=lambda r: (r.get("ex_date") or "", r.get("symbol") or ""))
    if rows:
        _atomic_write_rows(path, rows)
        try:
            from data.bhavcopy_store import reload_corporate_actions
            reload_corporate_actions()
        except Exception:
            pass
    events = load_events(path)
    available = bool(events)
    if available:
        print(
            f"[CA] official NSE actions · {sum(len(v) for v in events.values())} events · "
            f"{len(events)} symbols"
            + (f" · fetched {fetched}" if fetched else " · cached"),
            flush=True,
        )
    elif errors:
        print(f"[CA] official table unavailable · {errors[0]}", flush=True)
    return {
        "available": available,
        "symbols": len(events),
        "events": sum(len(v) for v in events.values()),
        "path": str(path),
        "source": "nse_corporate_actions",
        "fetched": fetched,
        "errors": errors[:4],
    }
