"""Persistent FII/DII cash-market history from NSE public endpoints.

NSE ``fiidiiTradeReact`` returns many recent sessions in one call. This store
merges live fetches into SQLite so the terminal, reports and Brain read the same
numbers — no Streamlit cache required.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from logger import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _ROOT / "logs" / "product" / "fii_dii.sqlite"
_STATE_PATH = _ROOT / "logs" / "product" / "fii_dii_backfill.json"

_NSE_BASE = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/reports/fii-dii",
}

# Refresh when store is empty or older than TTL — one NSE call returns many sessions.
_REFRESH_TTL_S = 3600 * 3
_FAIL_BACKOFF_S = 600
_last_fetch_attempt_s: float = 0.0
_last_fetch_fail_s: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_float(value: Any) -> float:
    try:
        return float(str(value).replace(",", "").strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_trade_react_rows(raw: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Parse NSE fiidiiTradeReact list → normalised daily cash rows (₹ Cr)."""
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        if "fiiBuyValue" in item or "fii_buy" in item:
            date_str = str(item.get("date") or item.get("Date") or "")
            try:
                dt = datetime.strptime(date_str.strip(), "%d-%b-%Y").date()
            except ValueError:
                try:
                    dt = datetime.strptime(date_str.strip(), "%d-%m-%Y").date()
                except ValueError:
                    continue
            fii_buy = _parse_float(item.get("fiiBuyValue") or item.get("fii_buy"))
            fii_sell = _parse_float(item.get("fiiSellValue") or item.get("fii_sell"))
            dii_buy = _parse_float(item.get("diiBuyValue") or item.get("dii_buy"))
            dii_sell = _parse_float(item.get("diiSellValue") or item.get("dii_sell"))
            out.append(
                {
                    "date": dt.isoformat(),
                    "fii_buy": round(fii_buy, 2),
                    "fii_sell": round(fii_sell, 2),
                    "fii_net": round(fii_buy - fii_sell, 2),
                    "dii_buy": round(dii_buy, 2),
                    "dii_sell": round(dii_sell, 2),
                    "dii_net": round(dii_buy - dii_sell, 2),
                }
            )
        elif "buyValue" in item and "category" in item:
            date_str = str(item.get("date") or "")
            try:
                dt = datetime.strptime(date_str.strip(), "%d-%b-%Y").date()
            except ValueError:
                try:
                    dt = datetime.strptime(date_str.strip(), "%d-%m-%Y").date()
                except ValueError:
                    continue
            cat = str(item.get("category", "")).upper()
            buy = _parse_float(item.get("buyValue"))
            sell = _parse_float(item.get("sellValue"))
            net = round(buy - sell, 2)
            existing = next((r for r in out if r["date"] == dt.isoformat()), None)
            if existing is None:
                existing = {
                    "date": dt.isoformat(),
                    "fii_buy": 0.0,
                    "fii_sell": 0.0,
                    "fii_net": 0.0,
                    "dii_buy": 0.0,
                    "dii_sell": 0.0,
                    "dii_net": 0.0,
                }
                out.append(existing)
            if "FII" in cat or "FPI" in cat:
                existing["fii_net"] = round(existing["fii_net"] + net, 2)
                existing["fii_buy"] = round(existing["fii_buy"] + buy, 2)
                existing["fii_sell"] = round(existing["fii_sell"] + sell, 2)
            elif "DII" in cat:
                existing["dii_net"] = round(existing["dii_net"] + net, 2)
                existing["dii_buy"] = round(existing["dii_buy"] + buy, 2)
                existing["dii_sell"] = round(existing["dii_sell"] + sell, 2)
    return out


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cash_flows (
            date TEXT PRIMARY KEY,
            fii_buy REAL NOT NULL,
            fii_sell REAL NOT NULL,
            fii_net REAL NOT NULL,
            dii_buy REAL NOT NULL,
            dii_sell REAL NOT NULL,
            dii_net REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def reset_store() -> None:
    """Hermetic tests: drop persisted FII/DII history."""
    global _last_fetch_attempt_s, _last_fetch_fail_s
    if _DB_PATH.exists():
        _DB_PATH.unlink()
    if _STATE_PATH.exists():
        _STATE_PATH.unlink()
    _last_fetch_attempt_s = 0.0
    _last_fetch_fail_s = 0.0


def upsert_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    """Insert or replace cash-market rows. Returns count written."""
    if not rows:
        return 0
    stamp = _now_iso()
    with _connect() as conn:
        n = 0
        for row in rows:
            date = str(row.get("date") or "").strip()
            if not date:
                continue
            conn.execute(
                """
                INSERT INTO cash_flows (
                    date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    fii_buy=excluded.fii_buy,
                    fii_sell=excluded.fii_sell,
                    fii_net=excluded.fii_net,
                    dii_buy=excluded.dii_buy,
                    dii_sell=excluded.dii_sell,
                    dii_net=excluded.dii_net,
                    updated_at=excluded.updated_at
                """,
                (
                    date,
                    float(row.get("fii_buy", 0)),
                    float(row.get("fii_sell", 0)),
                    float(row.get("fii_net", 0)),
                    float(row.get("dii_buy", 0)),
                    float(row.get("dii_sell", 0)),
                    float(row.get("dii_net", 0)),
                    stamp,
                ),
            )
            n += 1
        conn.commit()
    return n


def count_rows() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM cash_flows").fetchone()
        return int(row["n"] if row else 0)


def _store_staleness_s() -> float:
    """Seconds since the newest persisted row was written."""
    with _connect() as conn:
        row = conn.execute("SELECT MAX(updated_at) AS ts FROM cash_flows").fetchone()
    ts = str(row["ts"] if row and row["ts"] else "")
    if not ts:
        return float("inf")
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())
    except Exception:
        return float("inf")


def refresh_if_needed(
    *,
    force: bool = False,
    fetcher: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """
    Lazy sync: one NSE API call merges every session NSE returns (~30+ days).
    No bulk pre-download — runs only when the store is empty, stale, or forced.
    """
    global _last_fetch_attempt_s, _last_fetch_fail_s
    now = time.time()
    stale = _store_staleness_s()
    empty = count_rows() == 0
    need = force or empty or stale > _REFRESH_TTL_S

    if not need:
        return {
            "fetched": False,
            "reason": "fresh",
            "row_count": count_rows(),
            "staleness_s": round(stale, 0),
        }

    if not force and _last_fetch_fail_s and (now - _last_fetch_fail_s) < _FAIL_BACKOFF_S:
        return {
            "fetched": False,
            "reason": "backoff",
            "row_count": count_rows(),
            "error": "recent_fetch_failed",
        }

    _last_fetch_attempt_s = now
    try:
        rows = (fetcher or fetch_from_nse)()
        inserted = upsert_rows(rows)
        _last_fetch_fail_s = 0.0
        log.info("fii_dii_lazy_refresh", rows=inserted, total=count_rows())
        return {
            "fetched": True,
            "reason": "empty" if empty else "stale" if stale > _REFRESH_TTL_S else "forced",
            "rows_merged": inserted,
            "row_count": count_rows(),
        }
    except Exception as exc:
        _last_fetch_fail_s = now
        log.warning("fii_dii_lazy_refresh_failed", error=str(exc)[:120])
        return {
            "fetched": False,
            "reason": "error",
            "row_count": count_rows(),
            "error": type(exc).__name__,
        }


def get_history(days: int = 30) -> list[dict[str, Any]]:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=max(1, int(days)))).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net, updated_at
            FROM cash_flows
            WHERE date >= ?
            ORDER BY date DESC
            """,
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def _compute_streak(values: list[float]) -> int:
    if not values:
        return 0
    sign = 1 if values[0] >= 0 else -1
    streak = 0
    for v in values:
        if (sign == 1 and v >= 0) or (sign == -1 and v < 0):
            streak += sign
        else:
            break
    return streak


def summarize(days: int = 30, *, auto_refresh: bool = True) -> dict[str, Any]:
    if auto_refresh:
        refresh_if_needed()
    history = get_history(days)
    if not history:
        return {
            "available": False,
            "days": days,
            "sessions": 0,
            "history": [],
            "today": {},
            "totals": {},
            "fii_streak": 0,
            "dii_streak": 0,
            "bias": "",
            "note": "",
            "store_path": str(_DB_PATH),
            "row_count": 0,
        }
    today = history[0]
    fii_vals = [float(r["fii_net"]) for r in history]
    dii_vals = [float(r["dii_net"]) for r in history]
    fii_total = round(sum(fii_vals), 2)
    dii_total = round(sum(dii_vals), 2)
    fii_streak = _compute_streak(fii_vals)
    dii_streak = _compute_streak(dii_vals)

    from data.institutional_flows import parse_fii_dii

    bias_row = parse_fii_dii(
        [
            {"category": "FII/FPI", "netValue": str(today["fii_net"]), "date": today["date"]},
            {"category": "DII", "netValue": str(today["dii_net"]), "date": today["date"]},
        ]
    )
    bias = str((bias_row or {}).get("bias") or "")
    note = str((bias_row or {}).get("note") or "")

    return {
        "available": True,
        "days": days,
        "sessions": len(history),
        "history": history,
        "today": today,
        "totals": {
            "fii_net_cr": fii_total,
            "dii_net_cr": dii_total,
            "combined_net_cr": round(fii_total + dii_total, 2),
        },
        "fii_streak": fii_streak,
        "dii_streak": dii_streak,
        "bias": bias,
        "note": note,
        "store_path": str(_DB_PATH),
        "row_count": count_rows(),
        "latest_date": today["date"],
    }


def fetch_from_nse() -> list[dict[str, Any]]:
    """Live fetch from NSE fiidiiTradeReact (network)."""
    import requests

    session = requests.Session()
    session.headers.update(_HEADERS)
    session.get(_NSE_BASE, timeout=10)
    resp = session.get(f"{_NSE_BASE}/api/fiidiiTradeReact", timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    if not isinstance(raw, list):
        return []
    return parse_trade_react_rows(raw)


def run_backfill(
    *,
    days: int = 90,
    force_fetch: bool = True,
    fetcher: Callable[[], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Force-refresh alias for operators; normal reads use refresh_if_needed()."""
    refresh = refresh_if_needed(force=force_fetch, fetcher=fetcher)
    state = {
        "updated_at": _now_iso(),
        "days_requested": days,
        "refresh": refresh,
        "row_count": count_rows(),
        "sessions_in_window": len(get_history(days)),
        "complete": count_rows() > 0,
        "store_path": str(_DB_PATH),
    }
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def backfill_status() -> dict[str, Any]:
    state: dict[str, Any] = {}
    try:
        if _STATE_PATH.exists():
            state = dict(json.loads(_STATE_PATH.read_text(encoding="utf-8")))
    except Exception:
        state = {}
    summary = summarize(30, auto_refresh=False)
    stale_s = _store_staleness_s()
    return {
        "available": summary.get("available") or count_rows() > 0,
        "row_count": count_rows(),
        "latest_date": summary.get("latest_date") or "",
        "sessions_in_store": summary.get("sessions", 0),
        "staleness_s": round(stale_s, 0) if stale_s != float("inf") else None,
        "lazy_refresh": True,
        "refresh_ttl_s": _REFRESH_TTL_S,
        "last_forced_refresh": state,
        "state_path": str(_STATE_PATH),
        "store_path": str(_DB_PATH),
    }


def workspace_payload(days: int = 30, *, include_nifty_options: bool = True) -> dict[str, Any]:
    """API/dashboard payload: lazy NSE sync + cached bulk/options reads."""
    summary = summarize(days, auto_refresh=True)
    derivatives: dict[str, Any] = {}
    bulk_deals: list[dict[str, Any]] = []
    bulk_buys: list[str] = []
    flows_note = summary.get("note") or ""
    try:
        from data.fii_dii import get_fii_derivative_stats_uncached

        derivatives = get_fii_derivative_stats_uncached()
    except Exception:
        derivatives = {}
    try:
        from data.institutional_flows import get_flows

        flows = get_flows()
        bulk_deals = list(flows.get("bulk_deals") or [])
        bulk_buys = list(flows.get("bulk_buys") or [])
        if flows.get("fii_dii") and not summary.get("note"):
            flows_note = str((flows.get("fii_dii") or {}).get("note") or "")
    except Exception:
        pass
    nifty_options: dict[str, Any] = {"available": False}
    if include_nifty_options:
        try:
            from options.chain_fetch import chain_workspace_cached

            nifty_options = chain_workspace_cached("NIFTY")
        except Exception:
            pass
    return {
        "available": bool(summary.get("available")),
        "cash": summary,
        "derivatives": derivatives,
        "bulk_deals": bulk_deals[:100],
        "bulk_buy_symbols": bulk_buys[:50],
        "nifty_options": nifty_options,
        "insight": flows_note or summary.get("note") or "",
        "generated_at": _now_iso(),
        "lazy_sync": True,
    }
