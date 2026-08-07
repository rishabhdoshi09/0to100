"""US retail product facade — readiness, overview, scan, stock workspace.

Honesty contract:
  • Yahoo Finance = free primary EOD (Kite has no US cash market)
  • NASDAQ Trader = official listing directory
  • Index membership from Wikipedia + listing cross-check (live vs curated labeled)
  • No US options / F&O desk until a real OPRA-class source exists
  • Paper autopilot only — never places live US broker orders
  • Never invents prices, constituents, or setups
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from scan.us_market_scan_service import US_SCAN_PATH, load_us_scan


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def readiness() -> dict[str, Any]:
    from data import us_history_store as hist
    from data.us_universe import get_us_universe_with_names

    history = hist.status()
    try:
        names = dict(get_us_universe_with_names() or {})
        universe_count = len(names)
        universe_source = "nasdaq_trader_or_cache" if universe_count > 100 else "curated_fallback"
    except Exception as exc:
        names = {}
        universe_count = 0
        universe_source = f"unavailable:{exc}"

    scan = load_us_scan() or {}
    scan_available = bool(scan.get("records") is not None and scan)
    lanes = [
        {
            "key": "universe",
            "label": "US listings",
            "status": "READY" if universe_count >= 40 else "MISSING",
            "available": universe_count >= 40,
            "as_of": "",
            "details": f"{universe_count} symbols · source {universe_source}",
            "action": "Refresh US universe (NASDAQ Trader directory)",
        },
        {
            "key": "history",
            "label": "US EOD history cache",
            "status": "READY" if history.get("ready") else "INCOMPLETE",
            "available": bool(history.get("ready")),
            "as_of": history.get("latest_date") or "",
            "details": (
                f"{history.get('symbols', 0)} cached · latest {history.get('latest_date') or '—'}"
            ),
            "action": "Run US data prepare (Yahoo daily bars → disk cache)",
        },
        {
            "key": "scan",
            "label": "US market scan",
            "status": "READY" if scan_available else "MISSING",
            "available": scan_available,
            "as_of": str(scan.get("scanned_at") or ""),
            "details": (
                f"{len(scan.get('records') or [])} setups · scope {scan.get('scope') or '—'}"
                if scan_available else "No persisted US scan yet"
            ),
            "action": "Run US market scan (default S&P 500 liquid scope)",
        },
    ]
    ready_n = sum(1 for lane in lanes if lane["available"])
    state = "READY" if ready_n == len(lanes) else ("PARTIAL" if ready_n else "EMPTY")
    return {
        "schema_version": 1,
        "market": "US",
        "generated_at": _now_iso(),
        "state": state,
        "score": round(100 * ready_n / max(1, len(lanes))),
        "lanes": lanes,
        "history": history,
        "universe_size": universe_count,
        "universe_source": universe_source,
        "scan_path": str(US_SCAN_PATH),
        "honesty": (
            "US retail data uses NASDAQ Trader listings + Yahoo Finance daily bars. "
            "No live US broker orders. No US options chain."
        ),
        "places_orders": False,
        "recommended_action": (
            "US stack ready — open US Scanner"
            if state == "READY"
            else "Prepare US history, then run US market scan"
        ),
    }


def overview() -> dict[str, Any]:
    """Retail US market overview — indices + session + last scan summary."""
    from core.market_session import us_market_open, market_meta
    from data.us_data import us_live_prices

    meta = market_meta("US")
    session_open = bool(us_market_open())
    index_symbols = ["^GSPC", "^IXIC", "^DJI", "^VIX"]
    quotes = us_live_prices(index_symbols)
    indices = []
    labels = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow 30", "^VIX": "VIX"}
    for sym in index_symbols:
        q = quotes.get(sym) or {}
        name = labels.get(sym, sym)
        indices.append({
            "symbol": sym,
            "name": name,  # Market Decision Brief / pulse consumers
            "label": name,
            "price": q.get("price"),
            "previous_close": q.get("previous_close"),
            "chg_pct": q.get("chg_pct"),
            "available": bool(q.get("price")),
        })
    scan = load_us_scan() or {}
    summary = dict(scan.get("summary") or {})
    return {
        "schema_version": 1,
        "market": "US",
        "generated_at": _now_iso(),
        "session_open": session_open,
        "session_label": "US OPEN" if session_open else "US CLOSED",
        "timezone": meta.get("tz"),
        "currency": "USD",
        "indices": indices,
        "scan": {
            "available": bool(scan),
            "scanned_at": scan.get("scanned_at", ""),
            "scope": scan.get("scope", ""),
            "universe_size": scan.get("universe_size", 0),
            "summary": summary,
        },
        "source": "yfinance",
        "honesty": (
            "Index quotes via Yahoo fast_info (may be delayed). "
            "Not a substitute for a paid US market-data feed."
        ),
        "places_orders": False,
    }


def scan_payload() -> dict[str, Any]:
    payload = load_us_scan() or {}
    records = [dict(row) for row in (payload.get("records") or []) if isinstance(row, dict)]
    return {
        "available": bool(payload),
        "market": "US",
        "currency": "USD",
        "scanned_at": payload.get("scanned_at", ""),
        "universe_size": int(payload.get("universe_size", 0) or 0),
        "scope": payload.get("scope", ""),
        "source": payload.get("source", "yfinance"),
        "summary": dict(payload.get("summary") or {}),
        "records": records,
        "honesty": payload.get("honesty") or (
            "Persisted US scan from Yahoo daily bars. Not a live US order signal."
        ),
        "places_orders": False,
        "path": str(US_SCAN_PATH),
    }


def paper_status() -> dict[str, Any]:
    try:
        from execution.us_autopilot import get_status

        status = dict(get_status() or {})
    except Exception as exc:
        status = {"available": False, "error": str(exc)}
    status.setdefault("market", "US")
    status.setdefault("places_orders", False)
    status.setdefault(
        "honesty",
        "US autopilot is paper-only. Live US trading needs a real broker adapter (e.g. Alpaca).",
    )
    return status


def stock_workspace(symbol: str) -> dict[str, Any]:
    """Lightweight US stock workspace — chart bars + last scan row. No invented fundamentals."""
    sym = str(symbol or "").strip().upper()
    if not sym or len(sym) > 16:
        return {
            "available": False,
            "symbol": sym,
            "error": "Invalid US symbol",
            "places_orders": False,
        }
    from data import us_history_store as hist
    from data.us_universe import get_us_universe_with_names

    names = {}
    try:
        names = dict(get_us_universe_with_names() or {})
    except Exception:
        names = {}
    frame = hist.get_ohlcv(sym, allow_network=True)
    bars = []
    if frame is not None and len(frame):
        for index, row in frame.tail(220).iterrows():
            stamp = getattr(index, "date", lambda: index)()
            bars.append({
                "time": str(stamp),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0) or 0.0),
            })
    scan = scan_payload()
    row = next((r for r in scan.get("records") or [] if r.get("symbol") == sym), None)
    return {
        "schema_version": 1,
        "market": "US",
        "currency": "USD",
        "available": bool(bars) or bool(row),
        "symbol": sym,
        "company": names.get(sym, sym),
        "bars": bars,
        "history_source": "yfinance",
        "scan_row": row,
        "fundamentals": {
            "available": False,
            "message": "US fundamentals desk not wired — prices/setups only (no invented ratios).",
        },
        "options": {
            "available": False,
            "message": "No US options desk — OPRA-class source not configured.",
        },
        "honesty": (
            "US Stock Intelligence shows Yahoo daily history + last scan setup only. "
            "Not a live order ticket."
        ),
        "places_orders": False,
        "generated_at": _now_iso(),
    }


def dashboard() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "market": "US",
        "generated_at": _now_iso(),
        "readiness": readiness(),
        "overview": overview(),
        "scan": scan_payload(),
        "paper": paper_status(),
        "places_orders": False,
        "honesty": (
            "US retail dashboard — listings, Yahoo EOD, scan setups, paper autopilot. "
            "India NSE stack is unchanged."
        ),
    }


def education_concepts() -> list[dict[str, Any]]:
    """US-specific evergreen teach-ins (fixed copy, not invented blogs)."""
    return [
        {
            "id": "us-session",
            "title": "US regular session",
            "teach_point": (
                "NYSE/Nasdaq cash session is 09:30–16:00 America/New_York. "
                "Pre/post-market prints exist but QuantTerm US scans use settled daily bars."
            ),
            "why_it_matters": "Intraday incomplete daily bars are dropped so volume signals stay honest.",
        },
        {
            "id": "us-yahoo-primary",
            "title": "Why Yahoo is the US primary feed here",
            "teach_point": (
                "Zerodha Kite does not serve US cash equities. "
                "Yahoo Finance is the free legitimate primary for QuantTerm US EOD."
            ),
            "why_it_matters": "Delayed/free data is labeled — never presented as a paid institutional feed.",
        },
        {
            "id": "us-paper-only",
            "title": "US autopilot is paper-only",
            "teach_point": (
                "US paper autopilot journals simulated trades with US cost assumptions. "
                "Live US orders need a real broker adapter (not built)."
            ),
            "why_it_matters": "Retail discipline first — no pretend LIVE.",
        },
        {
            "id": "us-no-options",
            "title": "No US options desk yet",
            "teach_point": (
                "India F&O desk uses Kite/NSE sources. US options need OPRA-class data — "
                "absent means unavailable, not estimated Greeks."
            ),
            "why_it_matters": "Missing derivatives context is shown as missing.",
        },
    ]
