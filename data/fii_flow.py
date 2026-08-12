"""
FII/DII daily flow scraper from NSE India.
Public data — no authentication required.
"""

import time
import requests
from datetime import datetime

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_CACHE: dict = {}
_CACHE_TTL = 30 * 60  # 30 minutes in seconds
_CACHE_TS: float = 0.0

_NSE_BASE = "https://www.nseindia.com"
_NSE_API = f"{_NSE_BASE}/api/fiidiiTradeReact"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": f"{_NSE_BASE}/",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

_EMPTY_RESULT = {
    "error": "",
    "today": {},
    "history": [],
    "insight": "",
    "fii_streak": 0,
    "dii_streak": 0,
}


def _error_result(msg: str) -> dict:
    return {
        "error": msg,
        "today": {},
        "history": [],
        "insight": "Data unavailable",
        "fii_streak": 0,
        "dii_streak": 0,
    }


def _parse_float(val) -> float:
    """Parse a value that may be a string with commas or None."""
    try:
        return float(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _compute_streak(values: list[float]) -> int:
    """
    Given a list of net values (most-recent first), return streak count.
    Positive = consecutive buy (positive net) days.
    Negative = consecutive sell (negative net) days.
    """
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


def _build_insight(history: list[dict], today: dict) -> str:
    fii_net_today = today.get("fii_net_cr", 0.0)
    dii_net_today = today.get("dii_net_cr", 0.0)

    all_fii = [r["fii_net_cr"] for r in history]
    all_dii = [r["dii_net_cr"] for r in history]

    fii_streak = _compute_streak(all_fii)
    dii_streak = _compute_streak(all_dii)

    def _entity_phrase(entity: str, streak: int, net_today: float, all_nets: list[float]) -> str:
        direction = "buyers" if net_today >= 0 else "sellers"
        abs_streak = abs(streak)
        if abs_streak >= 2:
            total = sum(v for v in all_nets[:abs_streak])
            total_str = f"₹{abs(total):,.0f}Cr total"
            return f"{entity}s net {direction} for {abs_streak} consecutive session{'s' if abs_streak > 1 else ''} ({total_str})"
        elif abs(net_today) < 100:
            return f"{entity}s neutral (₹{net_today:+,.0f}Cr)"
        else:
            return f"{entity}s net {direction} today (₹{net_today:+,.0f}Cr)"

    fii_phrase = _entity_phrase("FII", fii_streak, fii_net_today, all_fii)
    dii_phrase = _entity_phrase("DII", dii_streak, dii_net_today, all_dii)
    return f"{fii_phrase}; {dii_phrase}."


def get_fii_dii_flow(days: int = 5) -> dict:
    global _CACHE, _CACHE_TS

    now = time.time()
    if _CACHE and (now - _CACHE_TS) < _CACHE_TTL:
        return _CACHE

    session = requests.Session()
    session.headers.update(_HEADERS)

    try:
        # Hit homepage first to get cookies
        session.get(_NSE_BASE, timeout=10)
    except Exception as exc:
        return _error_result(f"Failed to reach NSE homepage: {exc}")

    try:
        resp = session.get(_NSE_API, timeout=15)
        resp.raise_for_status()
        raw: list = resp.json()
    except Exception as exc:
        return _error_result(f"NSE API request failed: {exc}")

    if not isinstance(raw, list) or len(raw) == 0:
        return _error_result("Unexpected NSE API response format")

    try:
        history: list[dict] = []
        for entry in raw:
            # Each entry contains data for one date; FII and DII rows embedded.
            # NSE typically returns a dict per date with keys like:
            # date, fii_buy, fii_sell, dii_buy, dii_sell  OR
            # date, category, buyValue, sellValue  — depends on API version.

            date_str = (
                entry.get("date")
                or entry.get("Date")
                or entry.get("tradeDate")
                or ""
            )

            # Try flat format first (one dict per date, combined)
            if "fiiBuyValue" in entry or "fii_buy" in entry:
                fii_buy = _parse_float(entry.get("fiiBuyValue") or entry.get("fii_buy", 0))
                fii_sell = _parse_float(entry.get("fiiSellValue") or entry.get("fii_sell", 0))
                dii_buy = _parse_float(entry.get("diiBuyValue") or entry.get("dii_buy", 0))
                dii_sell = _parse_float(entry.get("diiSellValue") or entry.get("dii_sell", 0))
                history.append(
                    {
                        "date": date_str,
                        "fii_net_cr": round(fii_buy - fii_sell, 2),
                        "dii_net_cr": round(dii_buy - dii_sell, 2),
                    }
                )
            elif "buyValue" in entry and "category" in entry:
                # Row-per-category format — accumulate and merge
                cat = str(entry.get("category", "")).upper()
                buy = _parse_float(entry.get("buyValue", 0))
                sell = _parse_float(entry.get("sellValue", 0))
                net = round(buy - sell, 2)

                existing = next((h for h in history if h["date"] == date_str), None)
                if existing is None:
                    existing = {"date": date_str, "fii_net_cr": 0.0, "dii_net_cr": 0.0}
                    history.append(existing)

                if "FII" in cat:
                    existing["fii_net_cr"] = round(existing["fii_net_cr"] + net, 2)
                elif "DII" in cat:
                    existing["dii_net_cr"] = round(existing["dii_net_cr"] + net, 2)
            else:
                # Attempt generic key scan
                fii_buy = _parse_float(
                    entry.get("fiiBuyValue") or entry.get("FII_BUY_VALUE") or 0
                )
                fii_sell = _parse_float(
                    entry.get("fiiSellValue") or entry.get("FII_SELL_VALUE") or 0
                )
                dii_buy = _parse_float(
                    entry.get("diiBuyValue") or entry.get("DII_BUY_VALUE") or 0
                )
                dii_sell = _parse_float(
                    entry.get("diiSellValue") or entry.get("DII_SELL_VALUE") or 0
                )
                if fii_buy or fii_sell or dii_buy or dii_sell:
                    history.append(
                        {
                            "date": date_str,
                            "fii_net_cr": round(fii_buy - fii_sell, 2),
                            "dii_net_cr": round(dii_buy - dii_sell, 2),
                        }
                    )

        # Sort most-recent first
        history.sort(key=lambda r: r.get("date", ""), reverse=True)

        if not history:
            return _error_result("Could not parse any FII/DII rows from NSE response")

        today = history[0]
        trimmed_history = history[:days]

        fii_nets = [r["fii_net_cr"] for r in trimmed_history]
        dii_nets = [r["dii_net_cr"] for r in trimmed_history]

        result = {
            "error": "",
            "today": today,
            "history": trimmed_history,
            "fii_streak": _compute_streak(fii_nets),
            "dii_streak": _compute_streak(dii_nets),
            "insight": _build_insight(trimmed_history, today),
        }

        _CACHE = result
        _CACHE_TS = now
        return result

    except Exception as exc:
        return _error_result(f"Parsing error: {exc}")
