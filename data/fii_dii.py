"""
FII/DII activity and bulk/block deal data from NSE public endpoints.
No API key required — NSE publishes this daily.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger("quantterm.fii_dii")

_NSE_BASE = "https://www.nseindia.com"
_DERIV_STATS_TTL_S = 3600
_deriv_stats_cache: dict | None = None
_deriv_stats_cache_at: float = 0.0
_deriv_stats_unavailable_logged = False
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def _nse_session() -> requests.Session:
    """Create a requests Session with NSE cookies pre-loaded."""
    s = requests.Session()
    s.headers.update(_NSE_HEADERS)
    try:
        s.get(_NSE_BASE, timeout=10)
    except Exception as exc:
        logger.warning("NSE cookie pre-fetch failed: %s", exc)
    return s


def _activity_from_store(days: int) -> pd.DataFrame | None:
    try:
        from data.fii_dii_store import get_history

        rows = get_history(days)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date", ascending=False).reset_index(drop=True)
    except Exception:
        return None


def _fetch_fii_dii_activity_live(days: int = 30) -> pd.DataFrame:
    from data.fii_dii_store import refresh_if_needed

    refresh_if_needed()
    stored = _activity_from_store(days)
    if stored is not None and not stored.empty:
        cutoff = pd.Timestamp.today().normalize() - timedelta(days=days)
        return stored[stored["date"] >= cutoff].reset_index(drop=True)
    return pd.DataFrame(
        columns=["date", "fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]
    )


@st.cache_data(ttl=3600)
def get_fii_dii_activity(days: int = 30) -> pd.DataFrame:
    """
    FII/DII cash-market activity (₹ Cr). Reads persisted store first; live NSE fetch
    backfills the store when empty.
    """
    stored = _activity_from_store(days)
    if stored is not None and not stored.empty:
        cutoff = pd.Timestamp.today().normalize() - timedelta(days=days)
        return stored[stored["date"] >= cutoff].reset_index(drop=True)
    return _fetch_fii_dii_activity_live(days)


@st.cache_data(ttl=3600)
def get_bulk_deals(days: int = 10) -> pd.DataFrame:
    """
    Fetch bulk deals from NSE.

    Returns DataFrame with columns:
        date, symbol, client_name, buy_sell, quantity, price
    """
    try:
        session = _nse_session()
        resp = session.get(f"{_NSE_BASE}/api/bulkdeals", timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        data = raw if isinstance(raw, list) else raw.get("data", raw.get("bulkDeals", []))
        records = []
        for item in data:
            try:
                date = pd.to_datetime(
                    item.get("BD_DT_DATE", item.get("date", "")),
                    dayfirst=True, errors="coerce"
                )
                if pd.isna(date):
                    continue
                records.append(
                    {
                        "date": date.normalize(),
                        "symbol": str(item.get("BD_SYMBOL", item.get("symbol", ""))).upper().strip(),
                        "client_name": str(item.get("BD_CLIENT_NAME", item.get("clientName", ""))).strip(),
                        "buy_sell": str(item.get("BD_BUY_SELL", item.get("buySell", ""))).strip().upper(),
                        "quantity": int(float(str(item.get("BD_QTY_TRD", item.get("quantity", 0))).replace(",", "") or 0)),
                        "price": float(str(item.get("BD_TP_WATP", item.get("price", 0))).replace(",", "") or 0),
                    }
                )
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records).sort_values("date", ascending=False)
            cutoff = pd.Timestamp.today().normalize() - timedelta(days=days)
            df = df[df["date"] >= cutoff].reset_index(drop=True)
            logger.info("Bulk deals: loaded %d rows", len(df))
            return df

    except Exception as exc:
        logger.warning("NSE bulk deals endpoint failed: %s", exc)

    return pd.DataFrame(columns=["date", "symbol", "client_name", "buy_sell", "quantity", "price"])


@st.cache_data(ttl=3600)
def get_block_deals(days: int = 10) -> pd.DataFrame:
    """
    Fetch block deals from NSE.

    Returns DataFrame with columns:
        date, symbol, client_name, buy_sell, quantity, price
    """
    try:
        session = _nse_session()
        resp = session.get(f"{_NSE_BASE}/api/blockdeals", timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        data = raw if isinstance(raw, list) else raw.get("data", raw.get("blockDeals", []))
        records = []
        for item in data:
            try:
                date = pd.to_datetime(
                    item.get("BD_DT_DATE", item.get("date", "")),
                    dayfirst=True, errors="coerce"
                )
                if pd.isna(date):
                    continue
                records.append(
                    {
                        "date": date.normalize(),
                        "symbol": str(item.get("BD_SYMBOL", item.get("symbol", ""))).upper().strip(),
                        "client_name": str(item.get("BD_CLIENT_NAME", item.get("clientName", ""))).strip(),
                        "buy_sell": str(item.get("BD_BUY_SELL", item.get("buySell", ""))).strip().upper(),
                        "quantity": int(float(str(item.get("BD_QTY_TRD", item.get("quantity", 0))).replace(",", "") or 0)),
                        "price": float(str(item.get("BD_TP_WATP", item.get("price", 0))).replace(",", "") or 0),
                    }
                )
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records).sort_values("date", ascending=False)
            cutoff = pd.Timestamp.today().normalize() - timedelta(days=days)
            df = df[df["date"] >= cutoff].reset_index(drop=True)
            logger.info("Block deals: loaded %d rows", len(df))
            return df

    except Exception as exc:
        logger.warning("NSE block deals endpoint failed: %s", exc)

    return pd.DataFrame(columns=["date", "symbol", "client_name", "buy_sell", "quantity", "price"])


def _reset_derivative_stats_cache_for_tests() -> None:
  """Test helper — clears derivative-stats TTL cache."""
  global _deriv_stats_cache, _deriv_stats_cache_at, _deriv_stats_unavailable_logged
  _deriv_stats_cache = None
  _deriv_stats_cache_at = 0.0
  _deriv_stats_unavailable_logged = False


def _empty_derivative_stats(note: str) -> dict:
    return {
        "available": False,
        "index_futures_net": None,
        "index_options_net": None,
        "stock_futures_net": None,
        "stock_options_net": None,
        "total_net": None,
        "note": note,
        "source": "nse_public_api",
    }


def _parse_derivative_stats_rows(data: list) -> dict:
    """Parse NSE FII F&O positioning rows → net ₹ Cr by instrument bucket."""
    result: dict = {
        "available": True,
        "index_futures_net": 0.0,
        "index_options_net": 0.0,
        "stock_futures_net": 0.0,
        "stock_options_net": 0.0,
        "total_net": 0.0,
        "source": "nse_public_api",
    }

    for item in data:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", item.get("instrumentType", ""))).lower()

        def _val(key: str) -> float:
            v = item.get(key, 0)
            try:
                return float(str(v).replace(",", "") or 0)
            except Exception:
                return 0.0

        net = _val("netAmount")
        if net == 0.0 and ("buyAmount" in item or "sellAmount" in item):
            net = _val("buyAmount") - _val("sellAmount")
        if net == 0.0 and "netValue" in item:
            net = _val("netValue")

        if "index" in category and "future" in category:
            result["index_futures_net"] += net
        elif "index" in category and "option" in category:
            result["index_options_net"] += net
        elif "stock" in category and "future" in category:
            result["stock_futures_net"] += net
        elif "stock" in category and "option" in category:
            result["stock_options_net"] += net

    result["total_net"] = sum(
        result[k] for k in [
            "index_futures_net",
            "index_options_net",
            "stock_futures_net",
            "stock_options_net",
        ]
    )
    return result


def get_fii_derivative_stats_uncached() -> dict:
    """
    Fetch FII derivatives positioning from NSE when a public JSON feed exists.

    NSE removed the legacy ``api/fii-stats`` endpoint (404). We try
    ``merged-daily-reports?key=fiiStats`` on trading days; otherwise return
    ``available=False`` with null nets — never fabricated numbers.
    """
    global _deriv_stats_cache, _deriv_stats_cache_at, _deriv_stats_unavailable_logged

    now = time.time()
    if _deriv_stats_cache is not None and now - _deriv_stats_cache_at < _DERIV_STATS_TTL_S:
        return dict(_deriv_stats_cache)

    note = (
        "FII F&O positioning breakdown is not on a stable NSE JSON feed "
        "(legacy api/fii-stats removed). Cash-market FII/DII flows remain available."
    )
    try:
        session = _nse_session()
        resp = session.get(
            f"{_NSE_BASE}/api/merged-daily-reports?key=fiiStats",
            timeout=15,
        )
        if resp.ok:
            raw = resp.json()
            data = raw.get("data") if isinstance(raw, dict) else raw
            if isinstance(data, list) and data:
                result = _parse_derivative_stats_rows(data)
                _deriv_stats_cache = result
                _deriv_stats_cache_at = now
                logger.info(
                    "FII derivative stats fetched: total_net=%.0f Cr",
                    result["total_net"],
                )
                return dict(result)
    except Exception as exc:
        logger.debug("FII derivative stats fetch failed: %s", exc)

    if not _deriv_stats_unavailable_logged:
        logger.info(
            "FII derivative positioning unavailable from NSE public API; "
            "cash FII/DII flows still load via fiidiiTradeReact."
        )
        _deriv_stats_unavailable_logged = True

    result = _empty_derivative_stats(note)
    _deriv_stats_cache = result
    _deriv_stats_cache_at = now
    return dict(result)


@st.cache_data(ttl=3600)
def get_fii_derivative_stats() -> dict:
    return get_fii_derivative_stats_uncached()
