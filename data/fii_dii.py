"""
FII/DII activity and bulk/block deal data from NSE public endpoints.
No API key required — NSE publishes this daily.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger("quantterm.fii_dii")

_NSE_BASE = "https://www.nseindia.com"
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


def get_fii_derivative_stats_uncached() -> dict:
    """
    Fetch FII derivatives positioning from NSE.

    Returns dict with keys:
        index_futures_net, index_options_net, stock_futures_net,
        stock_options_net, total_net
    All values in ₹ Crore.
    """
    try:
        session = _nse_session()
        resp = session.get(f"{_NSE_BASE}/api/fii-stats", timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        # NSE returns a list; grab the latest/aggregate row
        data = raw if isinstance(raw, list) else [raw]
        result: dict = {
            "index_futures_net": 0.0,
            "index_options_net": 0.0,
            "stock_futures_net": 0.0,
            "stock_options_net": 0.0,
            "total_net": 0.0,
        }

        for item in data:
            category = str(item.get("category", item.get("instrumentType", ""))).lower()

            def _val(key: str) -> float:
                v = item.get(key, 0)
                try:
                    return float(str(v).replace(",", "") or 0)
                except Exception:
                    return 0.0

            net = _val("netAmount") or (_val("buyAmount") - _val("sellAmount"))

            if "index" in category and "future" in category:
                result["index_futures_net"] += net
            elif "index" in category and "option" in category:
                result["index_options_net"] += net
            elif "stock" in category and "future" in category:
                result["stock_futures_net"] += net
            elif "stock" in category and "option" in category:
                result["stock_options_net"] += net

        result["total_net"] = sum(
            result[k] for k in ["index_futures_net", "index_options_net",
                                 "stock_futures_net", "stock_options_net"]
        )
        logger.info("FII derivative stats fetched: total_net=%.0f Cr", result["total_net"])
        return result

    except Exception as exc:
        logger.warning("NSE FII derivative stats endpoint failed: %s", exc)
        return {
            "index_futures_net": None,
            "index_options_net": None,
            "stock_futures_net": None,
            "stock_options_net": None,
            "total_net": None,
        }


@st.cache_data(ttl=3600)
def get_fii_derivative_stats() -> dict:
    return get_fii_derivative_stats_uncached()
