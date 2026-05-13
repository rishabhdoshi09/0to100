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


@st.cache_data(ttl=3600)
def get_fii_dii_activity(days: int = 30) -> pd.DataFrame:
    """
    Fetch FII/DII cash-market activity from NSE.

    Returns DataFrame with columns:
        date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net
    All monetary values are in ₹ Crore.
    """
    # ── Primary: NSE fiidiiTradeReact endpoint ────────────────────────────────
    try:
        session = _nse_session()
        resp = session.get(
            f"{_NSE_BASE}/api/fiidiiTradeReact",
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()

        records = []
        for item in raw:
            try:
                date_str = item.get("date", "")
                date = pd.to_datetime(date_str, dayfirst=True, errors="coerce")
                if pd.isna(date):
                    continue
                records.append(
                    {
                        "date": date.normalize(),
                        "fii_buy": float(str(item.get("fiiBuyValue", 0)).replace(",", "") or 0),
                        "fii_sell": float(str(item.get("fiiSellValue", 0)).replace(",", "") or 0),
                        "fii_net": float(str(item.get("fiiNetValue", 0)).replace(",", "") or 0),
                        "dii_buy": float(str(item.get("diiBuyValue", 0)).replace(",", "") or 0),
                        "dii_sell": float(str(item.get("diiSellValue", 0)).replace(",", "") or 0),
                        "dii_net": float(str(item.get("diiNetValue", 0)).replace(",", "") or 0),
                    }
                )
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records).sort_values("date", ascending=False)
            cutoff = pd.Timestamp.today().normalize() - timedelta(days=days)
            df = df[df["date"] >= cutoff].reset_index(drop=True)
            logger.info("FII/DII: loaded %d rows from NSE primary endpoint", len(df))
            return df

    except Exception as exc:
        logger.warning("NSE FII/DII primary endpoint failed: %s", exc)

    # ── Backup: NSE archives XLS pattern ─────────────────────────────────────
    records = []
    for delta in range(days):
        dt = datetime.today() - timedelta(days=delta)
        if dt.weekday() >= 5:  # skip weekends
            continue
        date_str = dt.strftime("%d%m%Y")
        url = f"https://archives.nseindia.com/content/fo/fii_stats_{date_str}.xls"
        try:
            r = requests.get(url, headers=_NSE_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            xls = pd.read_excel(r.content, header=None)
            # NSE archive XLS structure: row 3-5 typically has FII/DII rows
            fii_row = xls[xls.iloc[:, 0].astype(str).str.contains("FII", na=False)]
            dii_row = xls[xls.iloc[:, 0].astype(str).str.contains("DII", na=False)]
            if fii_row.empty or dii_row.empty:
                continue
            fv = fii_row.iloc[0]
            dv = dii_row.iloc[0]
            records.append(
                {
                    "date": pd.Timestamp(dt.date()),
                    "fii_buy": float(str(fv.iloc[1]).replace(",", "") or 0),
                    "fii_sell": float(str(fv.iloc[2]).replace(",", "") or 0),
                    "fii_net": float(str(fv.iloc[3]).replace(",", "") or 0),
                    "dii_buy": float(str(dv.iloc[1]).replace(",", "") or 0),
                    "dii_sell": float(str(dv.iloc[2]).replace(",", "") or 0),
                    "dii_net": float(str(dv.iloc[3]).replace(",", "") or 0),
                }
            )
        except Exception:
            continue

    if records:
        df = pd.DataFrame(records).sort_values("date", ascending=False).reset_index(drop=True)
        logger.info("FII/DII: loaded %d rows from NSE archive backup", len(df))
        return df

    logger.error("FII/DII: all data sources failed — returning empty DataFrame")
    return pd.DataFrame(
        columns=["date", "fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]
    )


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


@st.cache_data(ttl=3600)
def get_fii_derivative_stats() -> dict:
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
