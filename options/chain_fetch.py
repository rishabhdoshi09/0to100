"""Option chain fetch without Streamlit — NSE public API first, yfinance last resort."""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

_WS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_WS_TTL_S = 300
_WS_FAIL_S: dict[str, float] = {}
_WS_FAIL_BACKOFF_S = 300


def fetch_option_chain(symbol: str = "NIFTY") -> tuple[Optional[pd.DataFrame], Optional[str]]:
    sym = str(symbol or "NIFTY").upper().strip()
    df, expiry = _fetch_nse(sym)
    if df is not None and not df.empty:
        return df, expiry
    return _fetch_yfinance(sym)


def _fetch_nse(symbol: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nseindia.com/option-chain",
        }
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        session.get("https://www.nseindia.com/option-chain", headers=headers, timeout=8)
        if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"):
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            session.get(
                f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}",
                headers=headers,
                timeout=8,
            )
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        data = None
        for attempt in range(3):
            time.sleep(0.6 * (attempt + 1))
            resp = session.get(url, headers=headers, timeout=12)
            if resp.status_code == 200 and resp.content:
                try:
                    data = resp.json()
                    break
                except Exception:
                    data = None
            if resp.status_code in (401, 403):
                session.get("https://www.nseindia.com", headers=headers, timeout=8)
        if data is None:
            return None, None
        records = data["records"]["data"]
        expiry = data["records"]["expiryDates"][0]
        rows = []
        for rec in records:
            if rec.get("expiryDate") != expiry:
                continue
            strike = rec["strikePrice"]
            ce = rec.get("CE", {})
            pe = rec.get("PE", {})
            rows.append(
                {
                    "strike": strike,
                    "ce_oi": ce.get("openInterest", 0),
                    "ce_coi": ce.get("changeinOpenInterest", 0),
                    "ce_iv": ce.get("impliedVolatility", 0),
                    "ce_ltp": ce.get("lastPrice", 0),
                    "ce_volume": ce.get("totalTradedVolume", 0),
                    "pe_oi": pe.get("openInterest", 0),
                    "pe_coi": pe.get("changeinOpenInterest", 0),
                    "pe_iv": pe.get("impliedVolatility", 0),
                    "pe_ltp": pe.get("lastPrice", 0),
                    "pe_volume": pe.get("totalTradedVolume", 0),
                }
            )
        if rows:
            return pd.DataFrame(rows), expiry
    except Exception:
        pass
    return None, None


def _fetch_yfinance(symbol: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        import yfinance as yf

        yf_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        }
        yf_sym = yf_map.get(symbol, f"{symbol}.NS")
        tk = yf.Ticker(yf_sym)
        exps = tk.options
        if not exps:
            return None, None
        expiry = exps[0]
        chain = tk.option_chain(expiry)
        calls = chain.calls[["strike", "openInterest", "impliedVolatility", "lastPrice", "volume"]].copy()
        puts = chain.puts[["strike", "openInterest", "impliedVolatility", "lastPrice", "volume"]].copy()
        calls.columns = ["strike", "ce_oi", "ce_iv", "ce_ltp", "ce_volume"]
        puts.columns = ["strike", "pe_oi", "pe_iv", "pe_ltp", "pe_volume"]
        calls["ce_iv"] = (calls["ce_iv"] * 100).round(2)
        puts["pe_iv"] = (puts["pe_iv"] * 100).round(2)
        df = pd.merge(calls, puts, on="strike", how="outer").fillna(0)
        df["ce_coi"] = 0
        df["pe_coi"] = 0
        return df.sort_values("strike").reset_index(drop=True), expiry
    except Exception:
        return None, None


def compute_pcr(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 1.0
    total_pe = df["pe_oi"].sum()
    total_ce = df["ce_oi"].sum()
    return round(total_pe / total_ce, 2) if total_ce > 0 else 1.0


def compute_max_pain(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    strikes = df["strike"].values
    ce_oi = df["ce_oi"].values
    pe_oi = df["pe_oi"].values
    losses = []
    for s in strikes:
        ce_loss = ((strikes - s).clip(min=0) * ce_oi).sum()
        pe_loss = ((s - strikes).clip(min=0) * pe_oi).sum()
        losses.append(ce_loss + pe_loss)
    return float(strikes[np.argmin(losses)])


def chain_workspace(symbol: str, spot: float | None = None) -> dict[str, Any]:
    sym = str(symbol or "NIFTY").upper().strip()
    df, expiry = fetch_option_chain(sym)
    if df is None or df.empty:
        return {
            "available": False,
            "symbol": sym,
            "message": (
                "Option chain unavailable from NSE (often blocked off-hours/weekends). "
                "Use Retry — Yahoo Finance fallback may still load strikes."
            ),
        }
    pcr = compute_pcr(df)
    max_pain = compute_max_pain(df)
    if pcr >= 1.3:
        bias, note = "BULLISH", f"PCR {pcr:.2f} — put OI dominates (support below)"
    elif pcr <= 0.7:
        bias, note = "BEARISH", f"PCR {pcr:.2f} — call OI dominates (resistance above)"
    else:
        bias, note = "NEUTRAL", f"PCR {pcr:.2f} — balanced options positioning"
    atm_iv = 0.0
    if spot and spot > 0:
        idx = (df["strike"] - spot).abs().idxmin()
        row = df.loc[idx]
        ce_iv = float(row.get("ce_iv", 0))
        pe_iv = float(row.get("pe_iv", 0))
        if ce_iv > 0 and pe_iv > 0:
            atm_iv = round((ce_iv + pe_iv) / 2, 2)
        else:
            atm_iv = round(max(ce_iv, pe_iv), 2)
    strikes = df.sort_values("strike")
    top_ce = strikes.nlargest(5, "ce_oi")[["strike", "ce_oi", "ce_coi"]].to_dict("records")
    top_pe = strikes.nlargest(5, "pe_oi")[["strike", "pe_oi", "pe_coi"]].to_dict("records")
    chain_rows = strikes.to_dict("records")
    if len(chain_rows) > 80:
        mid = len(chain_rows) // 2
        chain_rows = chain_rows[max(0, mid - 40): mid + 40]
    return {
        "available": True,
        "symbol": sym,
        "expiry": expiry,
        "pcr": pcr,
        "max_pain": max_pain,
        "bias": bias,
        "note": note,
        "atm_iv": atm_iv,
        "spot": spot,
        "top_call_oi": top_ce,
        "top_put_oi": top_pe,
        "chain": chain_rows,
    }


def chain_workspace_cached(
    symbol: str,
    spot: float | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """TTL-cached workspace payload — avoids NSE hammering on dashboard polls."""
    sym = str(symbol or "NIFTY").upper().strip()
    now = time.time()
    if not force:
        cached = _WS_CACHE.get(sym)
        if cached and (now - cached[0]) < _WS_TTL_S:
            return cached[1]
        fail_ts = _WS_FAIL_S.get(sym)
        if fail_ts and (now - fail_ts) < _WS_FAIL_BACKOFF_S:
            return {
                "available": False,
                "symbol": sym,
                "message": "Option chain temporarily unavailable; retry shortly.",
            }
    result = chain_workspace(sym, spot=spot)
    if result.get("available"):
        _WS_CACHE[sym] = (now, result)
        _WS_FAIL_S.pop(sym, None)
    else:
        _WS_FAIL_S[sym] = now
    return result
