"""Options analytics — PCR, Max Pain, IV percentile, OI buildup."""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional


@st.cache_data(ttl=300, show_spinner=False)
def get_option_chain(symbol: str = "NIFTY") -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch option chain. Tries NSE API first, yfinance fallback."""
    # ── Attempt 1: NSE public API ─────────────────────────────────────────────
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/option-chain",
        }
        session = requests.Session()
        # Seed cookies first
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        session.get(
            "https://www.nseindia.com/option-chain", headers=headers, timeout=8
        )

        # Index vs equity URL
        if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"):
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

        resp = session.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        data = resp.json()

        records = data["records"]["data"]
        expiry = data["records"]["expiryDates"][0]  # nearest expiry
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

    # ── Attempt 2: nsepython ──────────────────────────────────────────────────
    try:
        from nsepython import nse_optionchain_scrapper  # type: ignore

        raw = nse_optionchain_scrapper(symbol)
        records = raw["records"]["data"]
        expiry = raw["records"]["expiryDates"][0]
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

    # ── Attempt 3: yfinance fallback ──────────────────────────────────────────
    try:
        import yfinance as yf

        _YF_MAP = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
        }
        yf_sym = _YF_MAP.get(symbol, f"{symbol}.NS")
        tk = yf.Ticker(yf_sym)
        exps = tk.options
        if not exps:
            return None, None
        expiry = exps[0]
        chain = tk.option_chain(expiry)
        calls = chain.calls[["strike", "openInterest", "impliedVolatility", "lastPrice", "volume"]].copy()
        puts  = chain.puts[["strike", "openInterest", "impliedVolatility", "lastPrice", "volume"]].copy()

        calls.columns = ["strike", "ce_oi", "ce_iv", "ce_ltp", "ce_volume"]
        puts.columns  = ["strike", "pe_oi", "pe_iv", "pe_ltp", "pe_volume"]
        # yfinance IV is a decimal — convert to percentage
        calls["ce_iv"] = (calls["ce_iv"] * 100).round(2)
        puts["pe_iv"]  = (puts["pe_iv"]  * 100).round(2)

        df = pd.merge(calls, puts, on="strike", how="outer").fillna(0)
        df["ce_coi"] = 0
        df["pe_coi"] = 0
        df = df.sort_values("strike").reset_index(drop=True)
        return df, expiry
    except Exception:
        pass

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Analytical computations
# ─────────────────────────────────────────────────────────────────────────────

def compute_pcr(df: pd.DataFrame) -> float:
    """Put-Call Ratio by OI."""
    if df is None or df.empty:
        return 1.0
    total_pe = df["pe_oi"].sum()
    total_ce = df["ce_oi"].sum()
    return round(total_pe / total_ce, 2) if total_ce > 0 else 1.0


def compute_max_pain(df: pd.DataFrame) -> float:
    """Max Pain = strike where total option sellers' loss is minimised."""
    if df is None or df.empty:
        return 0.0
    strikes = df["strike"].values
    ce_oi   = df["ce_oi"].values
    pe_oi   = df["pe_oi"].values
    losses  = []
    for s in strikes:
        ce_loss = ((strikes - s).clip(min=0) * ce_oi).sum()
        pe_loss = ((s - strikes).clip(min=0) * pe_oi).sum()
        losses.append(ce_loss + pe_loss)
    return float(strikes[np.argmin(losses)])


def get_atm_iv(df: pd.DataFrame, spot: float) -> float:
    """Return average IV of the nearest ATM strike (CE + PE average)."""
    if df is None or df.empty or spot <= 0:
        return 0.0
    idx = (df["strike"] - spot).abs().idxmin()
    row = df.loc[idx]
    ce_iv = float(row.get("ce_iv", 0))
    pe_iv = float(row.get("pe_iv", 0))
    if ce_iv > 0 and pe_iv > 0:
        return round((ce_iv + pe_iv) / 2, 2)
    return round(max(ce_iv, pe_iv), 2)


def get_oi_buildup(df: pd.DataFrame, spot: float) -> dict:
    """Find strikes with highest OI buildup near ATM (±10%)."""
    if df is None or df.empty:
        return {}
    atm_range = df[
        (df["strike"] >= spot * 0.90) & (df["strike"] <= spot * 1.10)
    ]
    top_ce = (
        atm_range.nlargest(3, "ce_oi")[["strike", "ce_oi"]].to_dict("records")
    )
    top_pe = (
        atm_range.nlargest(3, "pe_oi")[["strike", "pe_oi"]].to_dict("records")
    )
    return {"resistance_levels": top_ce, "support_levels": top_pe}


def get_iv_percentile(df: pd.DataFrame) -> float:
    """IV Rank — where current ATM IV sits vs all strikes' IV range (0-100)."""
    if df is None or df.empty:
        return 0.0
    all_iv = pd.concat([df["ce_iv"], df["pe_iv"]]).replace(0, np.nan).dropna()
    if all_iv.empty:
        return 0.0
    iv_min, iv_max = all_iv.min(), all_iv.max()
    iv_now = all_iv.median()
    if iv_max == iv_min:
        return 50.0
    return round((iv_now - iv_min) / (iv_max - iv_min) * 100, 1)
