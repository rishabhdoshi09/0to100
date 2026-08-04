"""Options analytics — PCR, Max Pain, IV percentile, OI buildup."""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional


@st.cache_data(ttl=300, show_spinner=False)
def get_option_chain(symbol: str = "NIFTY") -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch option chain. Tries NSE API first, yfinance fallback."""
    from options.chain_fetch import fetch_option_chain

    df, expiry, _underlying = fetch_option_chain(symbol)
    return df, expiry


# ─────────────────────────────────────────────────────────────────────────────
# Analytical computations
# ─────────────────────────────────────────────────────────────────────────────

def nifty_options_summary() -> Optional[dict]:
    """
    One-line NIFTY options read for Pulse/JARVIS:
    {pcr, max_pain, bias, note}. None when the chain is unavailable.
    """
    try:
        from options.chain_fetch import compute_max_pain, compute_pcr, fetch_option_chain

        df, _expiry, _underlying = fetch_option_chain("NIFTY")
        if df is None or df.empty:
            return None
        pcr = compute_pcr(df)
        max_pain = compute_max_pain(df)
        if pcr >= 1.3:
            bias = "BULLISH"
            note = (f"PCR {pcr:.2f} — puts zyada likhe hain, sellers ko girne "
                    f"ki umeed NAHI (support strong)")
        elif pcr <= 0.7:
            bias = "BEARISH"
            note = f"PCR {pcr:.2f} — call writers haavi, upar resistance bhaari"
        else:
            bias = "NEUTRAL"
            note = f"PCR {pcr:.2f} — options market balanced"
        return {"pcr": pcr, "max_pain": max_pain, "bias": bias, "note": note}
    except Exception:
        return None


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
