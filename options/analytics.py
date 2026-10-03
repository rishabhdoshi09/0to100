"""Options analytics — PCR, Max Pain, IV percentile, OI buildup."""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


_KITE_UNDERLYING_KEYS = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "NIFTYNXT50": "NSE:NIFTY NEXT 50",
}


def _kite_option_chain(symbol: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Nearest-expiry chain built only from Zerodha Kite market-data reads."""
    from data.nfo_market import (
        NfoMarketDataClient,
        option_instruments,
        quote_to_option_contract,
        read_market_quotes,
        read_nfo_instruments,
        read_nfo_quotes,
    )

    wanted = str(symbol or "").strip().upper()
    client = NfoMarketDataClient.from_config()
    instruments = read_nfo_instruments(client)
    contracts = option_instruments(instruments, wanted, max_expiries=1)
    if not contracts:
        return None, None

    underlying_key = _KITE_UNDERLYING_KEYS.get(wanted, f"NSE:{wanted}")
    spot_rows = read_market_quotes([underlying_key], client=client)
    spot = float((spot_rows.get(underlying_key) or {}).get("last_price") or 0.0)
    if spot <= 0:
        return None, None

    symbols = [
        str(row.get("tradingsymbol") or "")
        for row in contracts
        if row.get("tradingsymbol")
    ]
    quotes = read_nfo_quotes(symbols, client=client)
    normalized = []
    for meta in contracts:
        tradingsymbol = str(meta.get("tradingsymbol") or "")
        quote = quotes.get(tradingsymbol)
        if not isinstance(quote, dict):
            continue
        normalized.append(
            quote_to_option_contract(meta, quote, spot=spot)
        )
    if not normalized:
        return None, None

    expiry = str(normalized[0].get("expiry") or "")
    by_strike: dict[float, dict] = {}
    for row in normalized:
        strike = float(row.get("strike") or 0.0)
        if strike <= 0:
            continue
        bucket = by_strike.setdefault(
            strike,
            {
                "strike": strike,
                "ce_oi": 0,
                "ce_coi": 0,
                "ce_iv": 0.0,
                "ce_ltp": 0.0,
                "ce_volume": 0,
                "pe_oi": 0,
                "pe_coi": 0,
                "pe_iv": 0.0,
                "pe_ltp": 0.0,
                "pe_volume": 0,
                "source": "ZERODHA_KITE_NFO_READ_ONLY",
            },
        )
        side = str(row.get("option_type") or "").upper()
        prefix = "ce" if side == "CE" else "pe" if side == "PE" else ""
        if not prefix:
            continue
        bucket[f"{prefix}_oi"] = int(row.get("oi") or 0)
        bucket[f"{prefix}_iv"] = float(row.get("iv") or 0.0)
        bucket[f"{prefix}_ltp"] = float(row.get("ltp") or 0.0)
        bucket[f"{prefix}_volume"] = int(row.get("volume") or 0)

    if not by_strike:
        return None, None
    df = pd.DataFrame([by_strike[strike] for strike in sorted(by_strike)])
    df.attrs["source"] = "ZERODHA_KITE_NFO_READ_ONLY"
    df.attrs["spot"] = spot
    df.attrs["iv_source"] = "IMPLIED_FROM_KITE_MARKET_QUOTES"
    df.attrs["coi_source"] = "UNAVAILABLE_FROM_KITE_FULL_QUOTE"
    return df, expiry


def get_option_chain(symbol: str = "NIFTY") -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch the nearest option chain under the Kite-authoritative policy.

    When Kite is configured, this function uses only Zerodha Kite instrument
    metadata + full NFO quotes. NSE/yfinance are continuity fallbacks only when
    Kite credentials are unavailable.
    """
    try:
        from data.kite_client import kite_credentials_available
        kite_authoritative = bool(kite_credentials_available())
    except Exception:
        kite_authoritative = False
    if kite_authoritative:
        try:
            return _kite_option_chain(symbol)
        except Exception:
            return None, None
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
        import time as _time

        session = requests.Session()
        # Seed cookies first (NSE rejects cookieless API hits), then let them
        # settle — a hit immediately after priming often still 401s.
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        session.get(
            "https://www.nseindia.com/option-chain", headers=headers, timeout=8
        )

        # Index vs equity URL
        if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"):
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

        # NSE frequently blocks the first hit and yields on a retry — try a few
        # times with backoff, re-priming cookies if the session gets rejected.
        data = None
        for _attempt in range(3):
            _time.sleep(0.6 * (_attempt + 1))
            resp = session.get(url, headers=headers, timeout=12)
            if resp.status_code == 200 and resp.content:
                try:
                    data = resp.json()
                    break
                except Exception:
                    data = None
            if resp.status_code in (401, 403):        # cookies stale → re-prime
                session.get("https://www.nseindia.com", headers=headers, timeout=8)
        if data is None:
            raise RuntimeError("NSE option-chain API unavailable after retries")

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

def nifty_options_summary() -> Optional[dict]:
    """
    One-line NIFTY options read for Pulse/JARVIS:
    {pcr, max_pain, bias, note}. None when the chain is unavailable.
    """
    try:
        df, _expiry = get_option_chain("NIFTY")
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
