"""Option chain fetch without Streamlit — NSE public API first, yfinance last resort.

NSE migrated from ``/api/option-chain-indices`` / ``equities`` to
``/api/option-chain-v3`` + ``/api/option-chain-contract-info`` (expiry required).
Legacy endpoints now 404; this module prefers v3 and falls back honestly.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import pandas as pd

_WS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_WS_TTL_S = 90
_WS_FAIL_S: dict[str, float] = {}
# Background warm failures must not lock the F&O Desk for 5 minutes.
_WS_FAIL_BACKOFF_S = 45

_INDEX_SYMBOLS = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"})


def fetch_option_chain(symbol: str = "NIFTY") -> tuple[Optional[pd.DataFrame], Optional[str]]:
    sym = str(symbol or "NIFTY").upper().strip()
    df, expiry = _fetch_nse(sym)
    if df is not None and not df.empty:
        return df, expiry
    return _fetch_yfinance(sym)


def _nse_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
        "X-Requested-With": "XMLHttpRequest",
    }


def _prime_nse_session(session) -> None:
    headers = _nse_headers()
    # option-chain page is more reliable for cookies than the homepage (often 403).
    session.get("https://www.nseindia.com/option-chain", headers=headers, timeout=10)
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
    except Exception:
        pass


def _rows_from_records(data: dict[str, Any], expiry: str | None) -> tuple[list[dict], str | None]:
    """Normalise legacy + v3 NSE option-chain JSON into strike rows."""
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, dict):
        return [], None
    raw_rows = records.get("data") or []
    expiry_dates = [str(x) for x in (records.get("expiryDates") or []) if x]
    chosen = str(expiry or (expiry_dates[0] if expiry_dates else "") or "")
    if not chosen and raw_rows and isinstance(raw_rows[0], dict):
        chosen = str(raw_rows[0].get("expiryDates") or raw_rows[0].get("expiryDate") or "")
    out: list[dict] = []
    for rec in raw_rows:
        if not isinstance(rec, dict):
            continue
        row_expiry = str(rec.get("expiryDates") or rec.get("expiryDate") or "")
        # v3 requests already scope by expiry; still filter when multiple expiries appear.
        if chosen and row_expiry and row_expiry != chosen:
            continue
        strike = rec.get("strikePrice")
        if strike is None:
            continue
        ce = rec.get("CE") if isinstance(rec.get("CE"), dict) else {}
        pe = rec.get("PE") if isinstance(rec.get("PE"), dict) else {}
        out.append(
            {
                "strike": strike,
                "ce_oi": ce.get("openInterest", 0) or 0,
                "ce_coi": ce.get("changeinOpenInterest", 0) or 0,
                "ce_iv": ce.get("impliedVolatility", 0) or 0,
                "ce_ltp": ce.get("lastPrice", 0) or 0,
                "ce_volume": ce.get("totalTradedVolume", 0) or 0,
                "pe_oi": pe.get("openInterest", 0) or 0,
                "pe_coi": pe.get("changeinOpenInterest", 0) or 0,
                "pe_iv": pe.get("impliedVolatility", 0) or 0,
                "pe_ltp": pe.get("lastPrice", 0) or 0,
                "pe_volume": pe.get("totalTradedVolume", 0) or 0,
            }
        )
    return out, (chosen or None)


def _fetch_nse_v3(session, symbol: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Current NSE option-chain API (requires nearest expiry from contract-info)."""
    headers = _nse_headers()
    info_resp = session.get(
        "https://www.nseindia.com/api/option-chain-contract-info",
        params={"symbol": symbol},
        headers=headers,
        timeout=12,
    )
    if info_resp.status_code != 200 or not info_resp.content:
        return None, None
    try:
        info = info_resp.json()
    except Exception:
        return None, None
    expiries = [str(x) for x in (info.get("expiryDates") or []) if x]
    if not expiries:
        return None, None
    expiry = expiries[0]
    chain_type = "Indices" if symbol in _INDEX_SYMBOLS else "Equity"
    data = None
    for attempt in range(3):
        time.sleep(0.4 * (attempt + 1))
        resp = session.get(
            "https://www.nseindia.com/api/option-chain-v3",
            params={"type": chain_type, "symbol": symbol, "expiry": expiry},
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200 and resp.content:
            try:
                payload = resp.json()
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("records"):
                data = payload
                break
        if resp.status_code in (401, 403):
            _prime_nse_session(session)
    if data is None:
        return None, None
    rows, chosen = _rows_from_records(data, expiry)
    if not rows:
        return None, None
    return pd.DataFrame(rows), chosen or expiry


def _fetch_nse_legacy(session, symbol: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Legacy indices/equities endpoints — kept as a short fallback while NSE transitions."""
    headers = _nse_headers()
    if symbol in _INDEX_SYMBOLS:
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    else:
        session.get(
            f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}",
            headers=headers,
            timeout=8,
        )
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    data = None
    for attempt in range(2):
        time.sleep(0.5 * (attempt + 1))
        resp = session.get(url, headers=headers, timeout=12)
        if resp.status_code == 200 and resp.content:
            try:
                payload = resp.json()
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("records"):
                data = payload
                break
        if resp.status_code in (401, 403):
            _prime_nse_session(session)
        if resp.status_code == 404:
            break
    if data is None:
        return None, None
    expiry_dates = [str(x) for x in ((data.get("records") or {}).get("expiryDates") or []) if x]
    expiry = expiry_dates[0] if expiry_dates else None
    rows, chosen = _rows_from_records(data, expiry)
    if not rows:
        return None, None
    return pd.DataFrame(rows), chosen or expiry


def _fetch_nse(symbol: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        import requests

        session = requests.Session()
        _prime_nse_session(session)
        df, expiry = _fetch_nse_v3(session, symbol)
        if df is not None and not df.empty:
            return df, expiry
        return _fetch_nse_legacy(session, symbol)
    except Exception:
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
                "Option chain unavailable from NSE (v3 API / cookie session). "
                "Click Refresh live chain — Yahoo Finance may still load some strikes. "
                "This is not a buy/sell signal desk."
            ),
            "source": "unavailable",
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
        ce_iv = float(row.get("ce_iv", 0) or 0)
        pe_iv = float(row.get("pe_iv", 0) or 0)
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
    total_ce_oi = float(df["ce_oi"].sum()) if "ce_oi" in df.columns else 0.0
    total_pe_oi = float(df["pe_oi"].sum()) if "pe_oi" in df.columns else 0.0
    iv_rank = 0.0
    try:
        all_iv = pd.concat([df["ce_iv"], df["pe_iv"]]).replace(0, np.nan).dropna()
        if not all_iv.empty and float(all_iv.max()) > float(all_iv.min()):
            probe = atm_iv if atm_iv > 0 else float(all_iv.median())
            iv_rank = round(
                (probe - float(all_iv.min())) / (float(all_iv.max()) - float(all_iv.min())) * 100,
                1,
            )
    except Exception:
        iv_rank = 0.0
    return {
        "available": True,
        "symbol": sym,
        "expiry": expiry,
        "pcr": pcr,
        "max_pain": max_pain,
        "bias": bias,
        "note": note,
        "atm_iv": atm_iv,
        "iv_rank": iv_rank,
        "spot": spot,
        "total_ce_oi": int(total_ce_oi),
        "total_pe_oi": int(total_pe_oi),
        "strike_count": int(len(df)),
        "top_call_oi": top_ce,
        "top_put_oi": top_pe,
        "chain": chain_rows,
        "greeks_available": False,
        "signal_desk": False,
        "source": "nse",
        "honesty": (
            "Live chain shows OI, IV, PCR and max pain for the nearest expiry. "
            "Black-Scholes Greeks and trade direction are not calculated — this is context, not a signal."
        ),
    }


def chain_workspace_memory_only(symbol: str = "NIFTY") -> dict[str, Any]:
    """Return an in-memory chain if present; never fetch. Dashboard-safe."""
    sym = str(symbol or "NIFTY").upper().strip()
    cached = _WS_CACHE.get(sym)
    if cached:
        return dict(cached[1])
    return {
        "available": False,
        "symbol": sym,
        "message": "Option chain not cached yet; dashboard skips live NSE fetch.",
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
                "message": (
                    "Option chain temporarily unavailable after a recent fetch miss "
                    f"(retry in {max(1, int(_WS_FAIL_BACKOFF_S - (now - fail_ts)))}s, "
                    "or click Refresh live chain)."
                ),
            }
    result = chain_workspace(sym, spot=spot)
    if result.get("available"):
        _WS_CACHE[sym] = (now, result)
        _WS_FAIL_S.pop(sym, None)
    else:
        _WS_FAIL_S[sym] = now
    return result
