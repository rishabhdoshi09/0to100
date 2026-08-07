"""Capture and persist EOD option-chain snapshots for research."""
from __future__ import annotations

from datetime import date
from typing import Iterable

from options.chain_fetch import compute_max_pain, compute_pcr, fetch_option_chain
from options.eod_store import save_chain_snapshot, store_status

DEFAULT_UNDERLYINGS = ("NIFTY", "BANKNIFTY", "FINNIFTY")


def _spot_for(symbol: str) -> float | None:
    """Best-effort spot for ATM IV; never blocks capture if quotes fail."""
    try:
        from data.live_quotes import get_index_quotes, get_live_quotes

        if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"):
            q = get_index_quotes([symbol]).get(symbol) or {}
        else:
            q = get_live_quotes([symbol]).get(symbol) or {}
        price = float(q.get("price") or 0)
        return price if price > 0 else None
    except Exception:
        return None


def capture_symbol(symbol: str, *, as_of: str | date | None = None, spot: float | None = None) -> dict:
    sym = str(symbol or "").upper().strip()
    df, expiry, underlying = fetch_option_chain(sym)
    if df is None or df.empty or not expiry:
        return {
            "symbol": sym,
            "available": False,
            "saved": False,
            "message": "option chain unavailable",
        }
    rows = df.to_dict("records")
    pcr = compute_pcr(df)
    max_pain = compute_max_pain(df)
    spot_px = float(spot) if spot and spot > 0 else (
        float(underlying) if underlying else _spot_for(sym)
    )
    atm_iv = 0.0
    if spot_px and spot_px > 0 and "strike" in df.columns:
        idx = (df["strike"] - spot_px).abs().idxmin()
        row = df.loc[idx]
        ce_iv = float(row.get("ce_iv", 0) or 0)
        pe_iv = float(row.get("pe_iv", 0) or 0)
        atm_iv = round((ce_iv + pe_iv) / 2, 2) if ce_iv and pe_iv else round(max(ce_iv, pe_iv), 2)
    saved = save_chain_snapshot(
        sym,
        as_of=as_of or date.today(),
        expiry=str(expiry),
        rows=rows,
        source="nse_or_yfinance",
        pcr=pcr,
        max_pain=max_pain,
        atm_iv=atm_iv,
        spot=spot_px,
    )
    return {
        "available": True,
        "saved": True,
        **saved,
        "pcr": pcr,
        "max_pain": max_pain,
        "atm_iv": atm_iv,
        "spot": spot_px,
    }


def capture_universe(
    symbols: Iterable[str] | None = None,
    *,
    as_of: str | date | None = None,
) -> dict:
    underlyings = [str(s).upper().strip() for s in (symbols or DEFAULT_UNDERLYINGS) if str(s).strip()]
    results = []
    saved = 0
    for sym in underlyings:
        item = capture_symbol(sym, as_of=as_of)
        results.append(item)
        if item.get("saved"):
            saved += 1
    status = store_status()
    return {
        "requested": len(underlyings),
        "saved": saved,
        "failed": len(underlyings) - saved,
        "results": results,
        "store": status,
        "available": bool(status.get("available")) or saved > 0,
        "latest_as_of": status.get("latest_as_of") or "",
        "as_of": str(as_of or date.today()),
    }
