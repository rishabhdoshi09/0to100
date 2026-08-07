"""
📊 Market breadth — index ke PEECHHE ki sachchai, apne hi scan se muft.

Nifty upar dikh sakta hai jab 5 heavyweight use utha rahe hon aur baaki
1,900 stocks gir rahe hon — wahi NARROW market hai jahan breakouts sabse
zyada fail hote hain. Institutional desks isliye breadth dekhte hain:
advance/decline, % of stocks above 50/200-DMA.

Data is gold — aur ye gold humare paas PEHLE SE tha: har 15-min ka
full-market scan ~2,000 stocks ka change_pct + trend flags compute karta
hai. Ye module bas usse nichodta hai — ZERO extra fetch, zero API load.

Evidence-gated: chhota sample (partial scan) → koi verdict nahi.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)

MIN_N = 300                    # itne stocks se kam = poora market nahi dikha


def breadth_from_results(results: list[dict]) -> dict:
    """Scan-store rows → breadth read.
    {n, advancers, decliners, adv_ratio, pct_above_50, pct_above_200,
     verdict: HEALTHY|MIXED|NARROW|"", line}
    Empty verdict below MIN_N (no claim on a partial picture)."""
    rows = [r for r in (results or []) if r.get("change_pct") is not None]
    n = len(rows)
    if n < MIN_N:
        return {"n": n, "advancers": 0, "decliners": 0, "adv_ratio": 0.0,
                "pct_above_50": 0.0, "pct_above_200": 0.0,
                "verdict": "", "line": ""}
    adv = sum(1 for r in rows if float(r["change_pct"]) > 0)
    dec = sum(1 for r in rows if float(r["change_pct"]) < 0)
    adv_ratio = round(adv / dec, 2) if dec else 99.0
    p50 = round(sum(1 for r in rows if r.get("above_sma50")) / n * 100, 1)
    p200 = round(sum(1 for r in rows if r.get("above_sma200")) / n * 100, 1)

    if adv_ratio >= 1.2 and p50 >= 55:
        verdict = "HEALTHY"
        line = (f"Market andar se sehatmand — {adv}:{dec} advance/decline, "
                f"{p50:.0f}% stocks 50-DMA ke upar. Breakouts ko hawa milegi.")
    elif adv_ratio < 0.8 or p50 < 40:
        verdict = "NARROW"
        line = (f"⚠️ Market andar se kamzor — {adv}:{dec} adv/decl, sirf "
                f"{p50:.0f}% stocks 50-DMA upar. Index bhale theek dikhe, "
                f"breakouts yahan fail hote hain.")
    else:
        verdict = "MIXED"
        line = (f"Breadth mixed — {adv}:{dec} adv/decl, {p50:.0f}% above "
                f"50-DMA. Selective raho.")
    return {"n": n, "advancers": adv, "decliners": dec,
            "adv_ratio": adv_ratio, "pct_above_50": p50,
            "pct_above_200": p200, "verdict": verdict, "line": line}


def breadth_from_cache() -> dict:
    """Breadth over the FULL bulk OHLCV cache — every analyzed stock, not
    just the signal-bearing winners. (Scan store sirf signal-waale rakhta
    hai — woh biased sample hai; asli breadth poore market se aati hai.)
    Zero network — cache pehle se bhara hota hai scan ke waqt."""
    rows: list[dict] = []
    try:
        import numpy as np
        from scan.bulk_fetcher import cached_symbols, get_cached
        for sym in cached_symbols():
            try:
                df = get_cached(sym)
                if df is None or len(df) < 51:
                    continue
                close = df["close"].values.astype(float)
                chg = ((close[-1] / close[-2]) - 1) * 100 if close[-2] else 0.0
                rows.append({
                    "change_pct": chg,
                    "above_sma50": bool(close[-1] > close[-50:].mean()),
                    "above_sma200": bool(len(close) >= 200
                                         and close[-1] > close[-200:].mean()),
                })
            except Exception:
                continue
    except Exception as exc:
        log.debug("breadth_cache_failed", error=str(exc))
    return breadth_from_results(rows)
