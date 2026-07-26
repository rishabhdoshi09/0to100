"""
🔎 Stock Research Report — one call that pulls EVERYTHING the system knows about
any symbol, so the search bar can research any 'x' stock on demand.

Composition only — it queries the existing resources and assembles them into one
structured report; it computes no new statistics. Every section is fail-open, so
a down feed or a US symbol (where the NSE-only Research OS is silent) yields a
partial report, never an error. Sections:

  • quote        — price, day change, 52-week range
  • setup        — the scanner's verdict, signals, entry/stop/target, R:R
  • sizing       — what the 1% rule would buy, and the max loss in rupees
  • why          — the plain-English case (why buy / why not / evidence / trust /
                   similar history) from the explainability layer
  • strategy_health — is THIS stock's kind of setup currently decaying? (drift)
  • technicals   — RSI, ATR%, trend flags
  • fundamentals — the headline numbers
  • context      — sector strength, market mood, market health
"""
from __future__ import annotations


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _quote(symbol: str, is_us: bool) -> dict:
    q = {}
    if is_us:
        try:
            import yfinance as yf
            fi = yf.Ticker(symbol).fast_info
            q = {"price": float(getattr(fi, "last_price", 0) or 0),
                 "prev": float(getattr(fi, "previous_close", 0) or 0),
                 "week52_high": float(getattr(fi, "year_high", 0) or 0),
                 "week52_low": float(getattr(fi, "year_low", 0) or 0)}
        except Exception:
            return {}
    else:
        try:
            from data.live_quotes import get_live_quotes
            r = (get_live_quotes([symbol]) or {}).get(symbol, {})
            q["price"] = float(r.get("price") or 0)
        except Exception:
            pass
        try:
            import pandas as pd  # noqa
            from data.bhavcopy_store import get_ohlcv
            df = get_ohlcv(symbol)
            if df is not None and len(df):
                if not q.get("price"):
                    q["price"] = float(df["close"].iloc[-1])
                q["prev"] = float(df["close"].iloc[-2]) if len(df) > 1 else q.get("price", 0)
                q["week52_high"] = float(df["high"].tail(252).max())
                q["week52_low"] = float(df["low"].tail(252).min())
        except Exception:
            pass
    if q.get("price") and q.get("prev"):
        q["change_pct"] = round((q["price"] - q["prev"]) / q["prev"] * 100, 2)
    return q


def _setup(symbol: str, is_us: bool):
    """Run the scanner on the symbol → its StockSignal (or None)."""
    from scan.unified_scanner import UnifiedScanner
    if is_us:
        from data.us_data import get_us_daily, sp500_return_30d
        df = get_us_daily(symbol)
        bench = sp500_return_30d()
    else:
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(symbol)
        try:
            from scan.unified_scanner import _nifty_return_30d
            bench = _nifty_return_30d()
        except Exception:
            bench = 0.0
    if df is None or len(df) < 60:
        return None
    sc = UnifiedScanner()
    sc._nifty_ret30 = bench
    return sc._analyze(symbol, df)


def _strategy_health(signal_keys: list[str]) -> list[dict]:
    """For the setups THIS stock is showing, is that kind of edge decaying,
    recovering, or stable right now? (Research OS drift — NSE only.)"""
    try:
        from research.drift import drift_report
        flagged = {d["signal"]: d for d in drift_report()}
    except Exception:
        return []
    out = []
    for k in signal_keys:
        d = flagged.get(k)
        if d:
            out.append({"signal": k, "status": d.get("status"),
                        "confidence": d.get("confidence"),
                        "insight": d.get("insight", "")})
    return out


def research_stock(symbol: str, is_us: bool = False,
                   capital: float = 100_000.0) -> dict:
    """The full, fail-open research report for one symbol."""
    symbol = (symbol or "").upper().strip()
    report: dict = {"symbol": symbol, "is_us": is_us,
                    "cur": "$" if is_us else "₹"}
    report["quote"] = _safe(lambda: _quote(symbol, is_us), {}) or {}

    r = _safe(lambda: _setup(symbol, is_us))
    if r is not None:
        rr = _safe(lambda: getattr(r, "risk_reward", 0.0), 0.0) or 0.0
        report["setup"] = {
            "verdict": getattr(r, "verdict", ""),
            "score": round(float(getattr(r, "score", 0) or 0), 1),
            "conviction": round(float(getattr(r, "breakout_conviction", 0) or 0), 0),
            "signals": list(getattr(r, "signal_labels", []) or []),
            "signal_keys": list(getattr(r, "signals", []) or []),
            "reasons": list(getattr(r, "reasons", []) or [])[:3],
            "entry": round(float(getattr(r, "entry", 0) or 0), 1),
            "stop": round(float(getattr(r, "stop", 0) or 0), 1),
            "target": round(float(getattr(r, "target", 0) or 0), 1),
            "rr": round(rr, 1),
            "rsi": round(float(getattr(r, "rsi", 0) or 0), 1),
        }
        # sizing — what the 1% rule would buy
        report["sizing"] = _safe(lambda: __import__(
            "risk.position_sizer", fromlist=["size_position"]).size_position(
            float(getattr(r, "entry", 0) or 0), float(getattr(r, "stop", 0) or 0),
            capital=capital), {})
        # the plain-English case + similar history (explainability layer)
        report["why"] = _safe(lambda: __import__(
            "research.explainability", fromlist=["row_intelligence"]).row_intelligence(
            symbol, verdict=getattr(r, "verdict", ""),
            signal=(getattr(r, "signal_labels", []) or [""])[0],
            signals=list(getattr(r, "signal_labels", []) or []),
            features={"rsi": getattr(r, "rsi", None),
                      "quality_score": getattr(r, "score", None)}), {})
        if not is_us:
            report["strategy_health"] = _strategy_health(
                report["setup"]["signal_keys"])
    else:
        report["setup"] = {"verdict": "NO SETUP",
                           "note": "Enough clean history nahi / koi setup nahi bana."}

    # technicals + fundamentals (agent tools)
    report["technicals"] = _safe(lambda: __import__(
        "agents.tools", fromlist=["get_technical_indicators"]).get_technical_indicators(
        symbol), {}) or {}
    report["fundamentals"] = _safe(lambda: __import__(
        "agents.tools", fromlist=["get_fundamentals"]).get_fundamentals(symbol), {}) or {}

    # market context (NSE only) — sector, mood, breadth
    if not is_us:
        ctx = {}
        ctx["market_mood"] = _safe(lambda: str(getattr(__import__(
            "core.regime_engine", fromlist=["compute_regime"]).compute_regime(),
            "market_regime", "") or ""), "")
        b = _safe(lambda: __import__("scan.breadth", fromlist=["breadth_from_cache"]
                                     ).breadth_from_cache(), {}) or {}
        ctx["market_health"] = b.get("verdict", "")
        ctx["pct_above_50dma"] = b.get("pct_above_50")
        ctx["sector"] = _safe(lambda: __import__(
            "scan.sector_heat", fromlist=["sector_of"]).sector_of(symbol), "") or ""
        report["context"] = ctx
    return report


def research_summary_line(report: dict) -> str:
    """A one-sentence plain read of the report — the headline for the search."""
    s = report.get("setup", {})
    v = s.get("verdict", "")
    cur = report.get("cur", "₹")
    q = report.get("quote", {})
    px = q.get("price")
    head = f"{report['symbol']} {cur}{px:,.1f}" if px else report["symbol"]
    if v in ("STRONG BUY", "BUY"):
        return (f"{head} — {v} setup ({', '.join(s.get('signals', [])[:2])}); "
                f"entry {cur}{s.get('entry', 0):,.1f}, stop {cur}{s.get('stop', 0):,.1f}, "
                f"target {cur}{s.get('target', 0):,.1f}.")
    return f"{head} — abhi koi buy setup nahi; neeche poori research dekho."
