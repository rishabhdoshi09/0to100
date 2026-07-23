"""
JARVIS Conviction Engine — the "why should I buy this" layer.

Takes the top technical candidates from the unified scanner and builds
a full conviction case for each, the way an analyst would:

  ✓ Technical setup      (VCP / breakout / momentum — from unified scanner)
  ✓ Volume confirmation  (surge vs 20-day average)
  ✓ Trend health         (above 200-day average)
  ✓ Buzzing              (in the news last 24h — headline attached)
  ✓ Earnings growth      (quarterly YoY growth via yfinance)
  ⚠ Earnings soon        (results within 3 days — event risk)

Conviction = technical score + evidence bonuses. Verdicts:
  STRONG BUY (≥75, 4+ checks) · BUY (≥55) · WATCH (rest)

Only enriches the TOP N candidates (default 40) so the per-stock
lookups stay cheap. Runs inside the background auto-scan thread.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from logger import get_logger

log = get_logger(__name__)

_ENRICH_TOP_N = 40


# ── News buzz ─────────────────────────────────────────────────────────────────

def _fetch_buzz_map(symbols: list[str]) -> dict[str, str]:
    """{symbol: headline} for stocks mentioned in the last 24h of news."""
    try:
        from news.fetcher import NewsFetcher
        from data.nse_universe import get_nse_universe_with_names
        articles = NewsFetcher().fetch_all(max_age_hours=24)
        if not articles:
            return {}
        names = get_nse_universe_with_names()
        buzz: dict[str, str] = {}
        for sym in symbols:
            if sym in buzz:
                continue
            # Match on symbol or the first word of the company name (≥4 chars)
            company = (names.get(sym) or "").split()[0] if names.get(sym) else ""
            needles = [sym.upper()]
            if len(company) >= 4:
                needles.append(company.upper())
            for a in articles:
                head = a.headline.upper()
                if any(n in head for n in needles):
                    buzz[sym] = a.headline[:110]
                    break
        return buzz
    except Exception as exc:
        log.debug("buzz_map_failed", error=str(exc))
        return {}


# ── Earnings ──────────────────────────────────────────────────────────────────

# Earnings facts change daily at most — cache per symbol per day so the
# 15-min scan cycle doesn't hammer yfinance with 40 slow .info calls.
_earn_cache: dict[str, dict] = {}
_earn_cache_day: str = ""


def _earnings_info(symbol: str) -> dict:
    """{growth_pct: float|None, days_to_results: int|None} — best effort,
    cached for the trading day."""
    global _earn_cache, _earn_cache_day
    from datetime import datetime as _dt
    today = _dt.now().strftime("%Y-%m-%d")
    if _earn_cache_day != today:
        _earn_cache = {}
        _earn_cache_day = today
    if symbol in _earn_cache:
        return dict(_earn_cache[symbol])

    out: dict = {"growth_pct": None, "days_to_results": None}
    try:
        import yfinance as yf
        info = yf.Ticker(f"{symbol}.NS").info or {}
        g = info.get("earningsQuarterlyGrowth")
        if g is not None:
            out["growth_pct"] = round(float(g) * 100, 1)
    except Exception:
        pass
    try:
        from data.earnings_calendar import get_earnings_date
        d = get_earnings_date(symbol)
        if d:
            out["days_to_results"] = (d - date.today()).days
    except Exception:
        pass
    _earn_cache[symbol] = dict(out)
    return out


# ── Main enrichment ───────────────────────────────────────────────────────────

def build_conviction(results: list[dict], top_n: int = _ENRICH_TOP_N) -> list[dict]:
    """
    Enrich the top technical candidates with buzz + earnings evidence and
    a plain-English conviction checklist. Returns ALL results — enriched
    ones first (sorted by conviction), rest unchanged.
    """
    if not results:
        return results

    head = results[:top_n]
    tail = results[top_n:]
    symbols = [r["symbol"] for r in head]

    buzz = _fetch_buzz_map(symbols)

    earnings: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_earnings_info, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                earnings[futs[fut]] = fut.result()
            except Exception:
                pass

    for r in head:
        sym = r["symbol"]
        checks: list[str] = []
        chase_risk = bool(r.get("chase_risk"))

        # Technical evidence (already in plain English from the scanner).
        # chase_risk stocks carry one extra LEADING reason (the "don't
        # chase" warning, deliberately unpaired so it shows first as the
        # card headline) — skip it here so signals[i] still pairs with
        # its own real reason instead of shifting by one.
        sig_reasons = r.get("reasons", [])[1:] if chase_risk else r.get("reasons", [])
        for sig, reason in zip(r.get("signals", []), sig_reasons):
            checks.append(f"✓ {sig}: {reason}")

        conviction = float(r.get("score", 0))

        if r.get("volume_ratio", 1.0) >= 1.5:
            checks.append(f"✓ Volume surge: {r['volume_ratio']:.1f}× normal trading")
            conviction += 4

        headline = buzz.get(sym)
        if headline:
            checks.append(f"✓ Buzzing: “{headline}”")
            conviction += 8
            r["buzzing"] = True

        e = earnings.get(sym, {})
        g = e.get("growth_pct")
        if g is not None and g >= 15:
            checks.append(f"✓ Good earnings: profit up {g:.0f}% vs last year")
            conviction += 10
        elif g is not None and g < -15:
            checks.append(f"⚠ Weak earnings: profit down {abs(g):.0f}% vs last year")
            conviction -= 10

        d = e.get("days_to_results")
        if d is not None and 0 <= d <= 3:
            checks.append(f"⚠ Results in {d} day{'s' if d != 1 else ''} — event risk, size small")
            conviction -= 5
        elif d is not None and 4 <= d <= 10:
            checks.append(f"• Results in {d} days — possible catalyst")

        conviction = max(0.0, min(100.0, conviction))
        n_pos = sum(1 for c in checks if c.startswith("✓"))
        if chase_risk:
            # scanner's own don't-chase safety demotion — a WATCH for a
            # SPECIFIC reason (extended, no confirmed trigger), not just a
            # score miss. Buzz/earnings evidence must never override it.
            verdict = "WATCH"
        elif conviction >= 75 and n_pos >= 4:
            verdict = "STRONG BUY"
        elif conviction >= 55 and n_pos >= 2:
            verdict = "BUY"
        else:
            verdict = "WATCH"

        r["checks"] = checks
        r["conviction"] = round(conviction, 1)
        r["verdict"] = verdict

    head.sort(key=lambda r: r.get("conviction", r.get("score", 0)), reverse=True)
    log.info("conviction_built", enriched=len(head),
             strong=sum(1 for r in head if r["verdict"] == "STRONG BUY"))
    return head + tail
