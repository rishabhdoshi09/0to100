"""
Portfolio Intelligence — treat capital as a portfolio, not a stock list.

The daily professional questions, answered from data we already track:

  • What is the portfolio's CURRENT expected value (per holding + blended)?
  • Which holding is the weakest by today's evidence?
  • Is there a materially stronger opportunity on the scan than my weakest
    holding (opportunity cost)?
  • Where does the next rupee earn the most (best deployable idea)?

Every open position competes daily against the best available ideas.

READ-ONLY BY DESIGN: this engine recommends — it never rotates capital
itself. Rotation costs real money (exit+entry charges, slippage, taxes),
so the advice carries a conservative EV-gap threshold and states the cost
caveat. The user (or a future gated executor) owns the click.

Pure decision core (`rotation_advice`) takes primitives → fully testable
without live scans or a live book.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)

# A swap must clear entry+exit costs, slippage and tax friction with room to
# spare — below this EV gap (percentage points), churn eats the difference.
MIN_EV_GAP_PCT = 2.5


def rotation_advice(holdings: list[dict], candidates: list[dict],
                    min_gap_pct: float = MIN_EV_GAP_PCT) -> dict:
    """Pure core. holdings: [{symbol, ev_pct|None}], candidates (not held):
    [{symbol, ev_pct|None, verdict}]. Returns:
      {portfolio_ev_pct, weakest, best_new, swap: {...}|None}
    A holding with ev_pct=None (no evidence today) is treated as the weakest
    candidate for replacement only if EVERY holding lacks EV — we don't punish
    a stock merely for missing data when others have real claims."""
    known = [h for h in holdings if h.get("ev_pct") is not None]
    port_ev = (round(sum(h["ev_pct"] for h in known) / len(known), 2)
               if known else None)

    weakest = None
    if known:
        weakest = min(known, key=lambda h: h["ev_pct"])
    elif holdings:
        weakest = holdings[0]

    scored = [c for c in candidates
              if c.get("ev_pct") is not None
              and c.get("verdict") in ("STRONG BUY", "BUY")]
    best_new = max(scored, key=lambda c: c["ev_pct"]) if scored else None

    swap = None
    if (weakest is not None and best_new is not None
            and weakest.get("ev_pct") is not None
            and best_new["ev_pct"] - weakest["ev_pct"] >= min_gap_pct):
        swap = {
            "out": weakest["symbol"], "out_ev": weakest["ev_pct"],
            "in": best_new["symbol"], "in_ev": best_new["ev_pct"],
            "gap_pct": round(best_new["ev_pct"] - weakest["ev_pct"], 2),
            "note": (f"{weakest['symbol']} (EV {weakest['ev_pct']:+.1f}%) vs "
                     f"{best_new['symbol']} (EV {best_new['ev_pct']:+.1f}%) — "
                     f"gap {best_new['ev_pct'] - weakest['ev_pct']:+.1f}pp. "
                     f"Costs/taxes/slippage ke baad hi sochna — advice hai, "
                     f"order nahi."),
        }
    return {"portfolio_ev_pct": port_ev, "weakest": weakest,
            "best_new": best_new, "swap": swap}


def daily_review() -> dict:
    """Compose today's portfolio intelligence from the live book + the scan
    store (EV-tagged). Safe on empty book/store — returns Nones."""
    try:
        from risk.position_manager import review_positions
        book = review_positions() or []
    except Exception:
        book = []
    try:
        from scan.auto_scan import get_results
        results, _n, _ts, _st = get_results()
    except Exception:
        results = []

    by_symbol = {r.get("symbol"): r for r in results}
    held = {p.get("symbol") for p in book}
    holdings = [{"symbol": p.get("symbol"),
                 "ev_pct": (by_symbol.get(p.get("symbol")) or {}).get("ev_pct")}
                for p in book]
    candidates = [{"symbol": r.get("symbol"), "ev_pct": r.get("ev_pct"),
                   "verdict": r.get("verdict")}
                  for r in results if r.get("symbol") not in held]
    out = rotation_advice(holdings, candidates)
    out["n_holdings"] = len(holdings)
    return out
