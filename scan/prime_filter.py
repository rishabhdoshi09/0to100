"""
💎 Prime filter — Telegram pe woh hi jaaye jo har data-layer se guzra ho.

"Sweet earnings" ka raasta zyada alerts nahi — SAHI alerts hain. Ye filter
har candidate ko system ke SAB evidence streams se guzaarta hai, aur sirf
survivors ko PRIME tag deta hai (push mein sabse upar, 💎 ke saath):

  L1  Verdict        — BUY / STRONG BUY hi (WATCH kabhi nahi)
  L2  Type           — Momentum ya Breakout category (user ka mandate)
  L3  Conviction     — high_conviction tier YA breakout_conviction ≥ 60
  L4  Evidence (EV)  — measured EV ho toh CONSERVATIVE EV > 0 zaroori
                       (proven-negative setup kabhi prime nahi)
  L5  Liquidity      — price × avg 20-day volume ≥ ₹5 crore/day —
                       LARGE QTY tabhi jab exit bhi large ho sake;
                       illiquid naam pe badi position = trap
  L6  Breadth        — market andar se NARROW ho toh koi prime nahi
                       (breakout graveyard mein VIP pass nahi milta)
  L7  Regime         — jo signal IS tape mein proven-leaky hai (regime
                       calibration demote), woh setup prime nahi

Pure + testable: primitives lete hain, koi I/O nahi. Demote-only design —
ye filter kisi ko BUY nahi banata, sirf best-of-the-best ko alag karta hai.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)

MIN_TURNOVER_CR = 5.0          # ₹5 cr/day avg turnover — large-qty liquidity
MIN_BREAKOUT_CONV = 60.0


def prime_check(r: dict, breadth_verdict: str = "",
                demoted_labels: set | None = None,
                min_turnover_cr: float = MIN_TURNOVER_CR
                ) -> tuple[bool, list[str], str]:
    """(is_prime, why[], fail_reason). fail_reason == "" when prime."""
    demoted_labels = demoted_labels or set()
    why: list[str] = []

    # L1 — verdict
    if r.get("verdict") not in ("STRONG BUY", "BUY"):
        return False, [], "verdict BUY nahi"
    # L2 — momentum/breakout mandate
    cats = set(r.get("categories") or [])
    if not (cats & {"Momentum", "Breakout"}):
        return False, [], "na momentum na breakout"
    why.append("/".join(sorted(cats & {"Momentum", "Breakout"})))
    # L3 — conviction tier
    conv = float(r.get("breakout_conviction") or 0)
    if r.get("high_conviction"):
        why.append("high-conviction tier (evidence-gated)")
    elif conv >= MIN_BREAKOUT_CONV:
        why.append(f"conviction {conv:.0f}")
    else:
        return False, [], f"conviction kam ({conv:.0f})"
    # L4 — measured EV must not be negative (conservative bound)
    ev_lb = r.get("ev_lb_pct")
    if ev_lb is not None:
        if float(ev_lb) <= 0:
            return False, [], f"conservative EV negative ({float(ev_lb):+.1f}%)"
        why.append(f"EV {float(r.get('ev_pct') or 0):+.1f}% "
                   f"({r.get('ev_conf', '?')})")
    # L5 — liquidity floor for LARGE qty
    turnover = float(r.get("price") or 0) * float(r.get("avg_vol20") or 0)
    if turnover < min_turnover_cr * 1e7:
        return False, [], (f"illiquid — avg turnover "
                           f"₹{turnover / 1e7:.1f}cr < ₹{min_turnover_cr:.0f}cr")
    why.append(f"liquid (₹{turnover / 1e7:.0f}cr/day)")
    # L6 — market internals
    if (breadth_verdict or "").upper() == "NARROW":
        return False, [], "market breadth NARROW — breakout graveyard"
    # L7 — regime-leaky signals out
    leaked = set(r.get("signals") or []) & demoted_labels
    if leaked:
        return False, [], f"is tape mein leaky: {sorted(leaked)[0]}"

    return True, why[:4], ""


def tag_prime(results: list[dict], breadth_verdict: str = "",
              demoted_labels: set | None = None) -> int:
    """In-place: r['prime'] + r['prime_why'] on survivors. Returns count."""
    n = 0
    for r in results:
        ok, why, _fail = prime_check(r, breadth_verdict, demoted_labels)
        if ok:
            r["prime"] = True
            r["prime_why"] = " · ".join(why)
            n += 1
    return n
