"""
Options verdict — turn raw chain metrics into ONE clear read.

The options page had the numbers (PCR, Max Pain, IV, OI walls) but no
synthesis — a trader saw data, not a decision. This composes them into a
single structural view:

  • bias        — BULLISH / BEARISH / RANGE / NEUTRAL, with the reason
  • range       — the OI walls: PE support → CE resistance, + where spot sits
  • max-pain    — the expiry magnet: distance + pull direction
  • IV regime   — premium mehenga/sasta → buy vs sell premium

It is a STRUCTURAL READ, not a trade signal — options orders are not wired
to execution here (on purpose). Pure + testable: it takes primitives, so it
never depends on a live chain to be verified.
"""
from __future__ import annotations


def _pcr_bias(pcr: float) -> str:
    # Contrarian reading (matches the page's existing scale): a high PCR means
    # puts are heavily written → sellers don't expect a fall → support strong.
    if pcr >= 1.3:
        return "BULLISH"
    if pcr <= 0.7:
        return "BEARISH"
    return "NEUTRAL"


def _iv_stance(iv_pct: float) -> tuple[str, str]:
    if iv_pct >= 70:
        return "expensive", ("Premium mehenga (IV high) — option SELLING / "
                             "credit spreads favoured; naked buying mein theta "
                             "bleed.")
    if iv_pct <= 30:
        return "cheap", ("Premium sasta (IV low) — option BUYING / debit "
                         "spreads favoured; sellers ko kam edge.")
    return "normal", "IV normal band mein — directional view pe chalo, edge IV se nahi."


def options_verdict(spot: float, pcr: float, max_pain: float,
                    support: float, resistance: float,
                    iv_pct: float) -> dict:
    """Compose the chain metrics into one structural verdict. All inputs are
    primitives so this is fully unit-testable."""
    if spot <= 0 or not support or not resistance or support >= resistance:
        return {"bias": "NEUTRAL",
                "verdict_line": "Chain data adhoori — clear structural read "
                                "nahi ban raha. Market hours mein dobara dekho.",
                "range": {}, "max_pain": {}, "iv": {}, "notes": []}

    notes: list[str] = []

    # ── OI walls → expected range + where spot sits ───────────────────────────
    width_pct = (resistance - support) / spot * 100
    pos = (spot - support) / (resistance - support)   # 0 = at support, 1 = at resist
    if pos <= 0.15:
        pos_note = (f"Spot support wall (₹{support:,.0f}) ke paas — bounce ya "
                    f"toota toh neeche tez move.")
    elif pos >= 0.85:
        pos_note = (f"Spot resistance wall (₹{resistance:,.0f}) ke paas — "
                    f"rejection ya breakout ka watch.")
    else:
        pos_note = (f"Spot range ke beech (₹{support:,.0f}–₹{resistance:,.0f}) — "
                    f"walls tak room dono taraf.")
    notes.append(pos_note)
    tight_range = width_pct <= 2.0
    if tight_range:
        notes.append(f"Walls tight ({width_pct:.1f}% wide) — range-bound / "
                     f"theta-friendly tape.")

    # ── Max Pain → expiry magnet ──────────────────────────────────────────────
    mp_dist = ((spot - max_pain) / max_pain * 100) if max_pain else 0.0
    if abs(mp_dist) < 0.3:
        mp_dir, mp_note = "at", (f"Spot max-pain (₹{max_pain:,.0f}) pe hi — "
                                 f"expiry tak yahin magnet reh sakta hai.")
    elif mp_dist < 0:
        mp_dir, mp_note = "up", (f"Spot max-pain se {abs(mp_dist):.1f}% neeche — "
                                 f"expiry tak upar (₹{max_pain:,.0f}) khichने ka "
                                 f"jhukav.")
    else:
        mp_dir, mp_note = "down", (f"Spot max-pain se {mp_dist:.1f}% upar — expiry "
                                   f"tak neeche (₹{max_pain:,.0f}) khinchne ka "
                                   f"jhukav.")
    notes.append(mp_note + " (Yeh khichav expiry ke paas sabse strong.)")

    # ── IV regime → buy vs sell premium ───────────────────────────────────────
    iv_stance, iv_hint = _iv_stance(iv_pct)
    notes.append(iv_hint)

    # ── Composite bias — PCR vote + max-pain pull, walls arbitrate ties ───────
    pcr_bias = _pcr_bias(pcr)
    bull = (pcr_bias == "BULLISH") + (mp_dir == "up")
    bear = (pcr_bias == "BEARISH") + (mp_dir == "down")
    if bull > bear:
        bias, reason = "BULLISH", (f"PCR {pcr:.2f} + max-pain pull up — "
                                   f"support side bhaari.")
    elif bear > bull:
        bias, reason = "BEARISH", (f"PCR {pcr:.2f} + max-pain pull down — "
                                   f"resistance side bhaari.")
    elif tight_range or mp_dir == "at":
        bias, reason = "RANGE", (f"PCR {pcr:.2f}, walls ₹{support:,.0f}–"
                                 f"₹{resistance:,.0f} — range-bound, kinaaron pe "
                                 f"fade karo.")
    else:
        bias, reason = "NEUTRAL", f"PCR {pcr:.2f} — signals mile-jule, direction saaf nahi."

    verdict_line = (f"{bias} · range ₹{support:,.0f}–₹{resistance:,.0f} · "
                    f"max-pain ₹{max_pain:,.0f} · IV {iv_pct:.0f}th ({iv_stance}). "
                    f"{reason}")

    return {
        "bias": bias,
        "reason": reason,
        "verdict_line": verdict_line,
        "range": {"support": support, "resistance": resistance, "spot": spot,
                  "width_pct": round(width_pct, 2),
                  "position": round(pos, 2)},
        "max_pain": {"level": max_pain, "dist_pct": round(mp_dist, 2),
                     "pull": mp_dir},
        "iv": {"percentile": iv_pct, "stance": iv_stance, "hint": iv_hint},
        "notes": notes,
    }
