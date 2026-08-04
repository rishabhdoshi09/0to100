"""Smart options *positioning read* — research context, never a buy ticket.

Combines live OI/IV/PCR/max-pain (and optional EOD trend + cash-scan facts)
into an explicit stance a human can use for tomorrow-watch / long-horizon
consideration.

Honesty rules (hard):
  - no invented Greeks, prices, or fills
  - no BUY / SELL / STRONG BUY labels from options alone
  - GO-style language means risk-coherent context, not “buy now”
  - missing evidence stays missing and lowers confidence
  - paper-first; never places orders
"""
from __future__ import annotations

from typing import Any, Mapping


# Research stances — never BUY/SELL.
SUPPORTIVE = "SUPPORTIVE"
CAUTION = "CAUTION"
HOSTILE = "HOSTILE"
NEUTRAL = "NEUTRAL"
INCOMPLETE = "INCOMPLETE"

# What a retail owner might *consider* next — still not an order.
CONSIDER_WATCH = "tomorrow_watch"
CONSIDER_WAIT = "wait_for_better_entry"
CONSIDER_AVOID = "avoid_chase"
CONSIDER_REFRESH = "refresh_chain_or_scan"


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _pcr_score(pcr: float | None) -> tuple[float | None, str]:
    """Map PCR to [-1, +1] long-context score. High PCR ≈ put wall / supportive for dips."""
    if pcr is None or pcr <= 0:
        return None, "PCR unavailable"
    if pcr >= 1.3:
        return 0.85, f"PCR {pcr:.2f} — put OI dominates (dip-buyers / support bias)"
    if pcr >= 1.05:
        return 0.45, f"PCR {pcr:.2f} — mild put-side skew"
    if pcr >= 0.85:
        return 0.05, f"PCR {pcr:.2f} — balanced"
    if pcr >= 0.70:
        return -0.35, f"PCR {pcr:.2f} — call-side skew (resistance risk)"
    return -0.75, f"PCR {pcr:.2f} — calls dominate (hostile for fresh longs)"


def _max_pain_score(spot: float | None, max_pain: float | None) -> tuple[float | None, str]:
    if spot is None or spot <= 0 or max_pain is None or max_pain <= 0:
        return None, "Max pain vs spot unavailable"
    gap_pct = (spot - max_pain) / spot * 100.0
    # Spot above max pain → gravity down toward pain (hostile for chase longs)
    if gap_pct >= 1.5:
        return -0.55, f"Spot {gap_pct:+.1f}% above max pain — pull-toward-pain risk"
    if gap_pct >= 0.5:
        return -0.20, f"Spot {gap_pct:+.1f}% above max pain — mild overhead gravity"
    if gap_pct > -0.5:
        return 0.15, f"Spot near max pain ({gap_pct:+.1f}%) — pinned zone"
    if gap_pct > -1.5:
        return 0.40, f"Spot {gap_pct:+.1f}% below max pain — upside magnet possible"
    return 0.65, f"Spot {gap_pct:+.1f}% below max pain — stronger upside magnet"


def _oi_wall_score(
    spot: float | None,
    top_calls: list[Mapping[str, Any]],
    top_puts: list[Mapping[str, Any]],
) -> tuple[float | None, str, list[str]]:
    """Nearest heavy CE wall above / PE wall below as resistance/support context."""
    notes: list[str] = []
    if spot is None or spot <= 0:
        return None, "OI wall context needs spot", notes
    call_above = []
    for row in top_calls or []:
        strike = _f(row.get("strike"))
        oi = _f(row.get("ce_oi")) or 0.0
        if strike is not None and strike > spot and oi > 0:
            call_above.append((strike, oi))
    put_below = []
    for row in top_puts or []:
        strike = _f(row.get("strike"))
        oi = _f(row.get("pe_oi")) or 0.0
        if strike is not None and strike < spot and oi > 0:
            put_below.append((strike, oi))
    call_above.sort(key=lambda x: x[0])
    put_below.sort(key=lambda x: -x[0])
    score = 0.0
    parts = 0
    if put_below:
        strike, oi = put_below[0]
        notes.append(f"Put wall ~{strike:.0f} (OI {oi:,.0f}) below spot")
        score += 0.35
        parts += 1
    if call_above:
        strike, oi = call_above[0]
        dist = (strike - spot) / spot * 100.0
        notes.append(f"Call wall ~{strike:.0f} (OI {oi:,.0f}) · {dist:.1f}% above")
        if dist <= 0.8:
            score -= 0.55  # very close resistance
        elif dist <= 2.0:
            score -= 0.25
        else:
            score += 0.05
        parts += 1
    if parts == 0:
        return None, "No clear OI walls near spot", notes
    return max(-1.0, min(1.0, score)), "OI walls mapped around spot", notes


def _iv_score(atm_iv: float | None, iv_rank: float | None) -> tuple[float | None, str]:
    if atm_iv is None:
        return None, "ATM IV unavailable"
    # High IV = expensive options / event risk — hostile for fresh chase.
    if iv_rank is not None and iv_rank >= 75:
        return -0.45, f"ATM IV {atm_iv:.1f}% · IV rank {iv_rank:.0f} (elevated)"
    if atm_iv >= 35:
        return -0.35, f"ATM IV {atm_iv:.1f}% — rich premium / event risk"
    if atm_iv <= 12:
        return 0.25, f"ATM IV {atm_iv:.1f}% — calm premium backdrop"
    return 0.05, f"ATM IV {atm_iv:.1f}% — ordinary"


def _coi_score(chain_rows: list[Mapping[str, Any]], spot: float | None) -> tuple[float | None, str]:
    """Net change-in-OI near ATM: put build = supportive, call build = hostile."""
    if not chain_rows or spot is None or spot <= 0:
        return None, "Near-ATM OI change unavailable"
    near = []
    for row in chain_rows:
        strike = _f(row.get("strike"))
        if strike is None:
            continue
        if abs(strike - spot) / spot <= 0.03:
            near.append(row)
    if not near:
        return None, "No strikes within 3% of spot for OI-change read"
    ce_coi = sum(_f(r.get("ce_coi")) or 0.0 for r in near)
    pe_coi = sum(_f(r.get("pe_coi")) or 0.0 for r in near)
    if ce_coi == 0 and pe_coi == 0:
        return None, "Change-in-OI is flat/zero on near strikes"
    net = pe_coi - ce_coi
    if net > 0:
        return min(0.7, 0.2 + net / (abs(ce_coi) + abs(pe_coi) + 1) * 0.5), (
            f"Near-ATM put OI building vs calls (ΔPE {pe_coi:+.0f} · ΔCE {ce_coi:+.0f})"
        )
    return max(-0.7, -0.2 + net / (abs(ce_coi) + abs(pe_coi) + 1) * 0.5), (
        f"Near-ATM call OI building vs puts (ΔCE {ce_coi:+.0f} · ΔPE {pe_coi:+.0f})"
    )


def _history_score(history_rows: list[Mapping[str, Any]]) -> tuple[float | None, str]:
    """Recent PCR trend from EOD snapshots — rising PCR ≈ improving put support."""
    rows = [r for r in (history_rows or []) if _f(r.get("pcr")) is not None]
    if len(rows) < 2:
        return None, "Need ≥2 EOD PCR snapshots for trend"
    # Newest first in store; take last up to 5 chronological.
    ordered = list(reversed(rows[:5]))
    first = _f(ordered[0].get("pcr"))
    last = _f(ordered[-1].get("pcr"))
    if first is None or last is None:
        return None, "EOD PCR trend incomplete"
    delta = last - first
    if delta >= 0.15:
        return 0.40, f"EOD PCR rising {first:.2f} → {last:.2f}"
    if delta <= -0.15:
        return -0.40, f"EOD PCR falling {first:.2f} → {last:.2f}"
    return 0.05, f"EOD PCR steady {first:.2f} → {last:.2f}"


def _cash_scan_overlay(scan_row: Mapping[str, Any] | None) -> tuple[float | None, list[str], list[str]]:
    """Optional cash-market scan facts — never invent when missing."""
    reasons: list[str] = []
    risks: list[str] = []
    if not scan_row:
        return None, reasons, risks
    score = 0.0
    parts = 0
    verdict = str(scan_row.get("verdict") or "").upper()
    if verdict in {"STRONG BUY", "BUY"}:
        score += 0.45
        parts += 1
        reasons.append(f"Cash scan verdict {verdict}")
    elif verdict == "WATCH":
        score += 0.05
        parts += 1
        reasons.append("Cash scan is WATCH only")
    edge = _f(scan_row.get("edge_r"))
    if edge is not None:
        parts += 1
        if edge <= -0.05:
            score -= 0.55
            risks.append(f"Measured LOSER edge {edge:+.2f}R")
        elif edge >= 0.08:
            score += 0.35
            reasons.append(f"Measured edge {edge:+.2f}R")
        else:
            reasons.append(f"Measured edge {edge:+.2f}R (near flat)")
    if scan_row.get("chase_risk"):
        score -= 0.50
        parts += 1
        risks.append("Scanner chase / extension risk")
    mom = _f(scan_row.get("score"))
    if mom is not None:
        parts += 1
        if mom >= 70:
            score += 0.25
            reasons.append(f"Momentum score {mom:.0f}")
        elif mom < 45:
            score -= 0.25
            risks.append(f"Weak momentum score {mom:.0f}")
    if parts == 0:
        return None, reasons, risks
    return max(-1.0, min(1.0, score / max(1, parts) * 1.2)), reasons, risks


def _stance_from_score(score: float | None, coverage: float) -> str:
    if score is None or coverage < 0.35:
        return INCOMPLETE
    if score >= 0.35:
        return SUPPORTIVE
    if score <= -0.35:
        return HOSTILE
    if abs(score) < 0.12:
        return NEUTRAL
    return CAUTION


def _consider_for(stance: str, risks: list[str], scan_row: Mapping[str, Any] | None) -> list[str]:
    tags: list[str] = []
    if stance == SUPPORTIVE:
        tags.append(CONSIDER_WATCH)
    elif stance == CAUTION:
        tags.append(CONSIDER_WAIT)
    elif stance == HOSTILE:
        tags.append(CONSIDER_AVOID)
    elif stance == INCOMPLETE:
        tags.append(CONSIDER_REFRESH)
    if any("chase" in r.lower() for r in risks):
        if CONSIDER_AVOID not in tags:
            tags.append(CONSIDER_AVOID)
    if scan_row and str(scan_row.get("verdict") or "").upper() in {"BUY", "STRONG BUY"} and stance == SUPPORTIVE:
        tags.append("cash_and_options_aligned")
    return tags


def build_positioning_read(
    chain: Mapping[str, Any] | None,
    *,
    history_rows: list[Mapping[str, Any]] | None = None,
    scan_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an honest options+cash research read from real fields only."""
    chain = dict(chain or {})
    if not chain.get("available"):
        return {
            "available": False,
            "stance": INCOMPLETE,
            "score": None,
            "confidence": 0.0,
            "headline": "Options chain unavailable — no positioning read",
            "reasons": [],
            "risks": [str(chain.get("message") or "Live chain missing")],
            "factors": [],
            "consider_for": [CONSIDER_REFRESH],
            "places_orders": False,
            "live_locked": True,
            "signal_desk": False,
            "honesty": (
                "This is not a buy signal. Without a live chain, QuantTerm will not invent "
                "PCR, OI walls, or trade direction."
            ),
        }

    reasons: list[str] = []
    risks: list[str] = []
    factors: list[dict[str, Any]] = []
    weights_scores: list[tuple[str, float, float]] = []  # name, weight, score -1..1

    pcr = _f(chain.get("pcr"))
    s, note = _pcr_score(pcr)
    if s is not None:
        weights_scores.append(("pcr", 0.30, s))
        factors.append({"name": "pcr", "score": round(s, 2), "note": note})
        (reasons if s >= 0 else risks).append(note)
    else:
        risks.append(note)

    spot = _f(chain.get("spot"))
    max_pain = _f(chain.get("max_pain"))
    s, note = _max_pain_score(spot, max_pain)
    if s is not None:
        weights_scores.append(("max_pain", 0.20, s))
        factors.append({"name": "max_pain", "score": round(s, 2), "note": note})
        (reasons if s >= 0 else risks).append(note)
    else:
        risks.append(note)

    s, note, wall_notes = _oi_wall_score(
        spot,
        list(chain.get("top_call_oi") or []),
        list(chain.get("top_put_oi") or []),
    )
    if s is not None:
        weights_scores.append(("oi_walls", 0.20, s))
        factors.append({"name": "oi_walls", "score": round(s, 2), "note": note})
        for item in wall_notes:
            (reasons if "Put wall" in item else risks).append(item)
    else:
        risks.append(note)

    s, note = _iv_score(_f(chain.get("atm_iv")), _f(chain.get("iv_rank")))
    if s is not None:
        weights_scores.append(("atm_iv", 0.10, s))
        factors.append({"name": "atm_iv", "score": round(s, 2), "note": note})
        (reasons if s >= 0 else risks).append(note)

    s, note = _coi_score(list(chain.get("chain") or []), spot)
    if s is not None:
        weights_scores.append(("near_atm_coi", 0.10, s))
        factors.append({"name": "near_atm_coi", "score": round(s, 2), "note": note})
        (reasons if s >= 0 else risks).append(note)

    s, note = _history_score(list(history_rows or []))
    if s is not None:
        weights_scores.append(("eod_pcr_trend", 0.10, s))
        factors.append({"name": "eod_pcr_trend", "score": round(s, 2), "note": note})
        (reasons if s >= 0 else risks).append(note)

    cash_score, cash_reasons, cash_risks = _cash_scan_overlay(scan_row)
    if cash_score is not None:
        weights_scores.append(("cash_scan", 0.25, cash_score))
        factors.append({"name": "cash_scan", "score": round(cash_score, 2), "note": "Cash scan overlay"})
        reasons.extend(cash_reasons)
        risks.extend(cash_risks)

    # Coverage vs a full ideal set of 7 components.
    coverage = len(weights_scores) / 7.0
    if not weights_scores:
        composite = None
    else:
        wsum = sum(w for _, w, _ in weights_scores)
        composite = sum(w * sc for _, w, sc in weights_scores) / wsum if wsum else None
        if composite is not None:
            composite = round(max(-1.0, min(1.0, composite)), 3)

    stance = _stance_from_score(composite, coverage)
    consider = _consider_for(stance, risks, scan_row)

    headlines = {
        SUPPORTIVE: "Options positioning is supportive for a long *research* watch",
        CAUTION: "Mixed options context — wait for a cleaner entry, do not chase",
        HOSTILE: "Options positioning is hostile for fresh long risk",
        NEUTRAL: "Options positioning is balanced — no edge from derivatives alone",
        INCOMPLETE: "Not enough options/cash evidence for a stance",
    }

    return {
        "available": True,
        "symbol": chain.get("symbol"),
        "expiry": chain.get("expiry"),
        "stance": stance,
        "score": composite,
        "confidence": round(min(1.0, coverage), 2),
        "headline": headlines[stance],
        "reasons": reasons[:8],
        "risks": risks[:8],
        "factors": factors,
        "consider_for": consider,
        "cash_scan_joined": bool(scan_row),
        "places_orders": False,
        "live_locked": True,
        "signal_desk": False,
        "honesty": (
            "Research positioning read from real OI/IV/PCR/max-pain"
            + (" + cash scan" if scan_row else "")
            + ". Not a buy/sell signal, not Greeks, not a live order. "
            "SUPPORTIVE means context is friendly for watching — still paper-first."
        ),
    }


def attach_positioning_read(
    chain: Mapping[str, Any],
    *,
    history_rows: list[Mapping[str, Any]] | None = None,
    scan_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return chain dict with ``positioning_read`` attached."""
    out = dict(chain)
    out["positioning_read"] = build_positioning_read(
        out,
        history_rows=history_rows,
        scan_row=scan_row,
    )
    # Keep top-level bias aligned with the smarter stance when chain is available.
    read = out["positioning_read"]
    if out.get("available") and read.get("stance") in {SUPPORTIVE, CAUTION, HOSTILE, NEUTRAL}:
        out["bias"] = {
            SUPPORTIVE: "BULLISH",
            CAUTION: "CAUTION",
            HOSTILE: "BEARISH",
            NEUTRAL: "NEUTRAL",
        }.get(str(read.get("stance")), out.get("bias"))
        out["note"] = str(read.get("headline") or out.get("note") or "")
    return out
