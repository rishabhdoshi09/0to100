"""Mixture-of-experts meta-layer for Recommendations.

Experts propose. Families vote. The meta-ranker does **not** average
SEPA 20 + Momentum 20 + Quality 20. Correlated tape / RS / momentum
collapse into Price Leadership.

Tiers:
  high_conviction — ≥3 independent family passes, why-now, entry ready, no hard conflict
  good_setup      — strong primary thesis + a second family, why-now, entry ready
  watch           — interesting but missing confirmation, near setup, or extended
  avoid           — hard fundamental reject or broken structure

Empty high-conviction is a successful output.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.breakout_quality import RSI_HARD
from product.reco_experts import (
    FAMILY_CATALYST,
    FAMILY_CONTEXT,
    FAMILY_FUND_CHANGE,
    FAMILY_LABELS,
    FAMILY_QUALITY,
    HORIZON_LABELS,
    attach_experts,
    _f,
    _signals,
)

# Tape-derived families are useful but not independent of each other.
_TAPE_FAMILIES = frozenset({"price_leadership", "structure", "participation"})
_NON_TAPE_FAMILIES = frozenset({
    FAMILY_QUALITY, FAMILY_FUND_CHANGE, FAMILY_CONTEXT, FAMILY_CATALYST,
})

# Thesis experts are proposals, not extra family votes.
_THESIS_ONLY = frozenset({"mom_quality"})

TIER_HIGH = "high_conviction"
TIER_GOOD = "good_setup"
TIER_WATCH = "watch"
TIER_AVOID = "avoid"

TIER_LABELS = {
    TIER_HIGH: "High Conviction",
    TIER_GOOD: "Good Setup",
    TIER_WATCH: "Watch",
    TIER_AVOID: "Avoid / Conflict",
}

TIER_RANK = {TIER_HIGH: 0, TIER_GOOD: 1, TIER_WATCH: 2, TIER_AVOID: 3}

ENTRY_READY = "ready"
ENTRY_NEAR = "near_setup"
ENTRY_EXTENDED = "extended"
ENTRY_BROKEN = "broken"
ENTRY_WATCH = "watch"

FAMILY_STRENGTH = {
    "pass": "Strong",
    "conflict": "Conflict",
    "fail": "Weak",
    "neutral": "Neutral",
    "unknown": "Unknown",
}

# Primary thesis preference when several experts propose.
_THESIS_PRIORITY = (
    "mom_quality",
    "earnings",
    "sepa",
    "vcp",
    "breakout",
    "pullback",
    "xs_momentum",
    "rs",
    "quality",
    "sector",
)


def entry_state(row: Mapping[str, Any]) -> str:
    if str(row.get("verdict") or "").upper() == "AVOID":
        return ENTRY_BROKEN
    if bool(row.get("chase_risk")):
        return ENTRY_EXTENDED
    rsi = _f(row.get("rsi"))
    if rsi is not None and rsi > RSI_HARD:
        return ENTRY_EXTENDED
    status = str(row.get("status") or "")
    if status == "Ready to trade":
        return ENTRY_READY
    if status in {"Watch for breakout", "Wait for pullback"} or "pullback" in status.lower():
        return ENTRY_NEAR
    grade = str(row.get("breakout_grade") or "").upper()
    if grade in {"A", "B"} and str(row.get("verdict") or "").upper() in {"BUY", "STRONG BUY"}:
        return ENTRY_READY
    return ENTRY_WATCH


def _family_votes(experts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_fam: dict[str, list[Mapping[str, Any]]] = {}
    for exp in experts:
        if exp.get("id") in _THESIS_ONLY:
            continue
        fam = str(exp.get("family") or "")
        if not fam:
            continue
        by_fam.setdefault(fam, []).append(exp)
    out: list[dict[str, Any]] = []
    for fam, items in by_fam.items():
        statuses = {str(i.get("status") or "unknown") for i in items}
        passes = [i for i in items if i.get("status") == "pass"]
        fails = [i for i in items if i.get("status") == "fail"]
        if passes and fails:
            status = "conflict"
        elif passes:
            status = "pass"
        elif fails:
            status = "fail"
        elif "neutral" in statuses:
            status = "neutral"
        else:
            status = "unknown"
        evidence: list[str] = []
        for item in passes or fails or items:
            evidence.extend(str(x) for x in (item.get("evidence") or [])[:2])
        out.append({
            "id": fam,
            "label": FAMILY_LABELS.get(fam, fam),
            "status": status,
            "strength": FAMILY_STRENGTH.get(status, "Unknown"),
            "experts": [str(i.get("id")) for i in items if i.get("status") == "pass"],
            "evidence": list(dict.fromkeys(evidence))[:4],
        })
    order = list(FAMILY_LABELS)
    out.sort(key=lambda item: order.index(item["id"]) if item["id"] in order else 99)
    return out


def _primary_thesis(experts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(e.get("id")): e for e in experts}
    # Combination theses first — more informative than a single name.
    earnings = by_id.get("earnings") or {}
    breakout = by_id.get("breakout") or {}
    vcp = by_id.get("vcp") or {}
    sector = by_id.get("sector") or {}
    mom_q = by_id.get("mom_quality") or {}
    if earnings.get("status") == "pass" and breakout.get("status") == "pass":
        return {
            "id": "earnings_breakout",
            "label": "Earnings Inflection + Breakout",
            "horizon": earnings.get("horizon") or "swing",
            "expert_ids": ["earnings", "breakout"],
        }
    if mom_q.get("status") == "pass":
        return {
            "id": "mom_quality",
            "label": "Momentum + Quality",
            "horizon": "position",
            "expert_ids": ["mom_quality", "xs_momentum", "quality"],
        }
    if vcp.get("status") == "pass" and sector.get("status") == "pass":
        return {
            "id": "vcp_sector",
            "label": "VCP + Sector Leadership",
            "horizon": "swing",
            "expert_ids": ["vcp", "sector"],
        }
    for key in _THESIS_PRIORITY:
        item = by_id.get(key) or {}
        if item.get("status") == "pass" and item.get("eligible"):
            return {
                "id": key,
                "label": str(item.get("thesis") or item.get("label") or key),
                "horizon": str(item.get("horizon") or "swing"),
                "expert_ids": [key],
            }
    return {
        "id": "",
        "label": "No independent thesis yet",
        "horizon": "swing",
        "expert_ids": [],
    }


def _why_now_from_experts(row: Mapping[str, Any], experts: Sequence[Mapping[str, Any]]) -> list[str]:
    bullets: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        line = " ".join(str(text or "").split())
        if not line or line.lower() in seen:
            return
        seen.add(line.lower())
        bullets.append(line)

    for exp in experts:
        if exp.get("status") != "pass":
            continue
        if exp.get("id") == "quality":
            # Classification tokens belong on the quality chip, not Why Now.
            continue
        for item in (exp.get("evidence") or [])[:2]:
            add(str(item))
    for sig, label in (
        ("BREAKOUT_52W", "52-week high breakout"),
        ("BREAKOUT_RES", "Resistance break with volume"),
        ("MOMENTUM", "Momentum improving"),
        ("VCP", "Base is tightening"),
        ("PRE_BREAKOUT", "Price is near the breakout pivot"),
    ):
        if sig in _signals(row):
            add(label)
    return bullets[:5]


def _conflicts(families: Sequence[Mapping[str, Any]], experts: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for fam in families:
        if fam.get("status") == "conflict":
            lines.append(f"{fam.get('label')}: pass and fail in the same family")
        if fam.get("status") == "fail":
            lines.append(f"{fam.get('label')}: {', '.join(fam.get('evidence') or []) or 'rejects'}")
    by_id = {str(e.get("id")): e for e in experts}
    sector = by_id.get("sector") or {}
    if sector.get("status") == "fail":
        lines.append("Strong stock inside a weak sector — conflict exposed, not auto-rejected")
    # Dedup
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:5]


def decide_tier(
    *,
    family_confirms: int,
    families: Sequence[Mapping[str, Any]],
    conflicts: Sequence[str],
    thesis: Mapping[str, Any],
    state: str,
    why_now: Sequence[str],
    quality_fail: bool,
) -> str:
    if quality_fail or state == ENTRY_BROKEN:
        return TIER_AVOID
    thesis_ok = bool(thesis.get("id"))
    if state != ENTRY_READY or not why_now or not thesis_ok:
        if thesis_ok or family_confirms >= 1:
            return TIER_WATCH
        return TIER_WATCH
    hard_conflict = any("Business Quality" in c and "rejects" in c.lower() for c in conflicts)
    if hard_conflict:
        return TIER_AVOID
    nontape = sum(
        1 for fam in families
        if fam.get("status") == "pass" and fam.get("id") in _NON_TAPE_FAMILIES
    )
    # High conviction needs agreement beyond the same scanner tape.
    if family_confirms >= 3 and nontape >= 1:
        return TIER_HIGH
    if family_confirms >= 2:
        return TIER_GOOD
    return TIER_WATCH


def attach_ensemble(row: Mapping[str, Any]) -> dict[str, Any]:
    """Fold expert panel into family votes, tier, thesis, and Buy eligibility."""
    out = dict(row)
    experts = list(out.get("experts") or [])
    families = _family_votes(experts)
    confirms = [f for f in families if f.get("status") == "pass"]
    fails = [f for f in families if f.get("status") in {"fail", "conflict"}]
    thesis = _primary_thesis(experts)
    state = entry_state(out)
    why = list(out.get("why_now") or []) or _why_now_from_experts(out, experts)
    conflicts = _conflicts(families, experts)
    quality = next((e for e in experts if e.get("id") == "quality"), {})
    quality_fail = quality.get("status") == "fail" and str(out.get("classification") or "") == "AVOID_REVIEW"
    tier = decide_tier(
        family_confirms=len(confirms),
        families=families,
        conflicts=conflicts,
        thesis=thesis,
        state=state,
        why_now=why,
        quality_fail=quality_fail,
    )
    recommend = tier in {TIER_HIGH, TIER_GOOD} and state == ENTRY_READY
    horizon = str(thesis.get("horizon") or "swing")
    out.update({
        "experts": experts,
        "families": families,
        "family_confirms": len(confirms),
        "family_fails": len(fails),
        "family_line": " + ".join(f["label"] for f in confirms) or "No independent evidence family confirmed",
        "primary_thesis": thesis.get("label") or "",
        "primary_thesis_id": thesis.get("id") or "",
        "thesis_horizon": horizon,
        "thesis_horizon_label": HORIZON_LABELS.get(horizon, horizon),
        "reco_tier": tier,
        "reco_tier_label": TIER_LABELS.get(tier, tier),
        "entry_state": state,
        "stock_quality": (
            "Excellent" if quality.get("status") == "pass" and (_f(out.get("fundamental_score")) or 0) >= 75
            else "Healthy" if quality.get("status") == "pass"
            else "Unmeasured" if quality.get("status") in {None, "unknown"}
            else "Weak"
        ),
        "timing": (
            "Ready" if state == ENTRY_READY
            else "Poor — extended / chase" if state == ENTRY_EXTENDED
            else "Near setup" if state == ENTRY_NEAR
            else "Broken" if state == ENTRY_BROKEN
            else "Not ready"
        ),
        "conflicts": conflicts,
        "ensemble_why_now": why,
        "allows_recommend": recommend,
        # Keep method_confirms as family confirms so Buy gating is family-based.
        "method_confirms": len(confirms),
        "method_fails": len(fails),
        "method_line": (
            " + ".join(f["label"] for f in confirms)
            if confirms else "No independent evidence family confirmed"
        ),
    })
    return out


def allows_buy(row: Mapping[str, Any]) -> bool:
    if "allows_recommend" in row:
        return bool(row.get("allows_recommend"))
    return bool(attach_ensemble(row).get("allows_recommend"))


def sort_key(card: Mapping[str, Any]) -> tuple:
    """Tier, then independent families, then scanner score. Not a weighted soup."""
    tier = str(card.get("reco_tier") or TIER_WATCH)
    confirms = 0
    try:
        confirms = int(card.get("family_confirms") or card.get("method_confirms") or 0)
    except (TypeError, ValueError):
        confirms = 0
    score = _f(card.get("score")) or 0.0
    return (TIER_RANK.get(tier, 9), -confirms, -score, str(card.get("symbol") or ""))


def attach_expert_layer(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    painted = attach_experts(rows)
    return [attach_ensemble(row) for row in painted]


def ensemble_summary(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    high = [c for c in cards if c.get("reco_tier") == TIER_HIGH]
    good = [c for c in cards if c.get("reco_tier") == TIER_GOOD]
    watch = [c for c in cards if c.get("reco_tier") == TIER_WATCH]
    avoid = [c for c in cards if c.get("reco_tier") == TIER_AVOID]
    empty = len(high) == 0
    family_ids: dict[str, int] = {}
    method_ids: dict[str, int] = {}
    checked_rows = 0
    for card in cards:
        checked_rows += 1
        for fam in card.get("families") or []:
            key = str(fam.get("label") or fam.get("id") or "").strip()
            if key:
                family_ids[key] = family_ids.get(key, 0) + (1 if fam.get("status") == "pass" else 0)
        for method in card.get("methods") or []:
            key = str(method.get("label") or method.get("id") or "").strip()
            if key:
                method_ids[key] = method_ids.get(key, 0) + (1 if method.get("status") == "pass" else 0)
    family_line = ", ".join(sorted(family_ids)[:8]) or "trend, breakout, quality, volume"
    empty_detail = ""
    if empty:
        empty_detail = (
            f"Checked {checked_rows} scored name{'s' if checked_rows != 1 else ''} across "
            f"{family_line}. Independent evidence families did not agree on a ready, "
            f"non-extended high-conviction setup. Empty is a successful output — the desk "
            f"does not invent a Buy list. Good setups: {len(good)}; watch: {len(watch)}."
        )
    return {
        "high_conviction_count": len(high),
        "good_setup_count": len(good),
        "watch_count": len(watch),
        "avoid_count": len(avoid),
        "checked_rows": checked_rows,
        "families_checked": sorted(family_ids),
        "methods_checked": sorted(method_ids),
        "empty_high_conviction": empty,
        "empty_line": (
            "NO HIGH-CONVICTION OPPORTUNITY"
            if empty else
            f"{len(high)} high-conviction name{'s' if len(high) != 1 else ''}"
        ),
        "empty_detail": empty_detail,
    }


def confirm_finalists(
    cards: Sequence[Mapping[str, Any]],
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Cache-only StockResearchEngine on a small finalist set. Never scrapes."""
    pool = [
        c for c in cards
        if c.get("reco_tier") in {TIER_HIGH, TIER_GOOD} and c.get("symbol")
    ][: max(0, int(limit))]
    if not pool:
        return [dict(c) for c in cards]
    try:
        from product.due_diligence.research_engine import StockResearchEngine
        engine = StockResearchEngine()
    except Exception:
        return [dict(c) for c in cards]
    by_sym: dict[str, dict[str, Any]] = {}
    for card in pool:
        sym = str(card.get("symbol") or "").upper()
        if not sym or sym in by_sym:
            continue
        try:
            report = engine.investigate(
                sym,
                scan_payload=scan_payload,
                long_term_payload=long_term_payload,
            )
        except Exception:
            continue
        quality = dict(report.get("fundamental_quality") or {})
        decision = dict(report.get("decision_coverage") or {})
        flags = list(report.get("red_flags") or [])
        vs = str(report.get("vs_technical_setup") or "")
        by_sym[sym] = {
            "deep_confirm": True,
            "fundamental_confirmation": (
                "SUPPORT" if "SUPPORT" in vs.upper()
                else "CONTRADICT" if "CONTRADICT" in vs.upper()
                else "UNMEASURED" if not vs or vs.upper() in {"UNMEASURED", "NEUTRAL"}
                else vs
            ),
            "research_decision_coverage": decision.get("coverage_pct") or report.get("decision_coverage_pct"),
            "research_quality_label": quality.get("label") or "Unmeasured",
            "research_red_flags": len(flags),
            "research_engine": "StockResearchEngine",
        }
    out: list[dict[str, Any]] = []
    for card in cards:
        painted = dict(card)
        extra = by_sym.get(str(painted.get("symbol") or "").upper())
        if extra:
            painted.update(extra)
        else:
            painted.setdefault("deep_confirm", False)
        out.append(painted)
    return out
