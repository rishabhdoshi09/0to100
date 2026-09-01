"""Final selection authority over the existing recommendation ensemble.

No new scanner. Existing nominations are allowed to remain recommendations only
when observed outcomes and company due diligence do not contradict them.
Positive learned/DD evidence can support a nomination; neither can create a BUY.
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

EMPIRICAL_MIN_N = 30
DD_MIN_DECISION_COVERAGE = 50.0
DD_MIN_INTELLIGENCE_COVERAGE = 50.0
DD_MIN_INTELLIGENCE_SCORE = 50.0
TIER_HIGH, TIER_GOOD, TIER_WATCH, TIER_AVOID = "high_conviction", "good_setup", "watch", "avoid"
TIER_LABELS = {TIER_HIGH: "High Conviction", TIER_GOOD: "Good Setup", TIER_WATCH: "Watch", TIER_AVOID: "Avoid / Conflict"}
_TIER_RANK = {TIER_HIGH: 0, TIER_GOOD: 1, TIER_WATCH: 2, TIER_AVOID: 3}


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""): return None
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int:
    try: return int(value or 0)
    except (TypeError, ValueError): return 0


def empirical_edge(card: Mapping[str, Any]) -> dict[str, Any]:
    panel = card.get("evidence_panel") if isinstance(card.get("evidence_panel"), Mapping) else {}
    case = card.get("case") if isinstance(card.get("case"), Mapping) else {}
    ev_n = _i(card.get("ev_n") or panel.get("sample_size"))
    ev_lb = _f(card.get("ev_lb_pct") if card.get("ev_lb_pct") is not None else panel.get("ev_lb_pct"))
    case_n = _i(card.get("case_n_similar") or case.get("n_similar"))
    case_exp = _f(card.get("case_expectancy_r") if card.get("case_expectancy_r") is not None else case.get("expectancy_r"))
    known: list[tuple[str, int, float]] = []
    if ev_n >= EMPIRICAL_MIN_N and ev_lb is not None: known.append(("live_edge", ev_n, ev_lb))
    if case_n >= EMPIRICAL_MIN_N and case_exp is not None: known.append(("similar_cases", case_n, case_exp))
    evidence: list[str] = []
    if ev_n: evidence.append(f"Live edge n={ev_n}" + (f" · conservative EV {ev_lb:+.2f}%" if ev_lb is not None else " · EV unavailable"))
    if case_n: evidence.append(f"Similar cases n={case_n}" + (f" · expectancy {case_exp:+.2f}R" if case_exp is not None else " · expectancy unavailable"))
    if not known:
        return {"status": "unknown", "sample_sufficient": False, "negative": False, "positive": False,
                "evidence": evidence or [f"Need ≥{EMPIRICAL_MIN_N} settled comparable outcomes"]}
    negative = any(value < 0 for _, _, value in known)
    positive = all(value > 0 for _, _, value in known)
    return {"status": "fail" if negative else "pass" if positive else "neutral", "sample_sufficient": True,
            "negative": negative, "positive": positive, "evidence": evidence,
            "inputs": [{"source": s, "n": n, "value": v} for s, n, v in known]}


def _downgrade(card: MutableMapping[str, Any], *, avoid: bool, reason: str) -> None:
    tier = TIER_AVOID if avoid or str(card.get("reco_tier") or "") == TIER_AVOID else TIER_WATCH
    card["reco_tier"], card["reco_tier_label"], card["allows_recommend"] = tier, TIER_LABELS[tier], False
    card["action_badge"] = "Avoid" if tier == TIER_AVOID else "Watch"
    for key in ("blockers", "conflicts"):
        rows = [str(x) for x in (card.get(key) or []) if x]
        if reason not in rows: rows.append(reason)
        card[key] = rows[:8]


def apply_empirical_gate(card: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(card)
    learned = empirical_edge(out)
    out["selection_learning"] = learned
    families = [dict(x) for x in (out.get("families") or []) if isinstance(x, Mapping) and str(x.get("id") or "") != "empirical_edge"]
    families.append({"id": "empirical_edge", "label": "Observed Edge", "status": learned["status"],
                     "strength": "Strong" if learned["status"] == "pass" else "Weak" if learned["status"] == "fail" else "Unknown",
                     "experts": ["live_forward_memory"] if learned["sample_sufficient"] else [], "evidence": learned["evidence"][:4]})
    out["families"] = families
    if learned["negative"] and str(out.get("reco_tier") or "") in {TIER_HIGH, TIER_GOOD}:
        _downgrade(out, avoid=False, reason="Observed Edge is sufficiently sampled and negative; recommendation blocked.")
    return out


def due_diligence_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    decision = report.get("decision_coverage") if isinstance(report.get("decision_coverage"), Mapping) else {}
    intel = report.get("fundamental_intelligence") if isinstance(report.get("fundamental_intelligence"), Mapping) else {}
    dc = _f(decision.get("coverage_pct") or report.get("decision_coverage_pct"))
    score, cov = _f(intel.get("score")), _f(intel.get("coverage_pct"))
    vs = str(report.get("vs_technical_setup") or "UNMEASURED").upper()
    breakers = [x for x in (report.get("thesis_breakers") or []) if isinstance(x, Mapping) and str(x.get("severity") or "").lower() in {"critical", "high", "severe"}]
    reasons: list[str] = []
    hard = False
    if "CONTRADICT" in vs:
        reasons.append(f"Due Diligence {vs.replace('_', ' ')} the technical setup")
        hard = "STRONGLY" in vs
    if breakers:
        reasons.append(f"{len(breakers)} critical thesis breaker(s) on file"); hard = True
    if dc is None or dc < DD_MIN_DECISION_COVERAGE: reasons.append("Decision coverage insufficient" if dc is None else f"Decision coverage {dc:.0f}% < {DD_MIN_DECISION_COVERAGE:.0f}%")
    if cov is None or cov < DD_MIN_INTELLIGENCE_COVERAGE: reasons.append("Fundamental Intelligence coverage insufficient" if cov is None else f"Fundamental Intelligence coverage {cov:.0f}% < {DD_MIN_INTELLIGENCE_COVERAGE:.0f}%")
    if score is not None and score < DD_MIN_INTELLIGENCE_SCORE: reasons.append(f"Fundamental Intelligence {score:.0f}/100 < {DD_MIN_INTELLIGENCE_SCORE:.0f}/100")
    if not reasons and "SUPPORT" not in vs: reasons.append(f"Due Diligence confirmation is {vs}, not SUPPORT")
    return {"passed": not reasons, "hard_reject": hard, "status": "pass" if not reasons else "fail",
            "decision_coverage_pct": dc, "fundamental_intelligence_score": score,
            "fundamental_intelligence_coverage_pct": cov, "vs_technical_setup": vs,
            "fundamental_confirmation": report.get("fundamental_confirmation") or vs,
            "critical_breakers": len(breakers), "delivery_state": report.get("delivery_state") or "FRESH",
            "snapshot_saved_at": report.get("snapshot_saved_at"), "reasons": reasons, "engine": "StockResearchEngine"}


def apply_due_diligence_gate(card: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(card)
    if str(out.get("reco_tier") or "") not in {TIER_HIGH, TIER_GOOD}:
        out.setdefault("deep_confirm", False); return out
    symbol = str(out.get("symbol") or "").strip().upper()
    try:
        from product.due_diligence import build_due_diligence
        report = build_due_diligence(symbol)
    except Exception as exc:
        out["deep_confirm"] = False
        out["due_diligence_gate"] = {"status": "unavailable", "passed": False, "reasons": [f"{type(exc).__name__}: {exc}"]}
        _downgrade(out, avoid=False, reason="Due Diligence unavailable; top recommendation blocked")
        return out
    gate = due_diligence_gate(report)
    out.update({"deep_confirm": True, "due_diligence_gate": gate,
                "fundamental_confirmation": gate["fundamental_confirmation"],
                "research_decision_coverage": gate["decision_coverage_pct"],
                "fundamental_intelligence_score": gate["fundamental_intelligence_score"],
                "fundamental_intelligence_coverage_pct": gate["fundamental_intelligence_coverage_pct"],
                "research_engine": "StockResearchEngine"})
    if not gate["passed"]:
        _downgrade(out, avoid=bool(gate["hard_reject"]), reason="Due Diligence gate: " + "; ".join(gate["reasons"]))
    return out


def apply_card_selection_authority(card: Mapping[str, Any], *, run_due_diligence: bool = True) -> dict[str, Any]:
    out = apply_empirical_gate(card)
    return apply_due_diligence_gate(out) if run_due_diligence else out


def apply_workspace_selection_authority(payload: Mapping[str, Any], *, max_due_diligence: int = 8) -> dict[str, Any]:
    out = dict(payload); categories: list[dict[str, Any]] = []; all_cards: list[dict[str, Any]] = []; dd_left = max(0, int(max_due_diligence))
    for raw_cat in payload.get("categories") or []:
        if not isinstance(raw_cat, Mapping): continue
        cat = dict(raw_cat); cards: list[dict[str, Any]] = []
        for raw in raw_cat.get("cards") or []:
            if not isinstance(raw, Mapping): continue
            card = apply_empirical_gate(raw)
            if str(card.get("reco_tier") or "") in {TIER_HIGH, TIER_GOOD}:
                if dd_left > 0: card = apply_due_diligence_gate(card); dd_left -= 1
                else: _downgrade(card, avoid=False, reason="Due Diligence finalist cap reached; unverified finalist blocked")
            cards.append(card); all_cards.append(card)
        cards.sort(key=lambda c: (_TIER_RANK.get(str(c.get("reco_tier") or TIER_WATCH), 9), -_i(c.get("family_confirms") or c.get("method_confirms")), -(_f(c.get("score")) or 0), str(c.get("symbol") or "")))
        cat["cards"], cat["count"] = cards, len(cards); categories.append(cat)
    out["categories"] = categories
    counts = {tier: sum(1 for c in all_cards if c.get("reco_tier") == tier) for tier in (TIER_HIGH, TIER_GOOD, TIER_WATCH, TIER_AVOID)}
    ensemble = dict(out.get("ensemble") or {})
    ensemble.update({"high_conviction_count": counts[TIER_HIGH], "good_setup_count": counts[TIER_GOOD], "watch_count": counts[TIER_WATCH], "avoid_count": counts[TIER_AVOID],
                     "empty_high_conviction": counts[TIER_HIGH] == 0,
                     "empty_line": "NO HIGH-CONVICTION OPPORTUNITY" if counts[TIER_HIGH] == 0 else f"{counts[TIER_HIGH]} high-conviction name{'s' if counts[TIER_HIGH] != 1 else ''}",
                     "selection_authority": True})
    out["ensemble"] = ensemble
    out["selection_authority"] = {"applied": True, "empirical_min_n": EMPIRICAL_MIN_N,
                                   "dd_min_decision_coverage_pct": DD_MIN_DECISION_COVERAGE,
                                   "dd_min_fundamental_intelligence_coverage_pct": DD_MIN_INTELLIGENCE_COVERAGE,
                                   "dd_min_fundamental_intelligence_score": DD_MIN_INTELLIGENCE_SCORE,
                                   "due_diligence_finalist_cap": max(0, int(max_due_diligence)),
                                   "principle": "Learning and Due Diligence may veto or downgrade; neither may create a Buy."}
    return out
