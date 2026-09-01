"""Point-in-time decision context for paper autopilot.

The autopilot executes Selection Authority, not a scanner BUY chip. This snapshot
is the full evidence pack frozen at decision time. Missing stays missing.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.reco_ensemble import TIER_GOOD, TIER_HIGH


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _method_map(card: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in card.get("methods") or []:
        if not isinstance(raw, Mapping):
            continue
        mid = str(raw.get("id") or "")
        if mid:
            out[mid] = dict(raw)
    return out


def extension_bucket(extension_pct: float | None) -> str:
    if extension_pct is None:
        return ""
    if extension_pct > 8:
        return "extended_gt_8"
    if extension_pct > 4:
        return "extended_4_8"
    return "extended_le_4"


def rs_bucket(percentile: float | None) -> str:
    if percentile is None:
        return ""
    if percentile >= 80:
        return "rs_top_decile"
    if percentile >= 60:
        return "rs_strong"
    if percentile >= 40:
        return "rs_mid"
    return "rs_weak"


def liquidity_bucket(volume_ratio: float | None) -> str:
    if volume_ratio is None:
        return ""
    if volume_ratio >= 1.5:
        return "liq_high"
    if volume_ratio >= 0.7:
        return "liq_ok"
    return "liq_thin"


def volatility_bucket(atr_pct: float | None) -> str:
    if atr_pct is None:
        return ""
    if atr_pct >= 4:
        return "vol_high"
    if atr_pct >= 2:
        return "vol_normal"
    return "vol_low"


def entry_quality(card: Mapping[str, Any]) -> str:
    state = str(card.get("entry_state") or "")
    if bool(card.get("chase_risk")) or state == "extended":
        return "chase"
    if state in {"ready", "pullback", "retest"}:
        return state
    if state in {"breakout", "gap", "pocket_pivot"}:
        return state
    return state or "unknown"


def dd_status(card: Mapping[str, Any]) -> str:
    explicit = str(card.get("dd_verdict") or card.get("dd_status") or "").strip().upper()
    if explicit:
        return explicit
    methods = _method_map(card)
    funds = methods.get("funds") or {}
    return str(funds.get("status") or "UNKNOWN").upper()


def snapshot(
    card: Mapping[str, Any],
    *,
    book=None,
    regime: str = "",
) -> dict[str, Any]:
    """Freeze the evidence the autopilot is allowed to see. Never invents values."""
    row = dict(card)
    methods = _method_map(row)
    ext = _f(row.get("extension_pct") or row.get("ema20_extension_pct") or row.get("from_ema20_pct"))
    vol = _f(row.get("volume_ratio"))
    rs = _f(row.get("rs_percentile") or row.get("rs") or row.get("rs_rank"))
    atr = _f(row.get("atr_pct") or row.get("atr14_pct"))
    missing: list[str] = []
    for key, label in (
        ("entry", "entry"),
        ("stop", "stop"),
        ("target", "target"),
        ("setup_label", "setup"),
        ("sector", "sector"),
    ):
        if row.get(key) in (None, "", "—"):
            missing.append(label)
    if not methods:
        missing.append("method_panel")
    if dd_status(row) in {"", "UNKNOWN"}:
        missing.append("due_diligence")
    ev = methods.get("ev") or {}
    case = methods.get("case") or {}
    empirical = {
        "live_ev_status": ev.get("status") or "unknown",
        "live_ev_detail": ev.get("detail") or "",
        "live_ev_points": ev.get("points"),
        "case_status": case.get("status") or "unknown",
        "case_detail": case.get("detail") or "",
        "expectancy_R": _f(row.get("expectancy_R") or row.get("empirical_expectancy_R") or ev.get("expectancy_R")),
        "hit_rate": _f(row.get("hit_rate") or row.get("empirical_hit_rate")),
        "avg_win_R": _f(row.get("avg_win_R")),
        "avg_loss_R": _f(row.get("avg_loss_R")),
        "sample_size": int(row.get("empirical_n") or ev.get("sample_size") or 0 or 0),
        "confidence": str(row.get("empirical_confidence") or (
            "INSUFFICIENT_EVIDENCE" if str(ev.get("status") or "") == "unknown" else "MEASURED"
        )),
    }
    held: list[str] = []
    open_risk = 0.0
    capital = None
    if book is not None:
        try:
            held = [str(getattr(p, "symbol", "") or "") for p in getattr(book, "open", {}).values()]
            open_risk = float(book.open_risk()) if hasattr(book, "open_risk") else 0.0
            capital = float(getattr(book, "capital", 0.0) or 0.0)
        except Exception:
            held = []
    portfolio = {
        "open_symbols": held,
        "open_count": len(held),
        "open_risk": open_risk,
        "capital": capital,
        "available_capital": None if capital is None else max(0.0, capital - open_risk),
        "regime": regime,
    }
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "setup_label": str(row.get("setup_label") or row.get("primary_thesis") or ""),
        "primary_thesis": row.get("primary_thesis") or "",
        "sector": str(row.get("sector") or ""),
        "reco_tier": str(row.get("reco_tier") or ""),
        "entry_state": str(row.get("entry_state") or ""),
        "entry_quality": entry_quality(row),
        "chase_risk": bool(row.get("chase_risk")),
        "extension_pct": ext,
        "extension": extension_bucket(ext),
        "volume_ratio": vol,
        "liquidity": liquidity_bucket(vol),
        "atr_pct": atr,
        "volatility": volatility_bucket(atr),
        "rs_percentile": rs,
        "rs_bucket": rs_bucket(rs),
        "regime": regime,
        "dd_status": dd_status(row),
        "stock_quality": row.get("stock_quality") or "",
        "fundamental_confirmation": row.get("fundamental_confirmation"),
        "research_quality_label": row.get("research_quality_label"),
        "allows_recommend": row.get("allows_recommend"),
        "family_confirms": row.get("family_confirms") or row.get("method_confirms") or 0,
        "methods": {k: {"status": v.get("status"), "points": v.get("points"), "detail": v.get("detail")} for k, v in methods.items()},
        "method_tape": (methods.get("tape") or {}).get("status") or "unknown",
        "method_sepa": (methods.get("sepa") or {}).get("status") or "unknown",
        "method_funds": (methods.get("funds") or {}).get("status") or "unknown",
        "method_trend": (methods.get("trend") or {}).get("status") or "unknown",
        "method_rs": (methods.get("rs") or {}).get("status") or "unknown",
        "method_ev": (methods.get("ev") or {}).get("status") or "unknown",
        "method_case": (methods.get("case") or {}).get("status") or "unknown",
        "method_sector": (methods.get("sector") or {}).get("status") or "unknown",
        "empirical": empirical,
        "portfolio": portfolio,
        "missing_evidence": missing,
        "conflicts": list(row.get("conflicts") or []),
        "entry": _f(row.get("entry") or row.get("entry_price") or row.get("cmp")),
        "stop": _f(row.get("stop") or row.get("stop_price")),
        "target": _f(row.get("target") or row.get("target_price")),
    }


def score_breakdown(
    card: Mapping[str, Any],
    policy: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Interpretable rank among already-eligible names. Not a BUY oracle."""
    ctx = dict(context or snapshot(card))
    policy = dict(policy or {})
    parts: list[dict[str, Any]] = []
    tier = str(card.get("reco_tier") or ctx.get("reco_tier") or "")
    base = 90.0 if tier == TIER_HIGH else 75.0 if tier == TIER_GOOD else 20.0
    parts.append({"key": "tier", "points": base, "source": f"reco_tier={tier or 'missing'}"})
    try:
        confirms = float(ctx.get("family_confirms") or 0)
    except (TypeError, ValueError):
        confirms = 0.0
    parts.append({"key": "evidence_families", "points": round(min(9.0, confirms * 3.0), 2), "source": f"confirms={int(confirms)}"})
    score = _f(card.get("score")) or 0.0
    parts.append({"key": "desk_score", "points": round(min(5.0, score / 20.0), 2), "source": "card.score (capped)"})
    dd = str(ctx.get("dd_status") or "UNKNOWN")
    if dd in {"PASS", "SUPPORT", "OK"}:
        parts.append({"key": "dd", "points": 3.0, "source": f"dd={dd}"})
    elif dd in {"FAIL", "FAILED", "BLOCK", "AVOID"}:
        parts.append({"key": "dd", "points": -8.0, "source": f"dd={dd}"})
    else:
        parts.append({"key": "dd", "points": 0.0, "source": f"dd={dd or 'UNKNOWN'} (missing stays missing)"})
    rs = ctx.get("rs_bucket") or ""
    parts.append({
        "key": "relative_strength",
        "points": 3.0 if rs == "rs_top_decile" else 1.5 if rs == "rs_strong" else 0.0,
        "source": rs or "rs missing",
    })
    eq = str(ctx.get("entry_quality") or "")
    parts.append({
        "key": "entry_quality",
        "points": 2.0 if eq in {"ready", "pullback", "retest"} else -6.0 if eq == "chase" else 0.0,
        "source": eq or "entry missing",
    })
    effect = str(policy.get("final_effect") or "NEUTRAL")
    policy_pts = 4.0 if effect == "SUPPORT" else -8.0 if effect == "PENALIZE" else -50.0 if effect == "BLOCK" else 0.0
    parts.append({
        "key": "empirical_policy",
        "points": policy_pts,
        "source": (
            f"{effect} n={policy.get('sample_size') or 0} "
            f"edge={policy.get('learned_edge_score') if policy.get('learned_edge_score') is not None else policy.get('expectancy_difference_R')}"
        ),
    })
    total = round(sum(float(p["points"]) for p in parts), 2)
    return {
        "selection_rank": total,
        "parts": parts,
        "note": "Rank among eligible names only. Hard blockers stay blockers. This is not an AI buy score.",
        "invents_buy": False,
    }


def explain(
    *,
    decision: str,
    reason_code: str,
    card: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    breakdown: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic why-took / why-not. An LLM must not manufacture this."""
    ctx = dict(context or snapshot(card))
    policy = dict(policy or {})
    plus: list[str] = []
    minus: list[str] = []
    setup = ctx.get("setup_label") or "setup unknown"
    sector = ctx.get("sector") or "sector unknown"
    if ctx.get("reco_tier") in {TIER_HIGH, TIER_GOOD}:
        plus.append(f"desk tier {ctx.get('reco_tier')}")
    else:
        minus.append(f"tier {ctx.get('reco_tier') or 'missing'} is not auto-enter")
    if int(ctx.get("family_confirms") or 0) >= 2:
        plus.append(f"{ctx.get('family_confirms')} evidence families")
    for mid in ("tape", "sepa", "funds", "trend", "rs", "sector"):
        status = str(ctx.get(f"method_{mid}") or "unknown")
        if status == "pass":
            plus.append(f"{mid} pass")
        elif status == "fail":
            minus.append(f"{mid} fail")
    dd = str(ctx.get("dd_status") or "UNKNOWN")
    if dd in {"PASS", "SUPPORT", "OK"}:
        plus.append(f"DD {dd}")
    elif dd in {"FAIL", "FAILED", "BLOCK", "AVOID"}:
        minus.append(f"DD {dd}")
    if ctx.get("chase_risk") or ctx.get("entry_quality") == "chase":
        minus.append("chase / extension")
    elif ctx.get("entry_quality") in {"ready", "pullback", "retest"}:
        plus.append(f"entry {ctx.get('entry_quality')}")
    effect = str(policy.get("final_effect") or "NEUTRAL")
    if effect == "SUPPORT":
        plus.append(f"learned policy SUPPORT n={policy.get('sample_size') or 0}")
    elif effect == "PENALIZE":
        minus.append("learned policy PENALIZE")
    elif effect == "BLOCK":
        minus.append("learned policy BLOCK")
    for item in ctx.get("missing_evidence") or []:
        minus.append(f"missing {item}")
    entry = ctx.get("entry")
    action = f"Leave alone ({reason_code})"
    if decision == "ENTER_NOW":
        action = f"Enter now near ₹{entry}" if entry else "Enter now"
    elif decision in {"WAIT", "WAIT_FOR_ENTRY"}:
        action = f"Watch for acceptable entry near ₹{entry}" if entry else "Wait for a valid entry"
    elif decision == "WATCH":
        action = "Watch only — not an auto-enter tier"
    title = {
        "ENTER_NOW": "WHY BOT TOOK THIS",
        "WAIT": "WHY BOT DID NOT TAKE THIS",
        "WATCH": "WHY BOT DID NOT TAKE THIS",
        "BLOCK": "WHY BOT DID NOT TAKE THIS",
        "PORTFOLIO_BLOCK": "WHY BOT DID NOT TAKE THIS",
        "NO_TRADE": "WHY BOT DID NOT TAKE THIS",
    }.get(decision, "WHY")
    return {
        "title": title,
        "decision": decision,
        "reason_code": reason_code,
        "plus": plus,
        "minus": minus,
        "action": action,
        "setup": setup,
        "sector": sector,
        "selection_rank": (breakdown or {}).get("selection_rank"),
        "policy_effect": effect,
        "invents_buy": False,
    }
