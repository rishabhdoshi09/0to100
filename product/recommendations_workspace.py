"""Reco-style recommendation desk — projections only, no new scans or scrapes.

Maps QuantTerm evidence into named research categories (Wealth Builders,
Super Trends, Momentum Breakouts, Recovery Setups) plus Active/Closed
lifecycle strips from long-term + signal-outcome trackers.

Productisation (Reco lesson, QuantTerm engine):
  complexity stays inside the scanner / EV / breadth / live-edge;
  the customer sees a decision card (entry, target, stop, why now,
  what changes our mind) with See Evidence for the machinery.

Honest rules:
  - Empty category → empty list (never fabricate picks)
  - Prices/upside from real entry/target/price fields only
  - CMP freshness tags come from live_technicals when available
  - Each scan symbol gets ONE primary category (no multi-bucket spam)
  - Expected payoff / Strong evidence require live sample size (≥30)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.breakout_quality import RSI_HARD, passes_volume_floor
from product.decision_card import (
    HORIZON_BY_CATEGORY,
    attach_live_ev,
    build_desk_context,
    decision_surface,
)
from product.radar_workspace import (
    enrich_long_term_row,
    enrich_scan_row,
    is_long_term_pick,
    is_sniper_breakout_candidate,
)
from product.reco_methods import (
    allows_buy,
    attach_method_scores,
    attach_research_overlays,
    sort_key as method_sort_key,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "logs" / "product" / "market_reports"

# Reco-Wealth analogues — QuantTerm evidence names (not a brand clone).
CATEGORIES: tuple[dict[str, str], ...] = (
    {
        "id": "wealth_builders",
        "label": "Wealth Builders",
        "blurb": "Long-term quality / GARP — compounders with ≥50% fundamental coverage, ranked by independent methods not SEPA alone.",
        "icon": "compound",
    },
    {
        "id": "super_trends",
        "label": "Super Trends",
        "blurb": "Momentum without chase — Buy needs two methods (tape, trend, RS, funds, SEPA, live EV).",
        "icon": "trend",
    },
    {
        "id": "momentum_breakouts",
        "label": "Momentum Breakouts",
        "blurb": "Sniper/graded tape plus a second method (funds, SEPA, RS, trend, live EV). SEPA alone is not a Buy.",
        "icon": "breakout",
    },
    {
        "id": "recovery_setups",
        "label": "Recovery Setups",
        "blurb": "True turnarounds only — double-bottom / accumulation (not coils), ranked by independent methods.",
        "icon": "recovery",
    },
)

# True recovery = turnaround evidence. Coils (CUP_HANDLE / POCKET_PIVOT / NR7)
# are setups, not recoveries — they must NOT land here.
_RECOVERY_SIGNALS = frozenset({"DOUBLE_BOTTOM", "ACCUMULATION"})
_ACTIONABLE_MOMENTUM = frozenset({
    "strong_actionable", "steady_leadership", "improving",
})
_WEALTH_CLASSES = frozenset({
    "QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE",
})
# Soft RSI ceiling on the research desk (hard scanner blow-off stays 82).
_DESK_RSI_SOFT = 72.0
_BUCKET_CAP = 40


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _signals(row: Mapping[str, Any]) -> set[str]:
    return {str(x).upper() for x in (row.get("signals") or [])}


def risk_tier(row: Mapping[str, Any]) -> str:
    """Low / Medium / High from existing flags — never invented."""
    if bool(row.get("chase_risk")):
        return "High"
    label = str(row.get("risk_label") or "").lower()
    if "chase" in label or "avoid" in label or "high" in label:
        return "High"
    rsi = _f(row.get("rsi"))
    if rsi >= 75:
        return "High"
    if rsi >= 65 or "pullback" in label or "review" in label or "medium" in label:
        return "Medium"
    flags = row.get("risk_flags") or []
    if flags:
        return "Medium"
    return "Low"


def upside_metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    """% move from entry and remaining room to target — None when inputs missing."""
    price = _f(row.get("price"))
    entry = _f(row.get("entry") or row.get("entry_price"))
    target = _f(row.get("target") or row.get("target_price"))
    from_entry = None
    to_target = None
    if entry > 0 and price > 0:
        from_entry = round((price / entry - 1.0) * 100.0, 1)
    if target > 0 and price > 0:
        to_target = round((target / price - 1.0) * 100.0, 1)
    return {
        "upside_from_entry_pct": from_entry,
        "upside_to_target_pct": to_target,
        "entry": entry or None,
        "target": target or None,
        "cmp": price or None,
    }


def _known_volume_ratio(row: Mapping[str, Any]) -> float | None:
    """Positive volume_ratio only. Stored 0 / missing is unknown, not a reject."""
    raw = row.get("volume_ratio")
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _listing_volume_ok(row: Mapping[str, Any]) -> bool:
    """Known volume below 0.7× rejects. Unknown volume does not hide the name."""
    vol = _known_volume_ratio(row)
    if vol is None:
        return True
    return vol >= 0.7


def _home_breakout_visible(row: Mapping[str, Any]) -> bool:
    """Same breakout names Home already lists — not every Ready-to-trade momentum BUY."""
    if bool(row.get("chase_risk")):
        return False
    sigs = _signals(row)
    status = str(row.get("status") or "")
    grade = str(row.get("breakout_grade") or "").upper()
    if sigs & {"PRE_BREAKOUT", "BREAKOUT_52W", "BREAKOUT_RES"}:
        return True
    if status == "Watch for breakout":
        return True
    if grade in {"A", "B"}:
        return True
    return False


def _action_badge(row: Mapping[str, Any], *, category_id: str = "") -> str:
    cls = str(row.get("classification") or "")
    if category_id == "wealth_builders":
        if cls == "QUALITY_BUT_EXPENSIVE":
            return "Research"
        if cls in {"QUALITY_COMPOUNDER", "GARP_CANDIDATE"}:
            return "Hold / Research"
    verdict = str(row.get("verdict") or "").upper()
    status = str(row.get("status") or "")
    grade = str(row.get("breakout_grade") or "").upper()
    if category_id == "momentum_breakouts":
        # Tape (sniper/grade) can nominate; Buy still needs two independent methods.
        if is_sniper_breakout_candidate(row) or grade in {"A", "B"}:
            if verdict in {"BUY", "STRONG BUY"} or status == "Ready to trade":
                return "Buy" if allows_buy(row) else "Watch"
        return "Watch"
    if verdict in {"BUY", "STRONG BUY"} or status == "Ready to trade":
        return "Buy" if allows_buy(row) else "Watch"
    if "breakout" in status.lower() or grade in {"A", "B"}:
        return "Watch"
    if cls in {"QUALITY_COMPOUNDER", "GARP_CANDIDATE"}:
        return "Hold / Research"
    return "Watch"


def card_from_row(
    row: Mapping[str, Any],
    *,
    category_id: str,
    category_label: str,
    qualify_reason: str = "",
    evidence_tags: Sequence[str] | None = None,
    market_ctx: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scored = attach_method_scores(row) if row.get("methods") is None else dict(row)
    ups = upside_metrics(scored)
    tags = [str(t) for t in (evidence_tags or []) if t]
    for item in scored.get("methods") or []:
        if item.get("status") == "pass" and item.get("label"):
            tags.append(str(item["label"]))
    tags = list(dict.fromkeys(tags))
    reason = qualify_reason or str(scored.get("reason") or "")
    confirms = int(scored.get("method_confirms") or 0)
    if confirms:
        line = str(scored.get("method_line") or "")
        extra = f"{confirms} methods: {line}" if line else f"{confirms} independent methods"
        reason = f"{reason} · {extra}" if reason else extra
    badge = _action_badge(scored, category_id=category_id)
    surface = decision_surface(
        scored,
        category_id=category_id,
        action_badge=badge,
        qualify_reason=reason,
        evidence_tags=tags,
        market_ctx=market_ctx,
    )
    card = {
        "symbol": str(scored.get("symbol") or "").upper(),
        "company": str(scored.get("company") or scored.get("symbol") or ""),
        "category_id": category_id,
        "category_label": category_label,
        "action_badge": badge,
        "risk_tier": risk_tier(scored),
        "risk_label": str(scored.get("risk_label") or risk_tier(scored)),
        "setup_label": str(scored.get("setup_label") or scored.get("status") or scored.get("classification") or ""),
        "sector": str(scored.get("sector") or "—"),
        "score": _f(scored.get("score") or scored.get("combined_score")),
        "rsi": _f(scored.get("rsi")) or None,
        "volume_ratio": _f(scored.get("volume_ratio")) or None,
        "price_tag": str(scored.get("price_tag") or ""),
        "tech_source": str(scored.get("tech_source") or ""),
        "reason": reason,
        "qualify_reason": reason,
        "evidence_tags": tags,
        "signals": sorted(_signals(scored)),
        "lifecycle": "active",
        "methods": list(scored.get("methods") or []),
        "method_confirms": confirms,
        "method_fails": int(scored.get("method_fails") or 0),
        "quality_score": scored.get("quality_score"),
        "method_line": str(scored.get("method_line") or ""),
        **ups,
        **surface,
    }
    return _attach_key_points(card, row=scored)


def _trend_structure_ok(row: Mapping[str, Any]) -> bool:
    """Pass when above a key SMA, or SMA unknown (don't invent a fail)."""
    above50 = row.get("above_sma50")
    above200 = row.get("above_sma200")
    if above50 is False and above200 is False:
        return False
    return True


def _is_momentum_breakout(row: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """Sniper / graded Buy, or Home-visible breakout Watch. Unknown volume does not hide."""
    if bool(row.get("chase_risk")):
        return False, "", []
    rsi_raw = row.get("rsi")
    if rsi_raw not in (None, ""):
        try:
            if float(rsi_raw) > RSI_HARD:
                return False, "", []
        except (TypeError, ValueError):
            pass
    if not _listing_volume_ok(row):
        return False, "", []
    grade = str(row.get("breakout_grade") or "").upper()
    sniper = is_sniper_breakout_candidate(row)
    visible = _home_breakout_visible(row)
    if not sniper and grade not in {"A", "B"} and not visible:
        return False, "", []
    state = str(row.get("breakout_state") or "")
    sigs = _signals(row)
    status = str(row.get("status") or "")
    cats = {str(c) for c in (row.get("categories") or [])}
    is_pre = (
        "PreBreakout" in cats
        or "PRE_BREAKOUT" in sigs
        or status == "Watch for breakout"
    )
    try:
        dist = float(row.get("pivot_distance_pct")) if row.get("pivot_distance_pct") is not None else None
    except (TypeError, ValueError):
        dist = None
    near_pivot = bool(is_pre and dist is not None and 0.0 <= dist <= 2.5)
    structure_ok = (
        state in {"near_breakout", "confirmed_breakout", "breakout_under_observation"}
        or grade in {"A", "B"}
        or near_pivot
        or bool(sigs & {"PRE_BREAKOUT", "BREAKOUT_52W", "BREAKOUT_RES"})
        or visible
    )
    # Reject ghost sniper hits with no breakout structure (state=not_in_breakout_lane).
    if not structure_ok:
        return False, "", []
    tags = sorted(sigs & {
        "PRE_BREAKOUT", "BREAKOUT_52W", "BREAKOUT_RES", "GOLDEN_CROSS", "VOL_SQUEEZE",
    })
    if grade:
        tags = [f"grade_{grade}"] + tags
    display_state = state
    if (not display_state or display_state == "not_in_breakout_lane") and near_pivot:
        display_state = "near_breakout"
    if (not display_state or display_state == "not_in_breakout_lane") and visible:
        display_state = "breakout_under_observation"
    if display_state and display_state != "not_in_breakout_lane":
        tags.append(display_state)
    vol = _known_volume_ratio(row)
    vol_bit = f" · vol {vol:.1f}×" if vol is not None else " · volume not on file"
    if sniper or grade in {"A", "B"}:
        why = (
            f"Sniper breakout{vol_bit}"
            + (f" · grade {grade}" if grade else "")
            + (
                f" · {display_state.replace('_', ' ')}"
                if display_state and display_state != "not_in_breakout_lane"
                else ""
            )
        )
    else:
        why = (
            f"Breakout on last scan{vol_bit} — Watch until sniper/grade confirms"
            + (
                f" · {display_state.replace('_', ' ')}"
                if display_state and display_state != "not_in_breakout_lane"
                else ""
            )
        )
    return True, why, tags


def _is_recovery(row: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """Double-bottom / accumulation only — coils are not recoveries."""
    if bool(row.get("chase_risk")):
        return False, "", []
    sigs = _signals(row)
    hits = sorted(sigs & _RECOVERY_SIGNALS)
    if not hits:
        return False, "", []
    rsi = _f(row.get("rsi"))
    if rsi > _DESK_RSI_SOFT:
        return False, "", []
    if str(row.get("verdict") or "").upper() == "AVOID":
        return False, "", []
    # Confirmed breakout already fired → not a recovery base anymore.
    if str(row.get("breakout_grade") or "").upper() in {"A", "B"}:
        return False, "", []
    if str(row.get("breakout_state") or "") == "confirmed_breakout":
        return False, "", []
    why = "Recovery · " + " + ".join(h.replace("_", " ").title() for h in hits)
    return True, why, hits


def _is_super_trend(row: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """Momentum leadership without chase / extension."""
    if bool(row.get("chase_risk")):
        return False, "", []
    if str(row.get("verdict") or "").upper() == "AVOID":
        return False, "", []
    rsi = _f(row.get("rsi"))
    if rsi > _DESK_RSI_SOFT:
        return False, "", []
    if not _trend_structure_ok(row):
        return False, "", []
    sigs = _signals(row)
    state = str(row.get("momentum_state") or "")
    has_mom_signal = "MOMENTUM" in sigs or "GOLDEN_CROSS" in sigs
    has_mom_state = state in _ACTIONABLE_MOMENTUM
    if not (has_mom_signal or has_mom_state):
        return False, "", []
    # Require real thrust — score alone used to spam mediocre names.
    score = _f(row.get("score"))
    mom5_raw = row.get("momentum_5d")
    mom5 = _f(mom5_raw) if mom5_raw is not None else None
    if mom5 is not None and mom5 < 0:
        return False, "", []  # Super Trends = rising leadership, not fading prints
    mom5_v = float(mom5 or 0.0)
    if score < 55 and mom5_v < 2.0 and state not in {"strong_actionable", "steady_leadership"}:
        return False, "", []
    if _f(row.get("volume_ratio")) > 0 and not passes_volume_floor(row):
        return False, "", []
    tags: list[str] = []
    if "MOMENTUM" in sigs:
        tags.append("MOMENTUM")
    if "GOLDEN_CROSS" in sigs:
        tags.append("GOLDEN_CROSS")
    if state and state != "not_momentum":
        tags.append(state)
    why = (
        f"Trend momentum · score {score:.0f}"
        + (f" · 5d {mom5_v:+.1f}%" if mom5 is not None else "")
        + (f" · {state.replace('_', ' ')}" if state in _ACTIONABLE_MOMENTUM else "")
    )
    return True, why, tags


def _wealth_qualify(row: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    if not is_long_term_pick(row):
        return False, "", []
    cls = str(row.get("classification") or "")
    cov = _f(row.get("fundamental_coverage"))
    tags = [cls, f"coverage_{cov:.0%}"]
    if cls == "QUALITY_BUT_EXPENSIVE":
        why = f"Quality but expensive · fund coverage {cov:.0%}"
    elif cls == "GARP_CANDIDATE":
        why = f"GARP candidate · fund coverage {cov:.0%}"
    else:
        why = f"Quality compounder · fund coverage {cov:.0%}"
    factors = [str(x) for x in (row.get("quality_factors") or [])[:2] if x]
    if factors:
        why = why + " · " + ", ".join(factors)
        tags.extend(factors)
    return True, why, tags


def primary_scan_category(row: Mapping[str, Any]) -> tuple[str, str, list[str]] | None:
    """Exclusive primary bucket for a scan row.

    Priority: Momentum Breakouts → Recovery → Super Trends.
    A symbol never appears in two scan categories.
    """
    ok, why, tags = _is_momentum_breakout(row)
    if ok:
        return "momentum_breakouts", why, tags
    ok, why, tags = _is_recovery(row)
    if ok:
        return "recovery_setups", why, tags
    ok, why, tags = _is_super_trend(row)
    if ok:
        return "super_trends", why, tags
    return None


def _empty_detail_for(
    category_id: str,
    *,
    cards: Sequence[Mapping[str, Any]],
    lt_rows: Sequence[Mapping[str, Any]],
    scan_rows: Sequence[Mapping[str, Any]],
) -> str:
    if cards:
        return ""
    if category_id == "wealth_builders":
        if not lt_rows:
            return "No long-term shortlist on file — run a long-term scan with fundamentals."
        needs = sum(
            1 for r in lt_rows
            if str(r.get("classification") or "") == "NEEDS_FUNDAMENTALS"
        )
        thin = sum(
            1 for r in lt_rows
            if str(r.get("classification") or "") in _WEALTH_CLASSES
            and _f(r.get("fundamental_coverage")) < 0.50
        )
        if needs == len(lt_rows):
            return (
                f"Long-term shortlist has {needs} name(s) but all are NEEDS_FUNDAMENTALS "
                "(coverage <50%). Refresh fundamentals, then re-run long-term scan."
            )
        if thin:
            return (
                "Quality classes present but fundamental coverage is below 50% — "
                "Wealth Builders stay empty until coverage is readable."
            )
        return (
            "No QUALITY_COMPOUNDER / GARP / QUALITY_BUT_EXPENSIVE with ≥50% coverage "
            "in the current long-term shortlist."
        )
    if category_id == "momentum_breakouts":
        return (
            "No BREAKOUT_52W / BREAKOUT_RES / PRE_BREAKOUT names (and no sniper/graded "
            f"breakouts) in this scan ({len(scan_rows)} rows). Known volume below 0.7× is "
            "excluded; missing volume is not treated as a fail."
        )
    if category_id == "recovery_setups":
        return (
            "No double-bottom / accumulation recoveries (coils like cup-handle / "
            "pocket-pivot / NR7 are excluded on purpose)."
        )
    if category_id == "super_trends":
        return (
            "No non-chase momentum leadership (MOMENTUM / actionable RS) above trend "
            "structure in this scan."
        )
    return "No matching setups in the current scan / long-term shortlist."


def _key_points_for_card(symbol: str, seeds: Sequence[str] | None = None) -> list[str]:
    """Plain-language points shown next to the stock — never invented prints."""
    points: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        line = " ".join(str(text or "").split())
        if not line or line.lower() in seen:
            return
        seen.add(line.lower())
        points.append(line)

    for item in seeds or []:
        add(str(item))
    try:
        from product.desk_note import MIX_SHIFT_DESKS
        frame = next(
            (f for f in MIX_SHIFT_DESKS if str(f.get("symbol") or "").upper() == symbol.upper()),
            None,
        )
    except Exception:
        frame = None
    if frame:
        add(str(frame.get("lens") or ""))
        for watch in list(frame.get("watch") or [])[:2]:
            add(str(watch))
    return points[:6]


def _attach_key_points(card: dict[str, Any], *, row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    seeds = list(card.get("why_now") or [])
    reason = str(card.get("reason") or "").strip()
    if reason:
        seeds.append(reason)
    card["key_points"] = _key_points_for_card(str(card.get("symbol") or ""), seeds)
    try:
        from product.case_memory import attach_case
        attach_case(card, row=row)
    except Exception:
        card.setdefault("case", {
            "n_similar": 0,
            "proven": False,
            "verdict": "unmeasured",
            "memory_line": "Case memory is unavailable on this snapshot.",
            "places_orders": False,
        })
    return card


def _empty_decision_fields(action_badge: str, category_id: str) -> dict[str, Any]:
    """Lifecycle rows still render on the Reco card without inventing prices/EV."""
    return {
        "stop": None,
        "buy_zone_low": None,
        "buy_zone_high": None,
        "horizon": HORIZON_BY_CATEGORY.get(category_id, ""),
        "opportunity_label": (
            "TRACKED" if action_badge in {"Tracked", "Open"}
            else "CLOSED" if action_badge in {"Closed", "Win", "Loss", "Void"}
            else "WATCH"
        ),
        "expected_payoff": "Unproven",
        "expected_payoff_detail": "Lifecycle row — expected payoff is not restated here.",
        "evidence": "Thin",
        "strategy_health": "Unmeasured",
        "strategy_health_detail": "",
        "market_support": "Unmeasured",
        "market_support_detail": "",
        "why_now": [],
        "key_points": [],
        "what_changes_mind": [],
        "next_step": "Open the deeper research for the live plan.",
        "evidence_panel": {
            "sample_size": None,
            "ev_pct": None,
            "ev_lb_pct": None,
            "p_win": None,
            "confidence": None,
            "score": None,
            "rsi": None,
            "volume_ratio": None,
            "signals": [],
            "price_tag": "",
            "tech_source": "",
            "fundamental_coverage": None,
            "provenance": "Long-term / signal-outcome tracker — not a fresh scan card.",
        },
    }


def _bucket_rows(
    scan_rows: Sequence[Mapping[str, Any]],
    lt_rows: Sequence[Mapping[str, Any]],
    market_ctx: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    wealth: list[dict[str, Any]] = []
    for r in lt_rows:
        ok, why, tags = _wealth_qualify(r)
        if not ok:
            continue
        wealth.append(card_from_row(
            r,
            category_id="wealth_builders",
            category_label="Wealth Builders",
            qualify_reason=why,
            evidence_tags=tags,
            market_ctx=market_ctx,
        ))
    wealth.sort(key=lambda c: (
        0 if "COMPOUNDER" in str(c.get("setup_label") or "") else
        1 if "GARP" in str(c.get("setup_label") or "") else 2,
        *method_sort_key(c),
    ))

    trends: list[dict[str, Any]] = []
    breakouts: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    labels = {m["id"]: m["label"] for m in CATEGORIES}

    for r in scan_rows:
        assigned = primary_scan_category(r)
        if not assigned:
            continue
        cat_id, why, tags = assigned
        card = card_from_row(
            r,
            category_id=cat_id,
            category_label=labels[cat_id],
            qualify_reason=why,
            evidence_tags=tags,
            market_ctx=market_ctx,
        )
        if cat_id == "momentum_breakouts":
            breakouts.append(card)
        elif cat_id == "recovery_setups":
            recovery.append(card)
        else:
            trends.append(card)

    trends.sort(key=method_sort_key)
    breakouts.sort(key=method_sort_key)
    recovery.sort(key=lambda c: (
        0 if "DOUBLE_BOTTOM" in (c.get("evidence_tags") or []) else 1,
        *method_sort_key(c),
    ))
    return {
        "wealth_builders": wealth[:_BUCKET_CAP],
        "super_trends": trends[:_BUCKET_CAP],
        "momentum_breakouts": breakouts[:_BUCKET_CAP],
        "recovery_setups": recovery[:_BUCKET_CAP],
    }


def _tracker_lifecycle() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    try:
        from core.long_term_tracker import active_picks, exited_picks
        for p in active_picks():
            price = _f(p.get("last_price") or p.get("entry_price"))
            entry = _f(p.get("entry_price"))
            card = {
                "symbol": str(p.get("symbol") or "").upper(),
                "company": str(p.get("symbol") or ""),
                "category_id": "wealth_builders",
                "category_label": "Wealth Builders",
                "action_badge": "Tracked",
                "risk_tier": "Medium",
                "risk_label": "Long-term tracked",
                "setup_label": "ACTIVE long-term pick",
                "sector": "—",
                "score": _f(p.get("score")),
                "lifecycle": "active",
                "source": "long_term_tracker",
                "cmp": price or None,
                "entry": entry or None,
                "target": None,
                "upside_from_entry_pct": (
                    round((price / entry - 1.0) * 100.0, 1) if entry > 0 and price > 0 else None
                ),
                "upside_to_target_pct": None,
                "reason": str(p.get("thesis") or "")[:160],
                "signals": [],
                **_empty_decision_fields("Tracked", "wealth_builders"),
            }
            active.append(_attach_key_points(card))
        for p in exited_picks(limit=40):
            entry = _f(p.get("entry_price"))
            exit_px = _f(p.get("exit_price") or p.get("last_price"))
            closed.append(_attach_key_points({
                "symbol": str(p.get("symbol") or "").upper(),
                "company": str(p.get("symbol") or ""),
                "category_id": "wealth_builders",
                "category_label": "Wealth Builders",
                "action_badge": "Closed",
                "risk_tier": "—",
                "risk_label": "Exited",
                "setup_label": "EXITED long-term pick",
                "sector": "—",
                "score": _f(p.get("score")),
                "lifecycle": "closed",
                "source": "long_term_tracker",
                "cmp": exit_px or None,
                "entry": entry or None,
                "target": None,
                "upside_from_entry_pct": _f(p.get("return_pct")) if p.get("return_pct") is not None else (
                    round((exit_px / entry - 1.0) * 100.0, 1) if entry > 0 and exit_px > 0 else None
                ),
                "upside_to_target_pct": None,
                "reason": str(p.get("exit_reason") or p.get("thesis") or "")[:160],
                **_empty_decision_fields("Closed", "wealth_builders"),
            }))
    except Exception:
        pass

    try:
        from core.signal_outcome_tracker import get_recent_signals
        for s in get_recent_signals(limit=80):
            worked = s.get("worked")
            entry = _f(s.get("entry_price"))
            target = _f(s.get("target_price"))
            outcome_px = _f(s.get("outcome_price"))
            base = {
                "symbol": str(s.get("symbol") or "").upper(),
                "company": str(s.get("symbol") or ""),
                "category_id": "momentum_breakouts",
                "category_label": "Momentum Breakouts",
                "risk_tier": "Medium",
                "risk_label": str(s.get("signal_type") or "signal"),
                "setup_label": str(s.get("signal_type") or "Signal"),
                "sector": "—",
                "score": 0.0,
                "source": "signal_outcome_tracker",
                "entry": entry or None,
                "target": target or None,
                "reason": f"Logged {s.get('logged_at') or ''}".strip(),
                "signals": [str(s.get("signal_type") or "").upper()] if s.get("signal_type") else [],
                **_empty_decision_fields("Open", "momentum_breakouts"),
            }
            if worked is None:
                active.append(_attach_key_points({
                    **base,
                    "action_badge": "Open",
                    "lifecycle": "active",
                    "cmp": entry or None,
                    "upside_from_entry_pct": None,
                    "upside_to_target_pct": (
                        round((target / entry - 1.0) * 100.0, 1) if entry > 0 and target > 0 else None
                    ),
                }))
            else:
                closed.append(_attach_key_points({
                    **base,
                    "action_badge": "Win" if worked == 1 else ("Void" if worked == -1 else "Loss"),
                    "lifecycle": "closed",
                    "cmp": outcome_px or None,
                    "upside_from_entry_pct": (
                        round(_f(s.get("outcome_pct")), 1) if s.get("outcome_pct") is not None else None
                    ),
                    "upside_to_target_pct": None,
                    **_empty_decision_fields(
                        "Win" if worked == 1 else ("Void" if worked == -1 else "Loss"),
                        "momentum_breakouts",
                    ),
                }))
    except Exception:
        pass

    active.sort(key=lambda c: (c.get("symbol") or ""))
    closed.sort(key=lambda c: (c.get("symbol") or ""))
    return active, closed


def build_recommendations_workspace(
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    refresh_technicals: bool = False,
    settle_cases: bool = False,
) -> dict[str, Any]:
    """Project recommendation categories + lifecycle from persisted product state.

    Page-open defaults skip live technical refresh and case settlement so the
    desk reads the last scan instead of recomputing hundreds of rows.
    """
    scan = dict(scan_payload or {})
    lt = dict(long_term_payload or {})
    scan_at = str(scan.get("scanned_at") or "")
    lt_at = str(lt.get("scanned_at") or "")

    scan_rows = [enrich_scan_row(dict(r), scanned_at=scan_at) for r in (scan.get("records") or [])]
    lt_rows = [enrich_long_term_row(dict(r), scanned_at=lt_at) for r in (lt.get("records") or [])]

    if refresh_technicals and (scan_rows or lt_rows):
        try:
            from product.live_technicals import refresh_rows_technicals
            # Latest CMP / RSI / entry / target for every row we will show.
            # Store-local after one live overlay — no per-symbol scrapes.
            scan_rows = refresh_rows_technicals(scan_rows, bulk_overlay=True)
            if lt_rows:
                lt_rows = refresh_rows_technicals(lt_rows, bulk_overlay=False)
        except Exception:
            pass

    try:
        from product.live_technicals import apply_current_trade_levels
        for row in (*scan_rows, *lt_rows):
            if _f(row.get("price") or row.get("cmp")) > 0:
                apply_current_trade_levels(row, None)
    except Exception:
        pass

    attach_live_ev(scan_rows)
    scan_rows, lt_rows = attach_research_overlays(
        scan_rows, lt_rows, scanned_at=scan_at,
    )
    if settle_cases:
        try:
            from product.case_memory import settle_due_cases
            settle_due_cases()
        except Exception:
            pass
    desk = build_desk_context(scan_rows)
    buckets = _bucket_rows(scan_rows, lt_rows, market_ctx=desk)
    active, closed = _tracker_lifecycle()

    categories = []
    assigned = 0
    for meta in CATEGORIES:
        cards = list(buckets.get(meta["id"]) or [])
        assigned += len(cards)
        categories.append({
            **meta,
            "count": len(cards),
            "cards": cards,
            "empty_detail": _empty_detail_for(
                meta["id"], cards=cards, lt_rows=lt_rows, scan_rows=scan_rows,
            ),
        })

    cmp_note = (
        "CMP is the latest official NSE bar (Kite overlay when the market is open; "
        "otherwise last EOD). Entry, stop and target are shown only when the scan "
        "already stored them — this desk does not invent 5%/10% plans. "
        "Each scan symbol maps to one primary category (breakout → recovery → trend). "
        "Buy requires two independent methods (tape, SEPA, funds, trend, RS, live EV, "
        "conviction, case memory, sector). SEPA alone is not a Buy. Missing methods stay empty."
    )
    if str(scan.get("records_status") or "") == "PRIOR_DAY_SNAPSHOT":
        cmp_note = "Scan file is a PRIOR-DAY SNAPSHOT — run a fresh market scan before acting. " + cmp_note

    return {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_scanned_at": scan_at,
        "long_term_scanned_at": lt_at,
        "records_status": str(scan.get("records_status") or ""),
        "same_ist_day": bool(scan.get("same_ist_day")),
        "scan_meta": {
            "market_scanned_at": scan_at,
            "market_row_count": len(scan_rows),
            "long_term_scanned_at": lt_at,
            "long_term_row_count": len(lt_rows),
            "assigned_count": assigned,
        },
        "cmp_note": cmp_note,
        "assignment_policy": (
            "exclusive_primary: momentum_breakouts > recovery_setups > super_trends; "
            "wealth_builders from long-term quality only; "
            "buy_requires_two_independent_methods"
        ),
        "methods_note": (
            "Nine research methods, all from saved system state: tape, SEPA overlay, "
            "long-term funds, trend structure, relative strength, live EV (≥30), "
            "conviction shortlist, case memory, sector leadership. Missing stays unknown. "
            "A Buy needs two passes — SEPA 100 with funds 0 is not enough."
        ),
        "desk": desk,
        "categories": categories,
        "lifecycle": {
            "active": active[:60],
            "closed": closed[:60],
            "active_count": len(active),
            "closed_count": len(closed),
        },
        "disclaimer": (
            "Research categories from QuantTerm evidence — not broker recommendations, "
            "not a promise of returns, not Reco Wealth. Expected payoff stays Unproven "
            "until ≥30 comparable outcomes exist."
        ),
    }


def _persist_pulse(pulse: Mapping[str, Any]) -> Path | None:
    day = str(pulse.get("as_of_ist") or "") or _ist_day()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"market_pulse_{day}.json"
    try:
        import json
        payload = {
            "id": f"market_pulse_{day}",
            "title": "Market Pulse",
            "kind": "market_pulse",
            "date": day,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pulse": dict(pulse),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path
    except Exception:
        return None


def _ist_day() -> str:
    try:
        from core.market_clock import today_ist
        return today_ist().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def _session_movers_from_scan(records: Sequence[Mapping[str, Any]], limit: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """1-day movers only. Missing change_pct stays empty — never substitute 5-day momentum."""
    rows: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").upper()
        price = _f(row.get("price"))
        chg = row.get("change_pct")
        if not symbol or price <= 0 or chg is None or chg == "":
            continue
        try:
            chg_f = float(chg)
        except (TypeError, ValueError):
            continue
        rows.append({"symbol": symbol, "price": round(price, 1), "chg_pct": round(chg_f, 2)})
    if not rows:
        return [], []
    rows.sort(key=lambda item: item["chg_pct"], reverse=True)
    return rows[:limit], list(reversed(rows[-limit:]))


def _scan_highlights(scan_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project the saved market scan onto Market Reports. Empty stays empty."""
    scan = dict(scan_payload or {})
    records = [r for r in (scan.get("records") or []) if isinstance(r, Mapping)]
    scanned_at = str(scan.get("scanned_at") or "")
    breakout_symbols: list[str] = []
    pre_symbols: list[str] = []
    ready = 0
    for row in records:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        sigs = {str(x).upper() for x in (row.get("signals") or [])}
        status = str(row.get("status") or "")
        grade = str(row.get("breakout_grade") or "").upper()
        if (sigs & {"BREAKOUT_52W", "BREAKOUT_RES"} or grade in {"A", "B"}) and symbol not in breakout_symbols:
            breakout_symbols.append(symbol)
        if (sigs & {"PRE_BREAKOUT"} or status == "Watch for breakout") and symbol not in pre_symbols:
            pre_symbols.append(symbol)
        if status == "Ready to trade":
            ready += 1
    gainers, losers = _session_movers_from_scan(records)
    empty_detail = ""
    if not records:
        empty_detail = (
            "No market scan on file. Scan market fills breakout context and session movers. "
            "This page does not invent headlines, index prints, or prices."
        )
    elif not breakout_symbols and not gainers:
        empty_detail = (
            f"Market scan has {len(records)} name(s) but no BREAKOUT tags and no 1-day "
            "change_pct for session movers. Refresh news separately for wrap headlines."
        )
    return {
        "scanned_at": scanned_at,
        "row_count": len(records),
        "same_ist_day": bool(scan.get("same_ist_day")),
        "ready_to_trade": ready,
        "breakout_symbols": breakout_symbols[:12],
        "pre_breakout_symbols": pre_symbols[:12],
        "session_gainers": gainers,
        "session_losers": losers,
        "empty_detail": empty_detail,
    }


def _enrich_pulse_from_scan(pulse: Mapping[str, Any], highlights: Mapping[str, Any]) -> dict[str, Any]:
    """Fill empty pulse slots from the saved scan. Never invent Nifty prints or news."""
    out = dict(pulse or {})
    if not (out.get("gainers") or []) and highlights.get("session_gainers"):
        out["gainers"] = list(highlights["session_gainers"])
    if not (out.get("losers") or []) and highlights.get("session_losers"):
        out["losers"] = list(highlights["session_losers"])
    existing_brk = out.get("breakouts_today") or []
    if not existing_brk and highlights.get("breakout_symbols"):
        out["breakouts_today"] = [{"symbol": s} for s in highlights["breakout_symbols"]]
    takes = [str(t).strip() for t in (out.get("takeaways") or []) if str(t).strip()]
    if not takes:
        extra: list[str] = []
        n = int(highlights.get("row_count") or 0)
        if n:
            extra.append(f"Last market scan has {n} name(s).")
        brk = list(highlights.get("breakout_symbols") or [])
        if brk:
            extra.append("Breakouts on last scan: " + ", ".join(brk[:6]))
        pre = list(highlights.get("pre_breakout_symbols") or [])
        if pre:
            extra.append("Near pivot: " + ", ".join(pre[:6]))
        if extra:
            out["takeaways"] = extra
    return out


def _compact_movers(rows: Any, limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(rows or [])[:limit]:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        item: dict[str, Any] = {"symbol": symbol}
        price = _f(row.get("price"))
        chg = row.get("chg_pct")
        if price > 0:
            item["price"] = price
        if chg is not None:
            item["chg_pct"] = _f(chg)
        out.append(item)
    return out


def _report_item(data: Mapping[str, Any], *, path: str = "", today: str = "") -> dict[str, Any]:
    pulse = dict(data.get("pulse") or {})
    day = str(data.get("date") or pulse.get("as_of_ist") or "")
    breakouts = pulse.get("breakouts_today") or []
    snapshot = dict(pulse.get("snapshot") or {})
    indices = []
    for idx in snapshot.get("indices") or []:
        if not isinstance(idx, Mapping):
            continue
        indices.append({
            "name": str(idx.get("name") or ""),
            "price": idx.get("price"),
            "chg_pct": idx.get("chg_pct"),
        })
    return {
        "id": str(data.get("id") or f"market_pulse_{day}"),
        "title": str(data.get("title") or "Market Pulse"),
        "kind": str(data.get("kind") or "market_pulse"),
        "date": day,
        "created_at": str(data.get("created_at") or ""),
        "is_new": bool(today and day == today),
        "badge": "Today" if today and day == today else "",
        "summary": _pulse_summary(pulse),
        "takeaways": [str(t) for t in (pulse.get("takeaways") or [])[:6]],
        "breakouts_today": [
            str(b.get("symbol") if isinstance(b, Mapping) else b)
            for b in breakouts if b
        ],
        "gainers": _compact_movers(pulse.get("gainers")),
        "losers": _compact_movers(pulse.get("losers")),
        "snapshot": {
            "indices": indices,
            "commentary": str(snapshot.get("commentary") or ""),
        },
        "as_of_ist": str(pulse.get("as_of_ist") or day),
        "path": path,
    }


def _list_saved_reports(limit: int = 30, *, today: str = "") -> list[dict[str, Any]]:
    if not REPORTS_DIR.exists():
        return []
    import json
    rows: list[dict[str, Any]] = []
    for path in sorted(REPORTS_DIR.glob("market_pulse_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append(_report_item(data, path=str(path), today=today))
        except Exception:
            continue
        if len(rows) >= limit:
            break
    rows.sort(key=lambda r: str(r.get("date") or ""), reverse=True)
    return rows


def _pulse_summary(pulse: Mapping[str, Any]) -> str:
    takes = [str(t) for t in (pulse.get("takeaways") or [])[:2]]
    if takes:
        return " · ".join(takes)
    return str(pulse.get("date") or "Market overview")


_PULSE_FILE_TTL_S = 15 * 60


def build_market_reports_workspace(
    *,
    persist_today: bool = True,
    news_payload: Mapping[str, Any] | None = None,
    scan_payload: Mapping[str, Any] | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Chronological Market Pulse list. Reuses today's file when it is fresh."""
    import json
    import time

    pulse: dict[str, Any] = {}
    error = ""
    today = _ist_day()
    path = REPORTS_DIR / f"market_pulse_{today}.json"
    if persist_today and path.exists() and not rebuild:
        try:
            age = time.time() - path.stat().st_mtime
            if 0 <= age < _PULSE_FILE_TTL_S:
                data = json.loads(path.read_text(encoding="utf-8"))
                pulse = dict(data.get("pulse") or {})
        except Exception:
            pulse = {}
    if not pulse:
        try:
            from reports.street_pulse import build_pulse
            pulse = build_pulse(force=rebuild) or {}
        except Exception as exec_pulse:
            error = str(exec_pulse)[:200]
        if pulse and persist_today:
            pulse.setdefault("as_of_ist", today)
            _persist_pulse(pulse)

    highlights = _scan_highlights(scan_payload)
    pulse = _enrich_pulse_from_scan(pulse, highlights)

    articles = list((news_payload or {}).get("articles") or [])
    news_meta = {
        "article_count": len(articles),
        "available": bool((news_payload or {}).get("available")),
    }

    desk_note: dict[str, Any] = {}
    try:
        from product.desk_note import build_desk_note
        desk_note = build_desk_note(articles=articles, scan_payload=scan_payload)
    except Exception as exec_note:
        desk_note = {"error": str(exec_note)[:200], "wrap": [], "desks": [], "explainers": []}

    reports = _list_saved_reports(today=today)
    has_today = any(r.get("date") == today for r in reports)
    pulse_has_rows = bool(
        (pulse.get("takeaways") or [])
        or (pulse.get("gainers") or [])
        or (pulse.get("breakouts_today") or [])
        or int(highlights.get("row_count") or 0)
    )
    if not has_today and pulse_has_rows:
        reports.insert(0, _report_item({
            "id": f"market_pulse_{today}",
            "title": "Market Pulse",
            "kind": "market_pulse",
            "date": today,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pulse": pulse,
        }, today=today))
    elif has_today:
        # Re-project today's list item from the scan-enriched pulse so breakouts
        # from the saved scan show even when the 15-minute file had empty slots.
        reports = [
            _report_item({
                "id": r.get("id") or f"market_pulse_{today}",
                "title": r.get("title") or "Market Pulse",
                "kind": r.get("kind") or "market_pulse",
                "date": today,
                "created_at": r.get("created_at") or "",
                "pulse": pulse,
            }, today=today)
            if r.get("date") == today else r
            for r in reports
        ]
    today_rows = [r for r in reports if r.get("is_new")]
    older = [r for r in reports if not r.get("is_new")]
    older.sort(key=lambda r: str(r.get("date") or ""), reverse=True)
    reports = today_rows + older

    as_of = str((pulse or {}).get("as_of_ist") or highlights.get("scanned_at") or today)
    sourced = int((desk_note or {}).get("wrap_sourced") or 0)
    empty_detail = ""
    if not sourced and not int(highlights.get("row_count") or 0) and not (pulse.get("takeaways") or []):
        empty_detail = (
            "No sourced wrap and no market scan on file. Scan market and refresh news "
            "to fill this archive. Opening the page does not start those jobs, and "
            "empty slots are not filled with invented headlines."
        )
    elif not sourced:
        empty_detail = (
            "No sourced wrap headlines yet — refresh news and filings. Breakout context "
            "below comes from the last market scan, not from invented copy."
        )
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_ist": as_of,
        "title": "Stay on top of the markets",
        "blurb": (
            "Daily Market Pulse plus a sourced desk note — wrap, teach-ins and company "
            "watch desks. Built from the last scan, official session files and saved news. "
            "Headlines stay sourced. Empty stays empty."
        ),
        "load_note": "Pulse reuses today's file for 15 minutes; it does not walk every bhavcopy symbol.",
        "reports": reports,
        "today_pulse": pulse,
        "desk_note": desk_note,
        "scan_highlights": highlights,
        "news_meta": news_meta,
        "empty_detail": empty_detail,
        "error": error,
        "disclaimer": "Market reports are research summaries, not trade instructions.",
    }
