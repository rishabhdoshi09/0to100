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

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "logs" / "product" / "market_reports"

# Reco-Wealth analogues — QuantTerm evidence names (not a brand clone).
CATEGORIES: tuple[dict[str, str], ...] = (
    {
        "id": "wealth_builders",
        "label": "Wealth Builders",
        "blurb": "Long-term quality / GARP — compounders with ≥50% fundamental coverage.",
        "icon": "compound",
    },
    {
        "id": "super_trends",
        "label": "Super Trends",
        "blurb": "Momentum without chase — trend + RS leadership, not extended.",
        "icon": "trend",
    },
    {
        "id": "momentum_breakouts",
        "label": "Momentum Breakouts",
        "blurb": "Sniper / graded breakouts with volume floor — near pivot or confirmed.",
        "icon": "breakout",
    },
    {
        "id": "recovery_setups",
        "label": "Recovery Setups",
        "blurb": "True turnarounds only — double-bottom / accumulation (not coils).",
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


def _action_badge(row: Mapping[str, Any], *, category_id: str = "") -> str:
    cls = str(row.get("classification") or "")
    if category_id == "wealth_builders":
        if cls == "QUALITY_BUT_EXPENSIVE":
            return "Research"
        if cls in {"QUALITY_COMPOUNDER", "GARP_CANDIDATE"}:
            return "Hold / Research"
    verdict = str(row.get("verdict") or "").upper()
    status = str(row.get("status") or "")
    if verdict in {"BUY", "STRONG BUY"} or status == "Ready to trade":
        return "Buy"
    if "breakout" in status.lower() or str(row.get("breakout_grade") or "").upper() in {"A", "B"}:
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
    ups = upside_metrics(row)
    tags = [str(t) for t in (evidence_tags or []) if t]
    reason = qualify_reason or str(row.get("reason") or "")
    badge = _action_badge(row, category_id=category_id)
    surface = decision_surface(
        row,
        category_id=category_id,
        action_badge=badge,
        qualify_reason=reason,
        evidence_tags=tags,
        market_ctx=market_ctx,
    )
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "company": str(row.get("company") or row.get("symbol") or ""),
        "category_id": category_id,
        "category_label": category_label,
        "action_badge": badge,
        "risk_tier": risk_tier(row),
        "risk_label": str(row.get("risk_label") or risk_tier(row)),
        "setup_label": str(row.get("setup_label") or row.get("status") or row.get("classification") or ""),
        "sector": str(row.get("sector") or "—"),
        "score": _f(row.get("score") or row.get("combined_score")),
        "rsi": _f(row.get("rsi")) or None,
        "volume_ratio": _f(row.get("volume_ratio")) or None,
        "price_tag": str(row.get("price_tag") or ""),
        "tech_source": str(row.get("tech_source") or ""),
        "reason": reason,
        "qualify_reason": reason,
        "evidence_tags": tags,
        "lifecycle": "active",
        **ups,
        **surface,
    }


def _trend_structure_ok(row: Mapping[str, Any]) -> bool:
    """Pass when above a key SMA, or SMA unknown (don't invent a fail)."""
    above50 = row.get("above_sma50")
    above200 = row.get("above_sma200")
    if above50 is False and above200 is False:
        return False
    return True


def _is_momentum_breakout(row: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """Sniper / graded breakouts — volume floor, no chase on the research desk."""
    if bool(row.get("chase_risk")):
        return False, "", []
    rsi = _f(row.get("rsi"))
    if rsi > RSI_HARD:
        return False, "", []
    if not passes_volume_floor(row):
        return False, "", []
    grade = str(row.get("breakout_grade") or "").upper()
    sniper = is_sniper_breakout_candidate(row)
    if not sniper and grade not in {"A", "B"}:
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
    if display_state and display_state != "not_in_breakout_lane":
        tags.append(display_state)
    why = (
        f"Sniper breakout · vol {_f(row.get('volume_ratio')):.1f}×"
        + (f" · grade {grade}" if grade else "")
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
            "No sniper/graded breakouts with volume floor and without chase risk "
            f"in this scan ({len(scan_rows)} rows)."
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
        -_f(c.get("score")),
        c["symbol"],
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

    trends.sort(key=lambda c: (-_f(c.get("score")), c["symbol"]))
    breakouts.sort(key=lambda c: (
        -_f(c.get("score")),
        c["symbol"],
    ))
    # Prefer higher breakout_quality when present on source rows — score already
    # embeds sniper boost after enrich; keep symbol tie-break stable.
    recovery.sort(key=lambda c: (
        0 if "DOUBLE_BOTTOM" in (c.get("evidence_tags") or []) else 1,
        -_f(c.get("score")),
        c["symbol"],
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
                **_empty_decision_fields("Tracked", "wealth_builders"),
            }
            active.append(card)
        for p in exited_picks(limit=40):
            entry = _f(p.get("entry_price"))
            exit_px = _f(p.get("exit_price") or p.get("last_price"))
            closed.append({
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
            })
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
                **_empty_decision_fields("Open", "momentum_breakouts"),
            }
            if worked is None:
                active.append({
                    **base,
                    "action_badge": "Open",
                    "lifecycle": "active",
                    "cmp": entry or None,
                    "upside_from_entry_pct": None,
                    "upside_to_target_pct": (
                        round((target / entry - 1.0) * 100.0, 1) if entry > 0 and target > 0 else None
                    ),
                })
            else:
                closed.append({
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
                })
    except Exception:
        pass

    active.sort(key=lambda c: (c.get("symbol") or ""))
    closed.sort(key=lambda c: (c.get("symbol") or ""))
    return active, closed


def build_recommendations_workspace(
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    refresh_technicals: bool = True,
) -> dict[str, Any]:
    """Project recommendation categories + lifecycle from persisted product state."""
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
    desk = build_desk_context(scan_rows)
    buckets = _bucket_rows(scan_rows, lt_rows, market_ctx=desk)
    active, closed = _tracker_lifecycle()

    categories = []
    for meta in CATEGORIES:
        cards = list(buckets.get(meta["id"]) or [])
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
        "Each scan symbol maps to one primary category (breakout → recovery → trend)."
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
        "cmp_note": cmp_note,
        "assignment_policy": (
            "exclusive_primary: momentum_breakouts > recovery_setups > super_trends; "
            "wealth_builders from long-term quality only"
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


def build_market_reports_workspace(
    *,
    persist_today: bool = True,
    news_payload: Mapping[str, Any] | None = None,
    scan_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Chronological Market Pulse list from live street pulse + saved day files."""
    pulse: dict[str, Any] = {}
    error = ""
    try:
        from reports.street_pulse import build_pulse
        pulse = build_pulse() or {}
    except Exception as exec_pulse:
        error = str(exec_pulse)[:200]

    desk_note: dict[str, Any] = {}
    try:
        from product.desk_note import build_desk_note
        articles = list((news_payload or {}).get("articles") or [])
        desk_note = build_desk_note(articles=articles, scan_payload=scan_payload)
    except Exception as exec_note:
        desk_note = {"error": str(exec_note)[:200], "wrap": [], "desks": [], "explainers": []}

    today = _ist_day()
    if pulse and persist_today:
        pulse.setdefault("as_of_ist", today)
        _persist_pulse(pulse)

    reports = _list_saved_reports(today=today)
    if pulse and not any(r.get("date") == today for r in reports):
        reports.insert(0, _report_item({
            "id": f"market_pulse_{today}",
            "title": "Market Pulse",
            "kind": "market_pulse",
            "date": today,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pulse": pulse,
        }, today=today))
    today_rows = [r for r in reports if r.get("is_new")]
    older = [r for r in reports if not r.get("is_new")]
    older.sort(key=lambda r: str(r.get("date") or ""), reverse=True)
    reports = today_rows + older

    as_of = str((pulse or {}).get("as_of_ist") or today)
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_ist": as_of,
        "title": "Stay on top of the markets",
        "blurb": (
            "Daily Market Pulse plus a sourced desk note — wrap, teach-ins and company "
            "watch desks. Headlines stay sourced. Empty stays empty."
        ),
        "reports": reports,
        "today_pulse": pulse,
        "desk_note": desk_note,
        "error": error,
        "disclaimer": "Market reports are research summaries, not trade instructions.",
    }
