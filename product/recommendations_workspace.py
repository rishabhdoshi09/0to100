"""Reco-style recommendation desk — projections only, no new scans or scrapes.

Maps QuantTerm evidence into named research categories (Wealth Builders,
Super Trends, Momentum Breakouts, Recovery Setups) plus Active/Closed
lifecycle strips from long-term + signal-outcome trackers.

Honest rules:
  - Empty category → empty list (never fabricate picks)
  - Prices/upside from real entry/target/price fields only
  - CMP freshness tags come from a bulk quote stamp on the shortlist
  - Each scan symbol gets ONE primary category (no multi-bucket spam)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.breakout_quality import RSI_HARD, passes_volume_floor
from product.radar_workspace import (
    enrich_long_term_row,
    enrich_scan_row,
    is_long_term_pick,
    is_sniper_breakout_candidate,
    merge_fundamental_context,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "logs" / "product" / "market_reports"

# Reco-Wealth analogues — QuantTerm evidence names (not a brand clone).
CATEGORIES: tuple[dict[str, str], ...] = (
    {
        "id": "best_setups",
        "label": "Best Setups",
        "blurb": (
            "Top Stocks — technical SEPA on official NSE history, stage and RS vs "
            "Nifty 50 from the same tape, then on-file valuation metrics (calculated "
            "in the long-term pack, not live-scraped). A high score is a research qualify, not a buy."
        ),
        "icon": "setup",
    },
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
) -> dict[str, Any]:
    from product.research_levels import attach_research_levels, levels_tag
    row = attach_research_levels(row)
    ups = upside_metrics(row)
    tags = [str(t) for t in (evidence_tags or []) if t]
    tag = levels_tag(str(row.get("levels_source") or ""))
    if tag and tag not in tags:
        tags.append(tag)
    reason = qualify_reason or str(row.get("reason") or "")
    ev_n = int(row.get("ev_n") or 0)
    ev_pct = row.get("ev_pct")
    ev = {}
    if ev_n >= 30 and ev_pct is not None:
        ev = {
            "ev_pct": _f(ev_pct),
            "ev_lb_pct": _f(row.get("ev_lb_pct")) if row.get("ev_lb_pct") is not None else None,
            "ev_n": ev_n,
            "ev_conf": str(row.get("ev_conf") or ""),
            "p_win": _f(row.get("p_win")) if row.get("p_win") is not None else None,
        }
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "company": str(row.get("company") or row.get("symbol") or ""),
        "category_id": category_id,
        "category_label": category_label,
        "action_badge": _action_badge(row, category_id=category_id),
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
        "stop": _f(row.get("stop")) or None,
        "levels_source": str(row.get("levels_source") or ""),
        "upside_from_buy_pct": row.get("upside_from_buy_pct"),
        **ups,
        **ev,
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
    sepa_note: str = "",
) -> str:
    if cards:
        return ""
    if category_id == "best_setups":
        return sepa_note or (
            "Best Setups needs a saved market scan plus official OHLCV. "
            "Run a scan, then open a stock to see the 7-rule breakdown even when this list is empty."
        )
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


def _bucket_rows(
    scan_rows: Sequence[Mapping[str, Any]],
    lt_rows: Sequence[Mapping[str, Any]],
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
        "best_setups": [],
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
                })
    except Exception:
        pass

    active.sort(key=lambda c: (c.get("symbol") or ""))
    closed.sort(key=lambda c: (c.get("symbol") or ""))
    return active, closed


def _stamp_live_cmp(categories: Sequence[Mapping[str, Any]]) -> None:
    """One bulk quote pass on the shortlist — never rebuild RSI for 120 scan rows.

    Ideas used to freeze on `refresh_rows_technicals(scan[:120])`. Category
    membership already comes from the saved scan; the desk only needs a fresh
    last print on the cards it will show.
    """
    cards: list[dict[str, Any]] = []
    for cat in categories:
        if isinstance(cat, dict):
            cards.extend(c for c in (cat.get("cards") or []) if isinstance(c, dict))
    symbols: list[str] = []
    seen: set[str] = set()
    for card in cards:
        sym = str(card.get("symbol") or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
    if not symbols:
        return
    try:
        from data.live_quotes import get_live_quotes
        quotes = get_live_quotes(symbols[:80], ttl=8.0, allow_google=False) or {}
    except Exception:
        return
    for card in cards:
        sym = str(card.get("symbol") or "").strip().upper()
        raw = quotes.get(sym) or {}
        try:
            px = float((raw.get("price") if isinstance(raw, Mapping) else raw) or 0.0)
        except (TypeError, ValueError):
            px = 0.0
        if px <= 0:
            continue
        card["cmp"] = round(px, 2)
        src = str(raw.get("source") or "") if isinstance(raw, Mapping) else ""
        card["price_tag"] = "LIVE" if src in {"kite", "nse"} else (src.upper() or "EOD")
        card["tech_source"] = src or "quote"
        chg = raw.get("chg_pct") if isinstance(raw, Mapping) else None
        try:
            if chg is not None:
                card["change_pct"] = round(float(chg), 2)
        except (TypeError, ValueError):
            pass
        entry = _f(card.get("entry"))
        target = _f(card.get("target"))
        if entry > 0:
            card["upside_from_entry_pct"] = round((px / entry - 1.0) * 100.0, 1)
        if target > 0 and px > 0:
            card["upside_to_target_pct"] = round((target / px - 1.0) * 100.0, 1)
        buy = entry if entry > 0 else px
        if target > 0 and buy > 0:
            card["upside_from_buy_pct"] = round((target / buy - 1.0) * 100.0, 1)


def _best_setup_cards(
    scan_rows: Sequence[Mapping[str, Any]],
    *,
    load_frame: Any = None,
    compute: bool = True,
    cache_key: str = "",
) -> tuple[list[dict[str, Any]], str]:
    if not compute:
        return [], "SEPA ranking skipped in this call."
    try:
        from product.sepa_setup import rank_best_setups, sepa_card_fields
        ranked, note = rank_best_setups(
            scan_rows, load_frame=load_frame, cache_key=cache_key,
        )
    except Exception as exc:
        return [], f"SEPA ranking unavailable: {str(exc)[:160]}"
    cards: list[dict[str, Any]] = []
    for sepa, row in ranked:
        extra = sepa_card_fields(sepa)
        why = extra.get("sepa_headline") or "SEPA template"
        passed = extra.get("sepa_passed")
        total = extra.get("sepa_total")
        score = extra.get("sepa_score")
        if passed is not None and total and score is not None:
            why = f"{why} · {passed}/{total} rules · {score}/100"
        card = card_from_row(
            row,
            category_id="best_setups",
            category_label="Best Setups",
            qualify_reason=why,
            evidence_tags=[
                tag for tag in (
                    extra.get("sepa_verdict") or "",
                    extra.get("stage_label") or "",
                    extra.get("rs_label") or "",
                    f"{passed}/{total}" if passed is not None else "",
                ) if tag
            ],
        )
        card.update({k: v for k, v in extra.items() if k != "setup_label"})
        if extra.get("setup_label"):
            card["setup_label"] = extra["setup_label"]
        card["action_badge"] = extra.get("sepa_verdict") or card.get("action_badge") or "Watch"
        from product.top_stocks import attach_to_card
        attach_to_card(card, row, sepa)
        cards.append(card)
    return cards, note


def build_recommendations_workspace(
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    refresh_technicals: bool = True,
    compute_sepa: bool = True,
    sepa_load_frame: Any = None,
) -> dict[str, Any]:
    """Project recommendation categories + lifecycle from persisted product state.

    ``refresh_technicals`` stamps live CMP on the shortlist with one quote
    call. It does not recompute RSI across the scan head.
    ``compute_sepa`` ranks Best Setups from official OHLCV (capped shortlist).
    """
    scan = dict(scan_payload or {})
    lt = dict(long_term_payload or {})
    scan_at = str(scan.get("scanned_at") or "")
    lt_at = str(lt.get("scanned_at") or "")

    scan_rows = [enrich_scan_row(dict(r), scanned_at=scan_at) for r in (scan.get("records") or [])]
    lt_rows = [enrich_long_term_row(dict(r), scanned_at=lt_at) for r in (lt.get("records") or [])]
    fund_by_symbol = {
        str(r.get("symbol") or "").upper(): r
        for r in lt_rows
        if str(r.get("symbol") or "")
    }
    scan_rows = [merge_fundamental_context(r, fund_by_symbol) for r in scan_rows]

    buckets = _bucket_rows(scan_rows, lt_rows)
    best_cards, sepa_note = _best_setup_cards(
        scan_rows, load_frame=sepa_load_frame, compute=compute_sepa,
        cache_key=scan_at,
    )
    buckets["best_setups"] = best_cards
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
                sepa_note=sepa_note if meta["id"] == "best_setups" else "",
            ),
        })

    if refresh_technicals:
        _stamp_live_cmp(categories)

    cmp_note = (
        "Top Stocks: SEPA + stage + 63-session RS vs Nifty 50 on official NSE history; "
        "last print from Kite or NSE (Google is not used here). Index strip is the NSE "
        "index store. Valuation metrics are calculated from the on-file long-term pack — "
        "missing ratios stay missing, no live scrape."
    )
    if str(scan.get("records_status") or "") == "PRIOR_DAY_SNAPSHOT":
        cmp_note = "Scan file is a PRIOR-DAY SNAPSHOT — run a fresh market scan before acting. " + cmp_note

    from product.sepa_setup import _session_label
    from product.top_stocks import tape_policy
    from product.monitor_context import INDEX_STRIP_NOTE, index_strip

    return {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "load_note": (
            "Top Stocks scores a capped scan shortlist on Minervini's 7-rule template "
            "from official OHLCV, then attaches stage, RS vs Nifty 50, and on-file "
            "fundamental ratios. Live CMP is Kite/NSE only."
        ),
        "typical_seconds": 8,
        "scan_scanned_at": scan_at,
        "long_term_scanned_at": lt_at,
        "records_status": str(scan.get("records_status") or ""),
        "same_ist_day": bool(scan.get("same_ist_day")),
        "cmp_note": cmp_note,
        "assignment_policy": (
            "best_setups: SEPA ≥40/100 on a scan shortlist; "
            "exclusive_primary: momentum_breakouts > recovery_setups > super_trends; "
            "wealth_builders from long-term quality only"
        ),
        "sepa_note": sepa_note,
        "tape": tape_policy(),
        "session": _session_label(),
        "indices": index_strip(),
        "index_strip_note": INDEX_STRIP_NOTE,
        "categories": categories,
        "lifecycle": {
            "active": active[:60],
            "closed": closed[:60],
            "active_count": len(active),
            "closed_count": len(closed),
        },
        "disclaimer": (
            "Research categories from QuantTerm evidence — not broker recommendations, "
            "not a promise of returns. SEPA is a published trend template, not Reco Wealth."
        ),
    }


def _persist_pulse(pulse: Mapping[str, Any]) -> Path | None:
    try:
        from core.market_clock import today_ist
        day = today_ist().isoformat()
    except Exception:
        day = datetime.now(timezone.utc).date().isoformat()
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


def _list_saved_reports(limit: int = 30) -> list[dict[str, Any]]:
    if not REPORTS_DIR.exists():
        return []
    import json
    rows: list[dict[str, Any]] = []
    for path in sorted(REPORTS_DIR.glob("market_pulse_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "id": str(data.get("id") or path.stem),
                "title": str(data.get("title") or "Market Pulse"),
                "kind": str(data.get("kind") or "market_pulse"),
                "date": str(data.get("date") or ""),
                "created_at": str(data.get("created_at") or ""),
                "is_new": False,
                "summary": _pulse_summary(data.get("pulse") or {}),
                "path": str(path),
            })
        except Exception:
            continue
        if len(rows) >= limit:
            break
    return rows


def _pulse_summary(pulse: Mapping[str, Any]) -> str:
    takes = [str(t) for t in (pulse.get("takeaways") or [])[:2]]
    if takes:
        return " · ".join(takes)
    return str(pulse.get("date") or "Market overview")


def build_market_reports_workspace(*, persist_today: bool = True, rebuild: bool = False) -> dict[str, Any]:
    """Chronological Market Pulse list. Reuses today's file when it is fresh."""
    import json
    import time
    pulse: dict[str, Any] = {}
    error = ""
    try:
        from core.market_clock import today_ist
        day = today_ist().isoformat()
    except Exception:
        day = datetime.now(timezone.utc).date().isoformat()
    path = REPORTS_DIR / f"market_pulse_{day}.json"
    if persist_today and path.exists() and not rebuild:
        try:
            age = time.time() - path.stat().st_mtime
            if age < 900:
                data = json.loads(path.read_text(encoding="utf-8"))
                pulse = dict(data.get("pulse") or {})
        except Exception:
            pulse = {}
    if not pulse:
        try:
            from reports.street_pulse import build_pulse
            pulse = build_pulse() or {}
        except Exception as exc:
            error = str(exc)[:200]
        if pulse and persist_today:
            _persist_pulse(pulse)

    reports = _list_saved_reports()
    if reports:
        reports[0]["is_new"] = True
        reports[0]["badge"] = "New market report"
    elif pulse:
        try:
            from core.market_clock import today_ist
            day = today_ist().isoformat()
        except Exception:
            day = datetime.now(timezone.utc).date().isoformat()
        reports = [{
            "id": f"market_pulse_{day}",
            "title": "Market Pulse",
            "kind": "market_pulse",
            "date": day,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_new": True,
            "badge": "New market report",
            "summary": _pulse_summary(pulse),
            "path": "",
        }]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": "Stay on top of the markets",
        "blurb": (
            "Daily Market Pulse from the last scan — movers, breakouts, headlines. "
            "Reused for 15 minutes so opening Pulse is a file read, not a market rebuild."
        ),
        "typical_seconds": 8,
        "load_note": "Pulse uses the last saved scan; it does not walk every bhavcopy symbol.",
        "reports": reports,
        "today_pulse": pulse,
        "error": error,
        "disclaimer": "Market reports are research summaries, not trade instructions.",
    }
