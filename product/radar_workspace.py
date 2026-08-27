"""Market radar projections: Breakouts, Momentum and Long-Term Picks lanes.

Pure functions over persisted scan/long-term payloads — no duplicate scanning.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from product.breakout_quality import (
    MIN_VOLUME_RATIO,
    RSI_BLOWOFF,
    RSI_HARD,
    attach_best_pick_meta,
    gate_breakout_quality,
    passes_volume_floor,
    volume_ratio as _volume_ratio_shared,
)

BREAKOUT_TAGS = frozenset({"BREAKOUT_52W", "BREAKOUT_RES", "GOLDEN_CROSS", "VOL_SQUEEZE"})
QUALITY_CLASSES = frozenset({"QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE"})
MIN_LT_FUNDAMENTAL_COVERAGE = 0.50
SEPA_OVERLAY_MIN = 40
SEPA_MERGE_KEYS = (
    "sepa_score",
    "sepa_max",
    "sepa_passed",
    "sepa_total",
    "sepa_verdict",
    "sepa_headline",
    "sepa_advice",
)
RANKING_LEGEND = {
    "best_setups": (
        "SEPA 7-rule overlay (Minervini Stage 2) on the saved scan. Need ≥40/100. "
        "Research rank only — not a buy, not 8/8 eligibility."
    ),
    "best_technical_breakout": (
        "Tape rank among sniper-confirm names: volume floor, not chasing, RSI ≤82. "
        "SEPA is not used here."
    ),
    "best_among_breakouts": (
        "Best among the best: snipers with a second screen (SEPA overlay ≥40 and/or "
        "usable long-term funds). Sort: more independent confirms first, then "
        "0.45·SEPA + 0.30·funds + 0.25·tape. Missing inputs score 0 — never invented. "
        "Not a buy."
    ),
}
BEST_OF_BEST_WEIGHTS = {"sepa": 0.45, "funds": 0.30, "tape": 0.25}
SCAN_SHARED_NOTE = (
    "One whole-market scan fills Home, Scanner (all tabs), Recommendations "
    "and long-term from the same saved scan. A Scan Now click jumps the queue "
    "and walks every category in one pass. Names can appear in more than one lane."
)
# Re-export shared floors so callers keep importing from this module.
# Volume < 1× is a HARD reject for sniper/best — never just a sort demote.


def is_long_term_pick(row: Mapping[str, Any]) -> bool:
    """Actionable long-term idea: quality class with enough fundamental evidence."""
    if str(row.get("classification", "")) not in QUALITY_CLASSES:
        return False
    return _f(row.get("fundamental_coverage")) >= MIN_LT_FUNDAMENTAL_COVERAGE


def _signals(row: Mapping[str, Any]) -> set[str]:
    return {str(item).upper() for item in (row.get("signals") or [])}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _volume_ratio(row: Mapping[str, Any]) -> float:
    return _volume_ratio_shared(row)


def is_sniper_breakout_candidate(row: Mapping[str, Any]) -> bool:
    """True when this row is in the sniper's breakout universe.

    Technical gates only (volume ≥1×, not chase, RSI ≤82). Fundamentals are
    NOT required here — they feed the separate best-among section.
    """
    ok, _, _ = gate_breakout_quality(row, for_best=False)
    if not ok:
        return False
    cats = {str(c) for c in (row.get("categories") or [])}
    sigs = _signals(row)
    status = str(row.get("status", "") or "")
    is_pre = (
        "PreBreakout" in cats
        or "PRE_BREAKOUT" in sigs
        or status == "Watch for breakout"
    )
    dist = row.get("pivot_distance_pct")
    try:
        dist_f = float(dist) if dist is not None else (0.0 if is_pre else 99.0)
    except (TypeError, ValueError):
        dist_f = 99.0
    near_pivot = is_pre and 0.0 <= dist_f <= 2.5
    grade = str(row.get("breakout_grade") or "").upper()
    confirmed = grade in {"A", "B"} or (
        bool(sigs & BREAKOUT_TAGS)
        and str(row.get("verdict", "")).upper() in {"BUY", "STRONG BUY"}
        and status == "Ready to trade"
    )
    if not (near_pivot or confirmed):
        return False
    if _f(row.get("avg_vol20")) <= 0 and _volume_ratio(row) <= 0:
        return False
    return True


def breakout_quality_score(row: Mapping[str, Any], *, for_best: bool = False) -> float:
    """Rank breakouts. Technical path ignores missing funds; best-among boosts them."""
    ok, _, gates = gate_breakout_quality(row, for_best=for_best)
    if not ok:
        return -1000.0

    grade = str(row.get("breakout_grade") or "").upper()
    grade_pts = {"A": 30.0, "B": 15.0}.get(grade, 0.0)
    conv = _f(row.get("breakout_conviction"))
    score = _f(row.get("score"))
    edge = _f(row.get("edge_r"))
    fund_pts = 0.0
    if for_best:
        from product.breakout_quality import has_usable_fundamentals
        if has_usable_fundamentals(row):
            fund_pts = min(20.0, _f(row.get("fundamental_score")) * 0.20)
            cls = str(row.get("classification") or "")
            if cls in {"QUALITY_COMPOUNDER", "GARP_CANDIDATE"}:
                fund_pts += 5.0
            elif cls == "QUALITY_BUT_EXPENSIVE":
                fund_pts += 2.0
        elif gates.get("fundamentals") == "unknown":
            fund_pts = -5.0

    vol = _volume_ratio(row)
    vol_pts = 12.0 if vol >= 2.0 else 6.0
    rsi = _f(row.get("rsi"))
    rsi_pen = 12.0 if rsi >= 65 else 0.0
    trend_pts = 6.0 if gates.get("trend") == "pass" else 0.0
    sniper_boost = 20.0 if is_sniper_breakout_candidate(row) else 0.0

    return round(
        grade_pts + conv * 0.35 + score * 0.25 + edge * 40.0
        + fund_pts + vol_pts + sniper_boost + trend_pts - rsi_pen,
        2,
    )


def pick_best_sniper_breakout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Best technical sniper candidate (no fundamentals required)."""
    pool = [
        dict(r) for r in rows
        if is_sniper_breakout_candidate(r) and passes_volume_floor(r)
    ]
    if not pool:
        return None
    for r in pool:
        r["breakout_quality"] = breakout_quality_score(r, for_best=False)
    pool.sort(key=lambda r: (
        -_f(r.get("breakout_quality")),
        -_f(r.get("breakout_conviction")),
        -_f(r.get("score")),
        str(r.get("symbol", "")),
    ))
    return attach_best_pick_meta(pool[0], with_context=False, for_best=False)


def _sepa_score(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("sepa_score") or 0)
    except (TypeError, ValueError):
        return 0


def has_sepa_overlay(row: Mapping[str, Any]) -> bool:
    """True when the persisted Minervini overlay already scored this name ≥40/100."""
    return _sepa_score(row) >= SEPA_OVERLAY_MIN


def has_second_screen(row: Mapping[str, Any]) -> bool:
    """Independent confirm on top of sniper tape: SEPA overlay and/or usable funds."""
    from product.breakout_quality import has_usable_fundamentals

    return has_usable_fundamentals(row) or has_sepa_overlay(row)


def second_screen_count(row: Mapping[str, Any]) -> int:
    from product.breakout_quality import has_usable_fundamentals

    return int(has_usable_fundamentals(row)) + int(has_sepa_overlay(row))


def best_among_why(row: Mapping[str, Any]) -> str:
    from product.breakout_quality import has_usable_fundamentals

    parts: list[str] = []
    if has_sepa_overlay(row):
        parts.append(f"SEPA {_sepa_score(row)}/100")
    if has_usable_fundamentals(row):
        cls = str(row.get("classification") or "").replace("_", " ") or "fundamentals"
        fund = row.get("fundamental_score")
        if fund is not None:
            parts.append(f"{cls} · fund {round(_f(fund))}")
        else:
            parts.append(cls)
    return " · ".join(parts) or "second screen"


def desk_action_badge(row: Mapping[str, Any]) -> str:
    verdict = str(row.get("verdict") or "").upper()
    status = str(row.get("status") or "")
    if verdict in {"BUY", "STRONG BUY"} or status == "Ready to trade":
        return "Buy"
    return "Watch"


def desk_risk_tier(row: Mapping[str, Any]) -> str:
    """Low / Medium / High from flags already on the row. Never invented."""
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
    if row.get("risk_flags"):
        return "Medium"
    return "Low"


def desk_upside_metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    """% from entry and room to target. None when price/entry/target are missing."""
    price = _f(row.get("price"))
    entry = _f(row.get("entry") or row.get("entry_price"))
    target = _f(row.get("target") or row.get("target_price"))
    stop = _f(row.get("stop") or row.get("stop_price"))
    from_entry = round((price / entry - 1.0) * 100.0, 1) if entry > 0 and price > 0 else None
    to_target = round((target / price - 1.0) * 100.0, 1) if target > 0 and price > 0 else None
    return {
        "upside_from_entry_pct": from_entry,
        "upside_to_target_pct": to_target,
        "entry": entry or None,
        "target": target or None,
        "stop": stop or None,
        "cmp": price or None,
    }


def best_of_best_parts(row: Mapping[str, Any]) -> dict[str, Any]:
    """Transparent 0–100 composite. Missing SEPA/funds contribute 0, never a fill-in."""
    from product.breakout_quality import has_usable_fundamentals

    sepa = float(_sepa_score(row)) if has_sepa_overlay(row) else 0.0
    fund = min(100.0, _f(row.get("fundamental_score"))) if has_usable_fundamentals(row) else 0.0
    tape_raw = breakout_quality_score(row, for_best=True)
    tape = max(0.0, min(100.0, tape_raw / 1.5)) if tape_raw > 0 else 0.0
    w = BEST_OF_BEST_WEIGHTS
    composite = round(w["sepa"] * sepa + w["funds"] * fund + w["tape"] * tape, 2)
    return {
        "sepa": round(sepa, 2),
        "funds": round(fund, 2),
        "tape": round(tape, 2),
        "composite": composite,
        "weights": dict(w),
    }


def decorate_best_of_best(row: Mapping[str, Any], rank: int) -> dict[str, Any]:
    from product.breakout_quality import has_usable_fundamentals

    out = attach_best_pick_meta(dict(row), with_context=False, for_best=True)
    parts = dict(out.get("best_of_best_parts") or best_of_best_parts(out))
    out.update(desk_upside_metrics(out))
    out["rank"] = int(rank)
    out["best_of_best_score"] = parts.get("composite")
    out["best_of_best_parts"] = parts
    out["action_badge"] = desk_action_badge(out)
    out["risk_tier"] = desk_risk_tier(out)
    out["best_among_kind"] = "second_screen"
    out["best_among_why"] = best_among_why(out)
    out["sepa_used"] = has_sepa_overlay(out)
    out["funds_used"] = has_usable_fundamentals(out)
    if out.get("change_5d_pct") is None and out.get("momentum_5d") is not None:
        out["change_5d_pct"] = _f(out.get("momentum_5d"))
    return out


def rank_best_of_best(rows: Sequence[Mapping[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    """Rank snipers that already cleared a second screen. Empty when none qualify."""
    pool: list[dict[str, Any]] = []
    for r in rows:
        if not is_sniper_breakout_candidate(r) or not passes_volume_floor(r):
            continue
        ok, _, _ = gate_breakout_quality(r, for_best=True)
        if not ok:
            continue
        if not has_second_screen(r):
            continue
        row = dict(r)
        row["breakout_quality"] = breakout_quality_score(row, for_best=True)
        row["second_screens"] = second_screen_count(row)
        parts = best_of_best_parts(row)
        row["best_of_best_score"] = parts["composite"]
        row["best_of_best_parts"] = parts
        pool.append(row)
    pool.sort(key=lambda r: (
        -int(r.get("second_screens") or 0),
        -_f(r.get("best_of_best_score")),
        -_sepa_score(r),
        -_f(r.get("fundamental_score")),
        -_f(r.get("breakout_quality")),
        str(r.get("symbol") or ""),
    ))
    return [decorate_best_of_best(row, i) for i, row in enumerate(pool[: max(0, int(limit))], start=1)]


def pick_best_among_fundamentals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """#1 of rank_best_of_best. None when no sniper has a second screen."""
    ranked = rank_best_of_best(rows, limit=1)
    return ranked[0] if ranked else None


def merge_fundamental_context(
    row: Mapping[str, Any],
    fund_by_symbol: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Attach long-term fundamental fields onto a scan row when available."""
    out = dict(row)
    if not fund_by_symbol:
        return out
    fund = fund_by_symbol.get(str(out.get("symbol", "") or "").upper())
    if not fund:
        return out
    for key in (
        "fundamental_score",
        "fundamental_coverage",
        "combined_score",
        "classification",
        "quality_factors",
        "risk_flags",
        "fundamentals",
        "sales_growth_3y",
        "profit_growth_3y",
    ):
        if key in fund and fund.get(key) is not None and out.get(key) is None:
            out[key] = fund.get(key)
    nested = fund.get("fundamentals")
    if isinstance(nested, Mapping) and out.get("fundamentals") is None:
        out["fundamentals"] = dict(nested)
    return out


def merge_sepa_overlay(
    row: Mapping[str, Any],
    sepa_by_symbol: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Attach persisted SEPA overlay fields. Missing overlay stays missing — never invented."""
    out = dict(row)
    if not sepa_by_symbol:
        return out
    sepa = sepa_by_symbol.get(str(out.get("symbol", "") or "").upper())
    if not sepa:
        return out
    for key in SEPA_MERGE_KEYS:
        if key in sepa and sepa.get(key) is not None and out.get(key) is None:
            out[key] = sepa.get(key)
    return out


def load_sepa_overlay_cards(
    scanned_at: str,
    sepa_cards: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cards = [dict(c) for c in (sepa_cards or []) if isinstance(c, Mapping) and c.get("symbol")]
    if cards:
        return cards
    try:
        from product.sepa_setup import load_persisted_best_setups

        persisted = load_persisted_best_setups(str(scanned_at or ""))
        if persisted:
            return [dict(c) for c in persisted[0] if isinstance(c, Mapping) and c.get("symbol")]
    except Exception:
        return []
    return []


def best_among_note(
    *,
    pick: Mapping[str, Any] | None,
    sniper_count: int,
    fund_count: int,
    sepa_count: int,
    overlap_count: int,
) -> str:
    if pick:
        why = str(pick.get("best_among_why") or pick.get("symbol") or "")
        return (
            f"Picked {pick.get('symbol')} — sniper plus {why}. "
            f"{overlap_count} sniper name(s) had a second screen "
            f"({sepa_count} SEPA overlay, {fund_count} long-term funds)."
        )
    return (
        f"No overlap yet: {sniper_count} sniper names, {sepa_count} SEPA overlay cards, "
        f"{fund_count} long-term quality names. Scan Now fills funds from the same "
        "saved scan; Refresh funds only reloads Screener. "
        "SEPA cards come from the saved scan overlay. Tape lane above is independent."
    )


def classify_breakout_state(row: Mapping[str, Any]) -> str:
    """Deterministic breakout sub-state for retail scanner tables."""
    status = str(row.get("status", "") or "")
    if bool(row.get("chase_risk")):
        return "extended_after_breakout"
    signals = _signals(row)
    # Volume floor is absolute — VOL_SQUEEZE is a pattern tag, not a volume waiver.
    vol_ok = passes_volume_floor(row)
    if "PRE_BREAKOUT" in signals or status == "Watch for breakout":
        return "near_breakout"
    if signals & BREAKOUT_TAGS:
        if str(row.get("verdict", "")).upper() == "BUY" and status == "Ready to trade":
            if not vol_ok:
                return "breakout_without_volume"
            return "confirmed_breakout"
        if status == "Wait for pullback":
            return "failed_breakout"
        return "breakout_under_observation"
    if status == "Wait for pullback":
        return "failed_or_extended"
    if not signals and status in {"Watch", "Watch for breakout"}:
        return "insufficient_data"
    return "not_in_breakout_lane"


def classify_momentum_state(row: Mapping[str, Any]) -> str:
    """Momentum strength vs extension without inventing new scores."""
    if "MOMENTUM" not in _signals(row):
        return "not_momentum"
    if bool(row.get("chase_risk")):
        return "strong_but_extended"
    status = str(row.get("status", "") or "")
    score = _f(row.get("score"))
    mom5 = _f(row.get("momentum_5d"))
    mom20 = _f(row.get("momentum_20d") or row.get("momentum_21d"))
    vol = _f(row.get("atr_pct") or row.get("volatility"))
    if status == "Ready to trade" and score >= 65:
        if vol >= 4.0:
            return "high_volatility_momentum"
        return "strong_actionable"
    if mom5 > 0 and mom20 > 0 and score >= 60:
        return "steady_leadership"
    if mom5 > 0 and score >= 55:
        return "improving"
    if mom5 < 0:
        return "weakening"
    hist = _f(row.get("history_days") or row.get("sessions") or 0)
    if hist and hist < 120:
        return "insufficient_history"
    return "watch_momentum"


def default_sector_lookup(symbol: str) -> str:
    try:
        from scan.sector_heat import sector_of
        return str(sector_of(symbol) or "")
    except Exception:
        return ""


def enrich_scan_row(
    row: Mapping[str, Any],
    *,
    sector_lookup: Callable[[str], str] | None = None,
    scanned_at: str = "",
) -> dict[str, Any]:
    lookup = sector_lookup or default_sector_lookup
    symbol = str(row.get("symbol", "") or "").upper()
    enriched = dict(row)
    enriched["sector"] = str(row.get("sector") or lookup(symbol) or "—")
    enriched["breakout_state"] = classify_breakout_state(row)
    enriched["momentum_state"] = classify_momentum_state(row)
    enriched["setup_label"] = str(row.get("status") or row.get("verdict") or "Watch")
    enriched["freshness"] = scanned_at or "unknown"
    enriched["change_5d_pct"] = _f(row.get("momentum_5d"))
    enriched["relative_strength"] = _f(row.get("score"))
    enriched["risk_label"] = (
        "Chase risk"
        if bool(row.get("chase_risk"))
        else "Pullback wait"
        if str(row.get("status", "")) == "Wait for pullback"
        else "Normal"
    )
    reason = row.get("reasons") or row.get("why") or []
    enriched["reason"] = str(reason[0] if isinstance(reason, list) and reason else row.get("why") or "")
    # Sniper ranking fields — always present so Market Scanner / radar can
    # surface a BEST candidate without a second pass.
    enriched["sniper_candidate"] = is_sniper_breakout_candidate(enriched)
    enriched["breakout_quality"] = breakout_quality_score(enriched)
    return enriched


def enrich_long_term_row(row: Mapping[str, Any], *, scanned_at: str = "") -> dict[str, Any]:
    lookup = default_sector_lookup
    symbol = str(row.get("symbol", "") or "").upper()
    enriched = dict(row)
    enriched["sector"] = str(row.get("sector") or lookup(symbol) or "—")
    enriched["setup_label"] = str(row.get("classification") or "Unclassified")
    enriched["freshness"] = scanned_at or "unknown"
    enriched["reason"] = ", ".join((row.get("quality_factors") or [])[:2]) or "Long-term screen"
    enriched["risk_label"] = ", ".join((row.get("risk_flags") or [])[:1]) or "Review risks"
    cov = row.get("fundamental_coverage")
    enriched["coverage_pct"] = round(_f(cov) * 100, 1) if cov is not None else None
    return enriched


def build_radar_home(
    *,
    scan_payload: Mapping[str, Any] | None,
    long_term_payload: Mapping[str, Any] | None,
    market: Any,
    sector_lookup: Callable[[str], str] | None = None,
    sepa_cards: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    scan_rows = [dict(r) for r in (scan_payload or {}).get("records", []) or [] if isinstance(r, Mapping)]
    long_rows = [dict(r) for r in (long_term_payload or {}).get("records", []) or [] if isinstance(r, Mapping)]
    scan_at = str((scan_payload or {}).get("scanned_at", "") or "")
    lt_at = str((long_term_payload or {}).get("scanned_at", "") or "")

    fund_by_symbol = {
        str(r.get("symbol", "") or "").upper(): r
        for r in long_rows
        if str(r.get("symbol", "") or "")
    }
    sepa_overlay = load_sepa_overlay_cards(scan_at, sepa_cards)
    sepa_by_symbol = {
        str(r.get("symbol", "") or "").upper(): r
        for r in sepa_overlay
        if str(r.get("symbol", "") or "")
    }
    enriched = [
        enrich_scan_row(
            merge_sepa_overlay(merge_fundamental_context(r, fund_by_symbol), sepa_by_symbol),
            sector_lookup=sector_lookup,
            scanned_at=scan_at,
        )
        for r in scan_rows
    ]

    # Pre-filter breakout-ish rows, then refresh RSI/price from the store
    # (bulk live overlay once). Cap the refresh set — refreshing hundreds of
    # rows with per-symbol quote scrapers made radar-home time out so the UI
    # showed ZERO snipers even when ~30 were eligible.
    breakout_states = {
        "confirmed_breakout", "near_breakout", "insufficient_confirmation", "extended_after_breakout",
    }
    breakouts = [r for r in enriched if r.get("breakout_state") in breakout_states]
    try:
        from product.live_technicals import refresh_rows_technicals
        priority = [
            r for r in breakouts
            if r.get("breakout_state") in {"confirmed_breakout", "near_breakout"}
            and not bool(r.get("chase_risk"))
        ]
        priority.sort(key=lambda r: (-_f(r.get("score")), r.get("symbol", "")))
        refresh_cap = 80
        to_refresh = priority[:refresh_cap]
        # Always include graded A/B names even if outside the score head.
        seen = {str(r.get("symbol", "")).upper() for r in to_refresh}
        for r in breakouts:
            sym = str(r.get("symbol", "")).upper()
            if sym in seen:
                continue
            if str(r.get("breakout_grade") or "").upper() in {"A", "B"} and not bool(r.get("chase_risk")):
                to_refresh.append(r)
                seen.add(sym)
            if len(to_refresh) >= refresh_cap + 20:
                break
        refreshed = {
            str(r.get("symbol", "")).upper(): r
            for r in refresh_rows_technicals(to_refresh, bulk_overlay=True)
        }
        breakouts = [refreshed.get(str(r.get("symbol", "")).upper(), r) for r in breakouts]
        by_sym = {str(r.get("symbol", "")).upper(): r for r in enriched}
        for sym, row in refreshed.items():
            if sym in by_sym:
                by_sym[sym].update({
                    k: row[k] for k in (
                        "price", "rsi", "volume_ratio", "tech_source", "price_tag",
                        "eod_as_of", "quote_source",
                    ) if k in row
                })
    except Exception:
        pass

    for row in breakouts:
        row["breakout_quality"] = breakout_quality_score(row)
        row["sniper_candidate"] = is_sniper_breakout_candidate(row)
    for row in enriched:
        if "sniper_candidate" not in row:
            row["breakout_quality"] = breakout_quality_score(row)
            row["sniper_candidate"] = is_sniper_breakout_candidate(row)

    # Sniper-first: only volume≥1× / non-blow-off / non-chase names lead the lane.
    breakouts.sort(key=lambda r: (
        not bool(r.get("sniper_candidate")),
        not passes_volume_floor(r),
        _f(r.get("rsi")) > RSI_HARD,
        bool(r.get("chase_risk")),
        r["breakout_state"] != "confirmed_breakout",
        r["breakout_state"] != "near_breakout",
        -_f(r.get("breakout_quality")),
        -_f(r.get("breakout_conviction")),
        -_f(r.get("score")),
        r.get("symbol", ""),
    ))

    momentum = [r for r in enriched if "MOMENTUM" in _signals(r)]
    # Refresh technicals for the momentum lane head (visible cards only).
    try:
        from product.live_technicals import refresh_rows_technicals
        momentum = refresh_rows_technicals(momentum[:24], bulk_overlay=False) + momentum[24:]
    except Exception:
        pass
    momentum.sort(key=lambda r: (bool(r.get("chase_risk")), -_f(r.get("score")), r.get("symbol", "")))

    long_picks = [
        enrich_long_term_row(r, scanned_at=lt_at)
        for r in long_rows
        if is_long_term_pick(r)
    ]
    long_picks.sort(key=lambda r: (-_f(r.get("combined_score")), r.get("symbol", "")))

    best_breakout = pick_best_sniper_breakout(breakouts)
    best_of_best = rank_best_of_best(breakouts, limit=8)
    best_among_fundamentals = best_of_best[0] if best_of_best else None
    sniper_breakouts = [
        attach_best_pick_meta(r, with_context=False, for_best=False)
        for r in breakouts
        if r.get("sniper_candidate") and passes_volume_floor(r)
    ]
    sniper_syms = {str(r.get("symbol", "") or "").upper() for r in sniper_breakouts}
    sepa_syms = {str(r.get("symbol", "") or "").upper() for r in sepa_overlay}
    fund_syms = {str(r.get("symbol", "") or "").upper() for r in long_picks}
    second_screen_syms = {
        str(r.get("symbol", "") or "").upper()
        for r in sniper_breakouts
        if has_second_screen(r)
    }
    among_note = best_among_note(
        pick=best_among_fundamentals,
        sniper_count=len(sniper_breakouts),
        fund_count=len(fund_syms),
        sepa_count=len(sepa_syms),
        overlap_count=len(second_screen_syms),
    )
    health = str(getattr(market, "health", "") or (market or {}).get("health", "Unavailable") if isinstance(market, Mapping) else "Unavailable")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_session": str(getattr(market, "trade_stance", "") or (market or {}).get("trade_stance", "") if isinstance(market, Mapping) else ""),
        "market_health": health,
        "breadth": str(getattr(market, "breadth", "") or (market or {}).get("breadth", "") if isinstance(market, Mapping) else ""),
        "nifty_change_1d": _f(getattr(market, "nifty_change_1d", 0) if not isinstance(market, Mapping) else market.get("nifty_change_1d")),
        "vix": _f(getattr(market, "vix", 0) if not isinstance(market, Mapping) else market.get("vix")),
        "leaders": list(getattr(market, "leaders", ()) or (market or {}).get("leaders", []) if isinstance(market, Mapping) else []),
        "laggards": list(getattr(market, "laggards", ()) or (market or {}).get("laggards", []) if isinstance(market, Mapping) else []),
        "scan_scanned_at": scan_at,
        "long_term_scanned_at": lt_at,
        "universe_size": int((scan_payload or {}).get("universe_size", 0) or 0),
        "best_breakout": best_breakout,
        "best_among_fundamentals": best_among_fundamentals,
        "best_of_best": best_of_best,
        "best_among_note": among_note,
        "sniper_candidates": sniper_breakouts[:12],
        "ranking_legend": dict(RANKING_LEGEND),
        "scan_shared_note": SCAN_SHARED_NOTE,
        "sepa_rank_used": bool(sepa_overlay),
        "second_screen_counts": {
            "sniper": len(sniper_syms),
            "sepa_overlay": len(sepa_syms),
            "long_term_funds": len(fund_syms),
            "sniper_with_second_screen": len(second_screen_syms),
        },
        "lanes": {
            "breakouts": breakouts[:12],
            "momentum": momentum[:12],
            "long_term_picks": long_picks[:12],
        },
        "counts": {
            "breakouts": len(breakouts),
            "momentum": len(momentum),
            "long_term_picks": len(long_picks),
            "sniper_breakouts": len(sniper_breakouts),
        },
    }


def enrich_scanner_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scanned_at: str = "",
    sector_lookup: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    return [enrich_scan_row(dict(row), sector_lookup=sector_lookup, scanned_at=scanned_at) for row in rows]
