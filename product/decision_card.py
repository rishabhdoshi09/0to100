"""Reco-like decision surface over QuantTerm evidence.

Complexity stays in the engine. The customer-facing card answers:

  what is the opportunity, where to enter, what is the target / stop,
  how much upside, what is the risk, why now, what would change our mind.

Never invent prices, EV, or breadth. Missing inputs stay None / Unproven /
Unmeasured — the same honesty contract as the rest of the product.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

# Category research horizons — labels for the *style* of the bucket,
# not a stock-specific forecast.
HORIZON_BY_CATEGORY = {
    "wealth_builders": "12–36 months",
    "super_trends": "3–9 months",
    "momentum_breakouts": "2–8 weeks",
    "recovery_setups": "3–12 months",
}

_SKIP_WHY_TAGS = frozenset({
    "QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE",
    "NEEDS_FUNDAMENTALS",
})
_WHY_NOW_LABELS = {
    "BREAKOUT_52W": "52-week high breakout",
    "BREAKOUT_RES": "Resistance break with volume",
    "GOLDEN_CROSS": "Trend structure turning up",
    "VOL_SQUEEZE": "Volatility squeeze resolving",
    "VCP": "Base is tightening",
    "FLAT_BASE": "Flat base near breakout",
    "CUP_HANDLE": "Cup-and-handle structure",
    "HIGH_TIGHT_FLAG": "High-tight flag",
    "ASC_TRIANGLE": "Ascending triangle",
    "DOUBLE_BOTTOM": "Double-bottom recovery structure",
    "PRE_BREAKOUT": "Price is near the breakout pivot",
    "ACCUMULATION": "Accumulation in the base",
    "DELIVERY_SPIKE": "Delivery buying is rising",
    "NR7_COIL": "Price is coiled (tight range)",
    "POCKET_PIVOT": "Pocket-pivot volume",
    "MOMENTUM": "Momentum improving",
    "PULLBACK_SUPPORT": "Pullback to support in an uptrend",
    "NEAR_BREAKOUT": "Price is near the breakout pivot",
    "CONFIRMED_BREAKOUT": "Breakout is confirmed",
    "BREAKOUT_UNDER_OBSERVATION": "Breakout is under observation",
    "STRONG_ACTIONABLE": "Setup is ready to trade",
    "STEADY_LEADERSHIP": "Steady leadership versus the market",
    "IMPROVING": "Momentum is improving",
}

_HEALTH_RANK = {"Degraded": 0, "Caution": 1, "Normal": 2, "Unmeasured": 3}
_BREADTH_TO_SUPPORT = {"HEALTHY": "Positive", "MIXED": "Mixed", "NARROW": "Weak"}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _signals(row: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for item in row.get("signals") or []:
        key = str(item or "").strip().upper()
        if key and key not in out:
            out.append(key)
    return out


def health_from_expectancy(expectancy_r: float) -> str:
    if expectancy_r >= 0.10:
        return "Normal"
    if expectancy_r >= -0.10:
        return "Caution"
    return "Degraded"


def buy_zone(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Entry band from a real entry. ATR widens it; never invents an entry."""
    entry = _f(row.get("entry") or row.get("entry_price"))
    if entry <= 0:
        return None, None
    atr_pct = _f(row.get("atr_pct") or row.get("volatility"))
    price = _f(row.get("price") or row.get("cmp")) or entry
    atr = price * atr_pct / 100.0 if atr_pct > 0 else 0.0
    if atr <= 0:
        rounded = round(entry, 2)
        return rounded, rounded
    lo = round(entry - 0.5 * atr, 2)
    hi = round(entry + 0.5 * atr, 2)
    stop = _f(row.get("stop") or row.get("stop_price"))
    if stop > 0:
        lo = max(lo, round(stop * 1.002, 2))
    if lo > hi:
        lo = hi = round(entry, 2)
    return lo, hi


def expected_payoff(row: Mapping[str, Any]) -> tuple[str, str]:
    """Positive / Negative / Unproven from conservative EV only."""
    n = row.get("ev_n")
    lb = row.get("ev_lb_pct")
    try:
        n_i = int(n) if n is not None else 0
    except (TypeError, ValueError):
        n_i = 0
    if lb is None or n_i < 30:
        return (
            "Unproven",
            "Fewer than 30 comparable outcomes — no expected-payoff claim.",
        )
    lb_f = _f(lb)
    conf = str(row.get("ev_conf") or "LOW")
    detail = f"Conservative EV {lb_f:+.1f}% (n={n_i}, {conf})"
    if lb_f > 0:
        return "Positive", detail
    if lb_f < 0:
        return "Negative", detail
    return "Unproven", detail + " — bound sits at zero."


def evidence_strength(row: Mapping[str, Any]) -> str:
    """Strong only when live EV confidence is HIGH. Structural setups stay thinner."""
    n = row.get("ev_n")
    conf = str(row.get("ev_conf") or "").upper()
    try:
        n_i = int(n) if n is not None else 0
    except (TypeError, ValueError):
        n_i = 0
    if n_i >= 30 and conf == "HIGH":
        return "Strong"
    if n_i >= 30 and conf == "MEDIUM":
        return "Moderate"
    cov = row.get("fundamental_coverage")
    score = _f(row.get("score") or row.get("combined_score"))
    grade = str(row.get("breakout_grade") or "").upper()
    if cov is not None and _f(cov) >= 0.70 and score >= 70:
        return "Moderate"
    if grade == "A" and score >= 70:
        return "Moderate"
    return "Thin"


def opportunity_label(action_badge: str) -> str:
    a = (action_badge or "").lower()
    if "buy" in a:
        return "OPPORTUNITY"
    if "research" in a or "hold" in a:
        return "RESEARCH"
    if a in {"win", "loss", "void", "closed"}:
        return "CLOSED"
    if a in {"open", "tracked"}:
        return "TRACKED"
    return "WATCH"


def why_now(
    row: Mapping[str, Any],
    *,
    qualify_reason: str = "",
    evidence_tags: Sequence[str] | None = None,
    limit: int = 4,
) -> list[str]:
    bullets: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        line = " ".join(str(text or "").split())
        if not line or line.lower() in seen:
            return
        seen.add(line.lower())
        bullets.append(line)

    for sig in _signals(row):
        if sig in _WHY_NOW_LABELS:
            add(_WHY_NOW_LABELS[sig])
    for tag in evidence_tags or []:
        key = str(tag or "").strip().upper()
        if key in _WHY_NOW_LABELS:
            add(_WHY_NOW_LABELS[key])
        elif (
            key
            and key not in _SKIP_WHY_TAGS
            and not key.startswith("GRADE_")
            and "COVERAGE_" not in key
        ):
            add(str(tag).replace("_", " "))
    vol = _f(row.get("volume_ratio"))
    if vol >= 1.3:
        add("Volume confirmation")
    if row.get("above_sma50") is True:
        add("Holding above the 50-day trend")
    reasons = row.get("reasons") or []
    if isinstance(reasons, list):
        for item in reasons[:3]:
            add(str(item))
    for factor in (row.get("quality_factors") or [])[:2]:
        add(str(factor))
    if qualify_reason:
        # Qualify line is often a packed summary — keep it if we still have room.
        if len(bullets) < limit:
            add(qualify_reason)
    if not bullets and str(row.get("reason") or "").strip():
        add(str(row.get("reason")))
    return bullets[:limit]


def what_changes_mind(row: Mapping[str, Any], *, category_id: str = "") -> list[str]:
    items: list[str] = []
    stop = _f(row.get("stop") or row.get("stop_price"))
    if stop > 0:
        items.append(f"Price closes below ₹{stop:,.2f}".replace(".00", ""))
    if bool(row.get("chase_risk")):
        items.append("Price stays extended — chase risk remains")
    sector = str(row.get("sector") or "").strip()
    if sector and sector not in {"—", "-", "Unknown"}:
        items.append(f"{sector} leadership deteriorates")
    rsi = _f(row.get("rsi"))
    if rsi >= 72:
        items.append("RSI stays in blow-off territory")
    if category_id == "wealth_builders":
        items.append("Fundamental coverage or quality factors break down")
    items.append("Evidence model detects strategy degradation")
    # Deduplicate while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:4]


def next_step(action_badge: str, *, chase_risk: bool = False) -> str:
    if chase_risk:
        return "Wait for a pullback into the buy zone — do not chase."
    a = (action_badge or "").lower()
    if "buy" in a:
        return "Review the buy zone, target and stop, then open the deeper research."
    if "research" in a or "hold" in a:
        return "Read the quality thesis before sizing — this is research, not a chase."
    if a in {"open", "tracked"}:
        return "This pick is already tracked — manage the existing plan."
    return "Watch for confirmation. Use See Evidence before acting."


def _card_strategy_health(
    row: Mapping[str, Any],
    market_ctx: Mapping[str, Any],
) -> tuple[str, str]:
    per = market_ctx.get("signal_health") or {}
    hits: list[tuple[str, str]] = []
    for sig in _signals(row):
        info = per.get(sig)
        if isinstance(info, Mapping) and info.get("health"):
            hits.append((str(info["health"]), sig))
    if hits:
        worst = min(hits, key=lambda pair: _HEALTH_RANK.get(pair[0], 3))
        info = per.get(worst[1]) or {}
        n = info.get("n")
        exp = info.get("expectancy_r")
        detail = f"{worst[1]} live expectancy {exp:+.2f}R" if exp is not None else worst[1]
        if n:
            detail += f" (n={n})"
        return worst[0], detail
    health = str(market_ctx.get("strategy_health") or "Unmeasured")
    detail = str(market_ctx.get("strategy_health_detail") or "")
    return health, detail


def evidence_panel(row: Mapping[str, Any]) -> dict[str, Any]:
    n = row.get("ev_n")
    try:
        sample = int(n) if n is not None else None
    except (TypeError, ValueError):
        sample = None
    cov = row.get("fundamental_coverage")
    return {
        "sample_size": sample,
        "ev_pct": row.get("ev_pct"),
        "ev_lb_pct": row.get("ev_lb_pct"),
        "p_win": row.get("p_win"),
        "confidence": row.get("ev_conf"),
        "score": _f(row.get("score") or row.get("combined_score")) or None,
        "rsi": _f(row.get("rsi")) or None,
        "volume_ratio": _f(row.get("volume_ratio")) or None,
        "signals": _signals(row),
        "price_tag": str(row.get("price_tag") or ""),
        "tech_source": str(row.get("tech_source") or ""),
        "fundamental_coverage": round(_f(cov) * 100.0, 1) if cov is not None else None,
        "provenance": (
            "Saved market scan; CMP from live overlay when available. "
            "Expected payoff only with ≥30 comparable outcomes."
        ),
    }


def build_desk_context(scan_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Once-per-workspace tape + live-edge snapshot. No network."""
    market_support = "Unmeasured"
    market_support_detail = (
        "Breadth needs ≥300 scan rows with a daily change — no market-support claim yet."
    )
    try:
        from scan.breadth import breadth_from_results

        br = breadth_from_results([dict(r) for r in (scan_rows or [])])
        verdict = str(br.get("verdict") or "")
        if verdict in _BREADTH_TO_SUPPORT:
            market_support = _BREADTH_TO_SUPPORT[verdict]
            market_support_detail = str(br.get("line") or verdict)
        else:
            market_support_detail = (
                f"Scan sample n={int(br.get('n') or 0)} is below the 300-stock breadth gate."
            )
    except Exception:
        pass

    strategy_health = "Unmeasured"
    strategy_health_detail = "Fewer than 30 closed outcomes — no strategy-health claim."
    signal_health: dict[str, dict[str, Any]] = {}
    live_n = 0
    try:
        from scan.live_edge import profile_edge

        prof = profile_edge()
        overall = prof.get("overall") or {}
        live_n = int(overall.get("n") or 0)
        if live_n >= 30:
            exp = float(overall.get("expectancy_r") or 0.0)
            strategy_health = health_from_expectancy(exp)
            strategy_health_detail = (
                f"Live expectancy {exp:+.2f}R over {live_n} closed outcomes."
            )
        for sig, stats in (prof.get("signals") or {}).items():
            n = int((stats or {}).get("n") or 0)
            if n < 30:
                continue
            exp = float((stats or {}).get("expectancy_r") or 0.0)
            signal_health[str(sig)] = {
                "health": health_from_expectancy(exp),
                "n": n,
                "expectancy_r": exp,
            }
    except Exception:
        pass

    return {
        "market_support": market_support,
        "market_support_detail": market_support_detail,
        "strategy_health": strategy_health,
        "strategy_health_detail": strategy_health_detail,
        "signal_health": signal_health,
        "live_n": live_n,
    }


def attach_live_ev(rows: Sequence[Mapping[str, Any]]) -> None:
    """Best-effort EV tags on scan rows. Silent when the tracker is empty."""
    live = [r for r in rows if isinstance(r, dict)]
    if not live:
        return
    try:
        from scan.ev_engine import tag_ev

        tag_ev(live)
    except Exception:
        return


def decision_surface(
    row: Mapping[str, Any],
    *,
    category_id: str,
    action_badge: str,
    qualify_reason: str = "",
    evidence_tags: Sequence[str] | None = None,
    market_ctx: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Retail-facing decision fields. All numbers come from `row` or stay empty."""
    ctx = dict(market_ctx or {})
    zone_lo, zone_hi = buy_zone(row)
    stop = _f(row.get("stop") or row.get("stop_price"))
    payoff, payoff_detail = expected_payoff(row)
    health, health_detail = _card_strategy_health(row, ctx)
    support = str(ctx.get("market_support") or "Unmeasured")
    badge = action_badge or "Watch"
    return {
        "stop": stop or None,
        "buy_zone_low": zone_lo,
        "buy_zone_high": zone_hi,
        "horizon": HORIZON_BY_CATEGORY.get(category_id, ""),
        "opportunity_label": opportunity_label(badge),
        "expected_payoff": payoff,
        "expected_payoff_detail": payoff_detail,
        "evidence": evidence_strength(row),
        "strategy_health": health,
        "strategy_health_detail": health_detail,
        "market_support": support,
        "market_support_detail": str(ctx.get("market_support_detail") or ""),
        "why_now": why_now(
            row, qualify_reason=qualify_reason, evidence_tags=evidence_tags,
        ),
        "what_changes_mind": what_changes_mind(row, category_id=category_id),
        "next_step": next_step(badge, chase_risk=bool(row.get("chase_risk"))),
        "evidence_panel": evidence_panel(row),
        "setup_quality": round(_f(row.get("score") or row.get("combined_score") or row.get("conviction_score"))) or None,
        "setup_quality_label": "Setup Quality",
    }
