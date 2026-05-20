"""
ui/pre_trade_checklist.py
Pre-trade gate that JARVIS runs before any trade is placed.
Scores and grades the trade idea across 7 dimensions.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChecklistItem:
    question: str
    passed: Optional[bool]   # True=pass, False=fail, None=caution/unknown
    detail: str
    weight: int              # 1-3


@dataclass
class ChecklistResult:
    symbol: str
    action: str
    items: list[ChecklistItem]
    score: float             # 0-100
    grade: str               # "GO" | "CAUTION" | "ABORT"
    summary: str
    timestamp: str


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------

def _check_regime_aligned(action: str) -> ChecklistItem:
    """Check 1 (weight=3): Is the trade direction aligned with the current regime?"""
    question = "Regime aligned with trade direction?"
    try:
        from core.regime_engine import compute_regime
        state = compute_regime()
        regime = state.market_regime
        bullish_regimes = {"TRENDING_BULL", "EXPANSION"}
        bearish_regimes = {"TRENDING_BEAR", "DISTRIBUTION"}

        if action.upper() == "BUY":
            if regime in bullish_regimes:
                return ChecklistItem(question, True, f"Regime is {regime} — favours longs.", 3)
            elif regime == "CHOPPY":
                return ChecklistItem(question, None, f"Regime is CHOPPY — proceed with caution on longs.", 3)
            else:
                return ChecklistItem(question, False, f"Regime is {regime} — unfavourable for longs.", 3)
        else:  # SELL / SHORT
            if regime in bearish_regimes:
                return ChecklistItem(question, True, f"Regime is {regime} — favours shorts.", 3)
            elif regime == "CHOPPY":
                return ChecklistItem(question, None, f"Regime is CHOPPY — proceed with caution on shorts.", 3)
            else:
                return ChecklistItem(question, False, f"Regime is {regime} — unfavourable for shorts.", 3)
    except Exception as exc:
        return ChecklistItem(question, None, f"Could not fetch regime: {exc}", 3)


def _check_position_size(price: float, stop: float, qty: int, account_size: float) -> ChecklistItem:
    """Check 2 (weight=3): Is the risk per trade within acceptable limits?"""
    question = "Position size within risk limits?"
    try:
        risk_per_share = abs(price - stop)
        total_risk = risk_per_share * qty
        risk_pct = (total_risk / account_size) * 100

        if risk_pct < 2.0:
            return ChecklistItem(
                question, True,
                f"Risk ₹{total_risk:,.0f} = {risk_pct:.2f}% of account — within 2% limit.",
                3,
            )
        elif risk_pct <= 3.0:
            return ChecklistItem(
                question, None,
                f"Risk ₹{total_risk:,.0f} = {risk_pct:.2f}% of account — borderline (2-3%).",
                3,
            )
        else:
            return ChecklistItem(
                question, False,
                f"Risk ₹{total_risk:,.0f} = {risk_pct:.2f}% of account — exceeds 3% limit.",
                3,
            )
    except Exception as exc:
        return ChecklistItem(question, None, f"Could not compute position size: {exc}", 3)


def _check_stop_defined(action: str, price: float, stop: float) -> ChecklistItem:
    """Check 3 (weight=3): Is a valid stop loss defined?"""
    question = "Stop loss properly defined?"
    if stop is None or stop <= 0:
        return ChecklistItem(question, False, "Stop is not set or is zero.", 3)

    if action.upper() == "BUY":
        if stop < price:
            distance_pct = ((price - stop) / price) * 100
            return ChecklistItem(
                question, True,
                f"Stop at ₹{stop:.2f} is {distance_pct:.1f}% below entry — valid for BUY.",
                3,
            )
        else:
            return ChecklistItem(
                question, False,
                f"Stop ₹{stop:.2f} is above entry ₹{price:.2f} — invalid for BUY.",
                3,
            )
    else:  # SELL / SHORT
        if stop > price:
            distance_pct = ((stop - price) / price) * 100
            return ChecklistItem(
                question, True,
                f"Stop at ₹{stop:.2f} is {distance_pct:.1f}% above entry — valid for SELL.",
                3,
            )
        else:
            return ChecklistItem(
                question, False,
                f"Stop ₹{stop:.2f} is below entry ₹{price:.2f} — invalid for SELL.",
                3,
            )


def _check_rr_ratio(action: str, price: float, stop: float, target: float) -> ChecklistItem:
    """Check 4 (weight=2): Is the reward-to-risk ratio acceptable?"""
    question = "Reward:Risk ratio acceptable?"
    try:
        if action.upper() == "BUY":
            risk = price - stop
            reward = target - price
        else:
            risk = stop - price
            reward = price - target

        if risk <= 0:
            return ChecklistItem(question, False, "Risk leg is zero or negative — cannot compute R:R.", 2)

        rr = reward / risk
        if rr >= 2.0:
            return ChecklistItem(question, True, f"R:R = {rr:.2f} — meets minimum 2:1 requirement.", 2)
        elif rr >= 1.5:
            return ChecklistItem(question, None, f"R:R = {rr:.2f} — borderline (1.5-2.0), consider adjusting target.", 2)
        else:
            return ChecklistItem(question, False, f"R:R = {rr:.2f} — below minimum 1.5:1 threshold.", 2)
    except Exception as exc:
        return ChecklistItem(question, None, f"Could not compute R:R: {exc}", 2)


def _check_not_overextended(symbol: str, action: str) -> ChecklistItem:
    """Check 5 (weight=2): Is the stock overextended on RSI?"""
    question = "Stock not technically overextended?"
    try:
        from agents.tools import get_technical_indicators
        data = get_technical_indicators(symbol)
        if "error" in data:
            return ChecklistItem(question, None, f"Could not fetch indicators: {data['error']}", 2)

        rsi = data.get("rsi")
        if rsi is None:
            return ChecklistItem(question, None, "RSI data not available — skipping check.", 2)

        rsi = float(rsi)
        if action.upper() == "BUY":
            if rsi > 75:
                return ChecklistItem(question, False, f"RSI={rsi:.1f} > 75 — stock is overbought, BUY is risky.", 2)
            else:
                return ChecklistItem(question, True, f"RSI={rsi:.1f} — not overbought, acceptable for BUY.", 2)
        else:  # SELL / SHORT
            if rsi < 25:
                return ChecklistItem(question, False, f"RSI={rsi:.1f} < 25 — stock is oversold, SHORT is risky.", 2)
            else:
                return ChecklistItem(question, True, f"RSI={rsi:.1f} — not oversold, acceptable for SELL.", 2)
    except Exception as exc:
        return ChecklistItem(question, None, f"Error fetching RSI: {exc}", 2)


def _check_opportunity_score() -> tuple[ChecklistItem, float]:
    """Check 6 (weight=2): Is the market opportunity score high enough?"""
    question = "Market opportunity score sufficient?"
    try:
        from core.intelligence_hub import compute_opportunity_score
        opp = compute_opportunity_score()
        total = opp.total

        if total > 45:
            item = ChecklistItem(
                question, True,
                f"Opportunity score {total:.0f}/100 ({opp.grade} — {opp.label}) — favourable.",
                2,
            )
        elif total >= 30:
            item = ChecklistItem(
                question, None,
                f"Opportunity score {total:.0f}/100 ({opp.grade} — {opp.label}) — marginal conditions.",
                2,
            )
        else:
            item = ChecklistItem(
                question, False,
                f"Opportunity score {total:.0f}/100 ({opp.grade} — {opp.label}) — conditions too poor.",
                2,
            )
        return item, total
    except Exception as exc:
        return ChecklistItem(question, None, f"Could not compute opportunity score: {exc}", 2), 50.0


def _check_not_against_trend(action: str) -> ChecklistItem:
    """Check 7 (weight=1): Is the trade not fighting the regime trend?"""
    question = "Trade not fighting the dominant trend?"
    try:
        from core.regime_engine import compute_regime
        state = compute_regime()
        regime_score = state.regime_score  # 0-100 bullishness

        if action.upper() == "BUY":
            if regime_score > 60:
                return ChecklistItem(
                    question, True,
                    f"Regime score {regime_score:.0f}/100 — trend supports long bias.",
                    1,
                )
            else:
                return ChecklistItem(
                    question, None,
                    f"Regime score {regime_score:.0f}/100 — trend does not clearly support longs.",
                    1,
                )
        else:  # SELL / SHORT
            if regime_score < 40:
                return ChecklistItem(
                    question, True,
                    f"Regime score {regime_score:.0f}/100 — trend supports short bias.",
                    1,
                )
            else:
                return ChecklistItem(
                    question, None,
                    f"Regime score {regime_score:.0f}/100 — trend does not clearly support shorts.",
                    1,
                )
    except Exception as exc:
        return ChecklistItem(question, None, f"Could not evaluate trend direction: {exc}", 1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_score(items: list[ChecklistItem]) -> float:
    """
    Weighted score: passed items count fully, caution (None) items count half.
    Score = sum(earned_weight) / sum(max_weight) * 100
    """
    total_weight = sum(item.weight for item in items)
    if total_weight == 0:
        return 0.0

    earned = 0.0
    for item in items:
        if item.passed is True:
            earned += item.weight
        elif item.passed is None:
            earned += item.weight * 0.5
        # False = 0

    return round((earned / total_weight) * 100, 1)


def _build_summary(symbol: str, action: str, score: float, grade: str, items: list[ChecklistItem]) -> str:
    failed = [i for i in items if i.passed is False]
    cautious = [i for i in items if i.passed is None]

    if grade == "GO":
        return (
            f"{symbol} {action} clears all critical checks with a score of {score:.0f}/100. "
            f"Proceed with normal position sizing and monitor for any intraday regime changes."
        )
    elif grade == "CAUTION":
        concerns = ", ".join(i.question.lower().rstrip("?") for i in (failed + cautious)[:2])
        return (
            f"{symbol} {action} has a score of {score:.0f}/100 with concerns around {concerns}. "
            f"Consider reducing position size by 50% or waiting for a better setup."
        )
    else:  # ABORT
        blockers = ", ".join(i.question.lower().rstrip("?") for i in failed[:2]) if failed else "multiple factors"
        return (
            f"{symbol} {action} scores only {score:.0f}/100 — trade fails on {blockers}. "
            f"Do not place this trade; wait for conditions to improve."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_checklist(
    symbol: str,
    action: str,
    price: float,
    stop: float,
    target: float,
    qty: int,
    account_size: float = 1_000_000,
) -> ChecklistResult:
    """
    Run the pre-trade checklist and return a scored, graded ChecklistResult.

    Parameters
    ----------
    symbol       : NSE symbol (e.g. "RELIANCE")
    action       : "BUY" or "SELL"
    price        : Entry price
    stop         : Stop loss price
    target       : Profit target price
    qty          : Number of shares
    account_size : Total account capital (default ₹10,00,000)
    """
    items: list[ChecklistItem] = []

    # 1. Regime aligned (weight=3)
    items.append(_check_regime_aligned(action))

    # 2. Position size safe (weight=3)
    items.append(_check_position_size(price, stop, qty, account_size))

    # 3. Stop defined (weight=3)
    items.append(_check_stop_defined(action, price, stop))

    # 4. R:R ratio (weight=2)
    items.append(_check_rr_ratio(action, price, stop, target))

    # 5. Not overextended (weight=2)
    items.append(_check_not_overextended(symbol, action))

    # 6. Opportunity score (weight=2)
    opp_item, _opp_total = _check_opportunity_score()
    items.append(opp_item)

    # 7. Not against trend (weight=1)
    items.append(_check_not_against_trend(action))

    score = _compute_score(items)

    if score >= 75:
        grade = "GO"
    elif score >= 50:
        grade = "CAUTION"
    else:
        grade = "ABORT"

    summary = _build_summary(symbol, action, score, grade, items)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return ChecklistResult(
        symbol=symbol,
        action=action.upper(),
        items=items,
        score=score,
        grade=grade,
        summary=summary,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Streamlit UI renderer
# ---------------------------------------------------------------------------

def render_checklist_ui(result: ChecklistResult) -> None:
    """Render the ChecklistResult as a Streamlit UI component."""
    try:
        import streamlit as st
    except ImportError:
        print(f"[ChecklistResult] {result.symbol} {result.action} | {result.grade} | Score: {result.score:.0f}")
        for item in result.items:
            icon = "✓" if item.passed is True else ("⚠" if item.passed is None else "✗")
            print(f"  {icon} {item.question}: {item.detail}")
        print(f"\n{result.summary}")
        return

    # Colour scheme
    _grade_colors = {
        "GO": ("#16a34a", "#dcfce7", "GO"),
        "CAUTION": ("#b45309", "#fef3c7", "CAUTION"),
        "ABORT": ("#dc2626", "#fee2e2", "ABORT"),
    }
    fg, bg, label = _grade_colors.get(result.grade, ("#374151", "#f3f4f6", result.grade))

    # Header card
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left:6px solid {fg};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:16px;
        ">
            <h2 style="margin:0;color:{fg};font-size:1.4rem;">
                Pre-Trade Checklist &nbsp;—&nbsp; {result.symbol} {result.action}
                &nbsp;&nbsp;<span style="font-size:1.6rem;">{label}</span>
            </h2>
            <p style="margin:4px 0 0 0;color:#374151;font-size:0.85rem;">
                {result.timestamp}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Score bar
    st.markdown(f"**Overall Score: {result.score:.0f} / 100**")
    bar_color = fg
    st.markdown(
        f"""
        <div style="background:#e5e7eb;border-radius:6px;height:14px;margin-bottom:16px;">
            <div style="
                width:{result.score:.0f}%;
                background:{bar_color};
                height:14px;
                border-radius:6px;
                transition:width 0.4s;
            "></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Checklist items
    st.markdown("**Checks:**")
    _icon_map = {True: "✅", False: "❌", None: "⚠️"}
    _color_map = {True: "#166534", False: "#991b1b", None: "#92400e"}

    for item in result.items:
        icon = _icon_map.get(item.passed, "❓")
        color = _color_map.get(item.passed, "#374151")
        weight_dots = "●" * item.weight
        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:flex-start;
                padding:8px 12px;
                margin-bottom:6px;
                border-radius:6px;
                background:#f9fafb;
                border:1px solid #e5e7eb;
            ">
                <span style="font-size:1.2rem;margin-right:10px;">{icon}</span>
                <div style="flex:1;">
                    <strong style="color:{color};">{item.question}</strong>
                    <span style="color:#9ca3af;font-size:0.75rem;margin-left:8px;" title="Weight">{weight_dots}</span>
                    <br/>
                    <span style="color:#6b7280;font-size:0.85rem;">{item.detail}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Summary
    st.markdown("---")
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-radius:8px;
            padding:14px 18px;
            border:1px solid {fg}40;
        ">
            <strong style="color:{fg};">Verdict:</strong>
            <span style="color:#1f2937;"> {result.summary}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
