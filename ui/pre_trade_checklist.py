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


def _check_volume(symbol: str) -> ChecklistItem:
    """Check: Is volume confirming the move?"""
    question = "Volume confirming the move?"
    try:
        from agents.tools import get_technical_indicators
        data = get_technical_indicators(symbol)
        if "error" in data:
            return ChecklistItem(question, None, f"Could not fetch volume data: {data['error']}", 2)

        vr = data.get("volume_ratio")
        avg_vol = data.get("avg_volume_20d")
        if vr is None:
            return ChecklistItem(question, None, "Volume data unavailable.", 2)

        vr = float(vr)
        avg_str = f" (20d avg: {int(avg_vol):,})" if avg_vol else ""
        if vr >= 1.5:
            return ChecklistItem(question, True, f"Volume ratio {vr:.2f}x above average{avg_str} — strong institutional participation.", 2)
        elif vr >= 0.8:
            return ChecklistItem(question, None, f"Volume ratio {vr:.2f}x — average interest{avg_str}. Prefer > 1.5x for conviction.", 2)
        else:
            return ChecklistItem(question, False, f"Volume ratio {vr:.2f}x — below average{avg_str}. Weak conviction, avoid.", 2)
    except Exception as exc:
        return ChecklistItem(question, None, f"Volume check error: {exc}", 2)


def _check_momentum(symbol: str, action: str) -> ChecklistItem:
    """Check: Is price momentum aligned with the trade direction?"""
    question = "Price momentum aligned with trade?"
    try:
        from agents.tools import get_technical_indicators
        data = get_technical_indicators(symbol)
        if "error" in data:
            return ChecklistItem(question, None, f"Could not fetch momentum data: {data['error']}", 2)

        mom_20 = data.get("momentum_20d_pct")
        mom_5  = data.get("momentum_5d_pct")
        above_ema20 = data.get("pct_above_sma20")  # reuse sma20 proxy
        ema20  = data.get("ema_20")
        ema50  = data.get("ema_50")
        price  = data.get("price") or data.get("ltp")

        parts = []
        score = 0  # 0-3 points

        if mom_20 is not None:
            m20 = float(mom_20)
            parts.append(f"20d mom {m20:+.1f}%")
            score += (1 if (action == "BUY" and m20 > 0) or (action != "BUY" and m20 < 0) else 0)

        if mom_5 is not None:
            m5 = float(mom_5)
            parts.append(f"5d mom {m5:+.1f}%")
            score += (1 if (action == "BUY" and m5 > 0) or (action != "BUY" and m5 < 0) else 0)

        if price and ema20 and ema50:
            p, e20, e50 = float(price), float(ema20), float(ema50)
            above_20 = p > e20
            above_50 = p > e50
            if action == "BUY":
                if above_20 and above_50:
                    parts.append("Price > EMA20 > EMA50 ✓")
                    score += 1
                elif above_20:
                    parts.append("Price > EMA20 but < EMA50")
                else:
                    parts.append("Price < EMA20 ✗")
            else:
                if not above_20 and not above_50:
                    parts.append("Price < EMA20 < EMA50 ✓")
                    score += 1
                elif not above_20:
                    parts.append("Price < EMA20 but > EMA50")
                else:
                    parts.append("Price > EMA20 ✗")

        detail = " | ".join(parts) if parts else "Insufficient data."
        if score >= 2:
            return ChecklistItem(question, True, detail, 2)
        elif score == 1:
            return ChecklistItem(question, None, detail, 2)
        else:
            return ChecklistItem(question, False, detail, 2)
    except Exception as exc:
        return ChecklistItem(question, None, f"Momentum check error: {exc}", 2)


def _check_news_sentiment(symbol: str, action: str) -> ChecklistItem:
    """Check: Is recent news sentiment supportive?"""
    question = "News sentiment supportive?"
    try:
        from agents.tools import get_recent_news
        articles = get_recent_news(symbol, days=3)

        if not articles or (len(articles) == 1 and "error" in articles[0]):
            return ChecklistItem(question, None, "No recent news found — neutral.", 1)

        positive_kw = {"surge", "rally", "beat", "upgrade", "buy", "strong", "growth", "profit",
                       "record", "outperform", "bullish", "positive", "gain", "rise", "up"}
        negative_kw = {"fall", "drop", "crash", "downgrade", "sell", "weak", "loss", "miss",
                       "fraud", "ban", "warning", "decline", "cut", "bearish", "negative", "down"}

        pos = neg = 0
        for a in articles[:5]:
            text = (a.get("headline", "") + " " + a.get("summary", "")).lower()
            pos += sum(1 for kw in positive_kw if kw in text)
            neg += sum(1 for kw in negative_kw if kw in text)

        total = pos + neg
        headline_sample = articles[0].get("headline", "")[:80] if articles else ""
        detail = f'Latest: "{headline_sample}..." | Positive signals: {pos}, Negative: {neg}'

        if action == "BUY":
            if pos > neg:
                return ChecklistItem(question, True, detail, 1)
            elif pos == neg:
                return ChecklistItem(question, None, detail, 1)
            else:
                return ChecklistItem(question, False, detail, 1)
        else:
            if neg > pos:
                return ChecklistItem(question, True, detail, 1)
            elif pos == neg:
                return ChecklistItem(question, None, detail, 1)
            else:
                return ChecklistItem(question, False, detail, 1)
    except Exception as exc:
        return ChecklistItem(question, None, f"News check error: {exc}", 1)


def _check_sector_strength(symbol: str, action: str) -> ChecklistItem:
    """Check: Is the stock's sector showing strength?"""
    question = "Sector showing relative strength?"
    try:
        from core.sector_rotation import compute_rotation_matrix
        matrix = compute_rotation_matrix()
        if not matrix:
            return ChecklistItem(question, None, "Sector rotation data unavailable.", 2)

        # Sort: leaders have positive 1w return, laggards negative
        sectors_list = list(matrix.values()) if isinstance(matrix, dict) else matrix
        sorted_sectors = sorted(sectors_list, key=lambda x: x.get("return_1w", 0), reverse=True)
        leaders  = [s["sector"] for s in sorted_sectors if s.get("return_1w", 0) > 0]
        laggards = [s["sector"] for s in sorted_sectors if s.get("return_1w", 0) < 0]

        top_leaders  = ", ".join(leaders[:3])  if leaders  else "none"
        top_laggards = ", ".join(laggards[:3]) if laggards else "none"
        detail = f"Leading: {top_leaders} | Lagging: {top_laggards}"

        # Try to map stock to a sector
        stock_sector = None
        try:
            from fundamentals.fetcher import get_deep_fundamentals
            fund = get_deep_fundamentals(symbol, force_refresh=False) or {}
            stock_sector = fund.get("sector") or fund.get("industry")
        except Exception:
            pass

        if stock_sector:
            detail = f"Stock sector: {stock_sector} | " + detail
            in_leaders  = any(stock_sector.lower() in l.lower() for l in leaders)
            in_laggards = any(stock_sector.lower() in l.lower() for l in laggards)
            if action == "BUY":
                if in_leaders:
                    return ChecklistItem(question, True, detail, 2)
                elif in_laggards:
                    return ChecklistItem(question, False, detail, 2)

        # Fallback: broad market breadth
        if action == "BUY":
            return ChecklistItem(question, True if len(leaders) >= len(laggards) else None, detail, 2)
        else:
            return ChecklistItem(question, True if len(laggards) >= len(leaders) else None, detail, 2)
    except Exception as exc:
        return ChecklistItem(question, None, f"Sector check error: {exc}", 2)


def _check_chart_pattern(symbol: str, action: str) -> ChecklistItem:
    """Check: Is there a valid chart pattern / technical structure?"""
    question = "Chart pattern / structure valid?"
    try:
        from agents.tools import get_technical_indicators
        data = get_technical_indicators(symbol)
        if "error" in data:
            return ChecklistItem(question, None, f"Could not fetch chart data: {data['error']}", 2)

        findings = []
        score = 0  # 0-4

        # Golden / death cross
        gc = data.get("golden_cross")
        if gc is True:
            findings.append("Golden cross (SMA50 > SMA200) ✓")
            score += 1 if action == "BUY" else 0
        elif gc is False:
            findings.append("Death cross (SMA50 < SMA200)")
            score += 1 if action != "BUY" else 0

        # Z-score: is price extended?
        zscore = data.get("zscore_20")
        if zscore is not None:
            z = float(zscore)
            if abs(z) < 1.5:
                findings.append(f"Z-score {z:.2f} (within normal range) ✓")
                score += 1
            elif abs(z) < 2.5:
                findings.append(f"Z-score {z:.2f} (mildly extended)")
            else:
                findings.append(f"Z-score {z:.2f} (extremely extended) ✗")

        # Trend direction
        trend = data.get("trend_5d")
        if trend is not None:
            t = float(trend)
            if action == "BUY" and t > 0:
                findings.append(f"5d trend slope positive (+{t:.3f}) ✓")
                score += 1
            elif action != "BUY" and t < 0:
                findings.append(f"5d trend slope negative ({t:.3f}) ✓")
                score += 1
            else:
                findings.append(f"5d trend slope {t:+.3f} (against trade direction)")

        # ATR context
        atr_pct = data.get("atr_pct")
        if atr_pct is not None:
            findings.append(f"ATR {float(atr_pct):.1f}% daily range")

        detail = " | ".join(findings) if findings else "Insufficient data for pattern check."

        if score >= 3:
            return ChecklistItem(question, True, detail, 2)
        elif score >= 1:
            return ChecklistItem(question, None, detail, 2)
        else:
            return ChecklistItem(question, False, detail, 2)
    except Exception as exc:
        return ChecklistItem(question, None, f"Pattern check error: {exc}", 2)


def _check_market_timing() -> ChecklistItem:
    """Check: Is this a good time of day to enter a trade?"""
    question = "Intraday timing optimal for entry?"
    try:
        from core.market_timing import get_timing_score
        t = get_timing_score()
        score = t["score"]
        label = t["label"]
        reason = t["reason"]
        time_str = t.get("time", "")
        detail = f"{label} ({score}/100) at {time_str} — {reason}"
        if label in ("OPTIMAL", "GOOD"):
            return ChecklistItem(question, True, detail, 1)
        elif label in ("CAUTION", "WEAK"):
            return ChecklistItem(question, None, detail, 1)
        else:  # DANGER, CLOSED, PRE-MARKET
            return ChecklistItem(question, False, detail, 1)
    except Exception as exc:
        return ChecklistItem(question, None, f"Timing check error: {exc}", 1)


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

    # 5. Not overextended / RSI (weight=2)
    items.append(_check_not_overextended(symbol, action))

    # 6. Opportunity score (weight=2)
    opp_item, _opp_total = _check_opportunity_score()
    items.append(opp_item)

    # 7. Not against trend (weight=1)
    items.append(_check_not_against_trend(action))

    # 8. Volume confirmation (weight=2)
    items.append(_check_volume(symbol))

    # 9. Momentum alignment (weight=2)
    items.append(_check_momentum(symbol, action))

    # 10. News sentiment (weight=1)
    items.append(_check_news_sentiment(symbol, action))

    # 11. Sector strength (weight=2)
    items.append(_check_sector_strength(symbol, action))

    # 12. Chart pattern / structure (weight=2)
    items.append(_check_chart_pattern(symbol, action))

    # 13. Intraday timing (weight=1)
    items.append(_check_market_timing())

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
