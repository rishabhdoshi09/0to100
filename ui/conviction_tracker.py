"""
Conviction Tracker — thesis validity score for open positions.
Tells you whether the original reason for entering a trade still holds.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import streamlit as st

IST = timezone(timedelta(hours=5, minutes=30))


def score_conviction(
    symbol: str,
    entry_price: float,
    archetype: str,
    regime_at_entry: str,
) -> dict:
    """
    Compute a thesis validity score for an open position.

    Returns:
        {
            "score": float,       # 0-100
            "verdict": str,       # "STRONG" / "INTACT" / "WEAKENING" / "BROKEN"
            "factors": list[str], # what's supporting / hurting the thesis
            "action": str,        # "Hold" / "Consider partial exit" / "Exit — thesis broken"
        }
    """
    score = 100.0
    factors: list[str] = []

    # ── 1. Fetch current price ────────────────────────────────────────────────
    current_price = 0.0
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.fast_info
        current_price = float(getattr(info, "last_price", 0) or 0)
    except Exception:
        pass

    # ── 2. Price vs entry ─────────────────────────────────────────────────────
    try:
        if entry_price > 0 and current_price > 0:
            price_chg_pct = (current_price - entry_price) / entry_price
            if price_chg_pct < -0.03:
                score -= 20
                factors.append("Price below entry by >3% ✗")
            elif price_chg_pct >= 0:
                factors.append(f"Price above entry (+{price_chg_pct*100:.1f}%) ✓")
            else:
                factors.append(f"Price slightly below entry ({price_chg_pct*100:.1f}%)")
    except Exception:
        pass

    # ── 3. Volume trend ───────────────────────────────────────────────────────
    rsi_value: float | None = None
    try:
        import yfinance as yf
        import numpy as np

        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="30d")
        if hist is not None and len(hist) >= 20:
            vol_series = hist["Volume"].astype(float)
            avg_20d = vol_series.iloc[-20:].mean()
            avg_3d = vol_series.iloc[-3:].mean()
            if avg_20d > 0:
                vol_ratio = avg_3d / avg_20d
                if vol_ratio < 0.7:
                    score -= 15
                    factors.append("Volume drying up (3d avg < 70% of 20d avg) ✗")
                else:
                    factors.append("Volume above average ✓")

            # RSI from close prices
            close = hist["Close"].astype(float)
            if len(close) >= 15:
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = (-delta).clip(lower=0)
                avg_gain = gain.rolling(14).mean().iloc[-1]
                avg_loss = loss.rolling(14).mean().iloc[-1]
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi_value = float(100 - 100 / (1 + rs))
                elif avg_gain > 0:
                    rsi_value = 100.0
                else:
                    rsi_value = 50.0
    except Exception:
        pass

    # ── 4. Regime match ───────────────────────────────────────────────────────
    try:
        from ui.regime_bar import get_regime
        current_regime = get_regime().get("regime", "UNKNOWN")
        bear_regimes = {"BEAR", "DISTRIBUTION", "BEARISH", "DOWNTREND"}
        if current_regime != regime_at_entry and current_regime.upper() in bear_regimes:
            score -= 25
            factors.append(f"Regime turned against ({regime_at_entry} → {current_regime}) ✗")
        else:
            factors.append(f"Regime still favorable ({current_regime}) ✓")
    except Exception:
        factors.append("Regime data unavailable")

    # ── 5. Days held ──────────────────────────────────────────────────────────
    try:
        days_held = st.session_state.get(f"conviction_entry_date_{symbol}", None)
        if days_held is not None:
            entry_date = datetime.fromisoformat(str(days_held))
            now = datetime.now(IST)
            if entry_date.tzinfo is None:
                entry_date = entry_date.replace(tzinfo=IST)
            days = (now - entry_date).days
            if days > 10:
                score -= 10
                factors.append(f"Setup taking too long ({days} days held) ✗")
        # If no date stored, skip penalty
    except Exception:
        pass

    # ── 6. RSI check ─────────────────────────────────────────────────────────
    try:
        if rsi_value is not None:
            if rsi_value < 40:
                score -= 20
                factors.append(f"Momentum lost (RSI {rsi_value:.0f}) ✗")
            else:
                factors.append(f"RSI {rsi_value:.0f} — healthy ✓")
    except Exception:
        pass

    score = max(0.0, min(100.0, score))

    # ── Verdict ───────────────────────────────────────────────────────────────
    if score >= 75:
        verdict = "STRONG"
        action = "Hold"
    elif score >= 50:
        verdict = "INTACT"
        action = "Hold with tight stop"
    elif score >= 25:
        verdict = "WEAKENING"
        action = "Consider partial exit"
    else:
        verdict = "BROKEN"
        action = "Exit — thesis broken"

    return {
        "score": score,
        "verdict": verdict,
        "factors": factors,
        "action": action,
        "current_price": current_price,
        "rsi": rsi_value,
    }


def render_conviction_panel(positions: list[dict]) -> None:
    """
    Render a compact conviction card for each open position.

    Each position dict should have at minimum:
        tradingsymbol, average_price
    Optional: archetype, regime_at_entry
    """
    if not positions:
        return

    st.markdown(
        "<div style='color:#8892a4;text-transform:uppercase;font-size:.72rem;"
        "letter-spacing:.07em;margin:1.2rem 0 .6rem'>Conviction Tracker</div>",
        unsafe_allow_html=True,
    )

    _VERDICT_COLORS = {
        "STRONG": "#00d4a0",
        "INTACT": "#00d4ff",
        "WEAKENING": "#f59e0b",
        "BROKEN": "#ff4466",
    }

    for pos in positions:
        sym = pos.get("tradingsymbol", pos.get("symbol", ""))
        entry_px = float(pos.get("average_price", pos.get("entry_price", 0)) or 0)
        archetype = pos.get("archetype", pos.get("setup", ""))
        regime_at_entry = pos.get("regime_at_entry", st.session_state.get("session_opening_regime", "UNKNOWN"))

        result = score_conviction(sym, entry_px, archetype, regime_at_entry)

        score = result["score"]
        verdict = result["verdict"]
        factors = result["factors"]
        action = result["action"]
        cur_px = result.get("current_price", 0.0)

        verdict_color = _VERDICT_COLORS.get(verdict, "#8892a4")

        # P&L % vs entry
        if entry_px > 0 and cur_px > 0:
            chg_pct = (cur_px - entry_px) / entry_px * 100
            chg_sign = "+" if chg_pct >= 0 else ""
            chg_color = "#00d4a0" if chg_pct >= 0 else "#ff4466"
            price_line = (
                f"Entry ₹{entry_px:,.2f} &nbsp; Current ₹{cur_px:,.2f} &nbsp; "
                f"<span style='color:{chg_color};font-weight:700'>"
                f"({chg_sign}{chg_pct:.1f}%)</span>"
            )
        elif entry_px > 0:
            price_line = f"Entry ₹{entry_px:,.2f}"
        else:
            price_line = ""

        # Progress bar (filled blocks)
        bar_filled = int(score / 10)
        bar_html = (
            f"<span style='color:{verdict_color};font-family:monospace;letter-spacing:1px'>"
            + "█" * bar_filled
            + "<span style='opacity:.25'>"
            + "░" * (10 - bar_filled)
            + "</span></span>"
        )

        # Factor list
        factor_html = "".join(
            f"<div style='font-size:.75rem;color:#8892a4;margin:.1rem 0'>&nbsp;• {f}</div>"
            for f in factors
        )

        stop_price = f"₹{entry_px:,.2f}" if entry_px > 0 else "your stop"

        card_html = (
            f"<div style='background:rgba(255,255,255,.03);"
            f"border:1px solid rgba(255,255,255,.08);border-left:3px solid {verdict_color};"
            f"border-radius:10px;padding:14px 18px;margin-bottom:10px'>"
            # Header row
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:baseline;margin-bottom:4px'>"
            f"<span style='color:#e8eaf0;font-size:1rem;font-weight:700;"
            f"font-family:JetBrains Mono,monospace'>{sym}</span>"
            f"<span style='font-size:.78rem;color:#8892a4'>{price_line}</span>"
            f"</div>"
            # Score row
            f"<div style='margin:.4rem 0'>"
            f"<span style='font-size:.8rem;color:#8892a4'>Conviction:&nbsp;</span>"
            f"{bar_html}&nbsp;"
            f"<span style='color:{verdict_color};font-weight:700;font-size:.88rem'>"
            f"{score:.0f}/100 — {verdict}</span>"
            f"</div>"
            # Factors
            f"{factor_html}"
            # Action
            f"<div style='margin-top:.5rem;font-size:.8rem;"
            f"color:{verdict_color};font-weight:600'>"
            f"Action: {action}"
            + (f" at {stop_price}" if "stop" in action.lower() else "")
            + f"</div>"
            f"</div>"
        )

        st.markdown(card_html, unsafe_allow_html=True)
