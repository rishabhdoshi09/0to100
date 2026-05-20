"""
Plain English Mode — translates all quant jargon to human language.

A retail investor reading "LOW_VOL_COMPRESSION regime with OFFENSIVE rotation"
should instead see "Very calm market — big move may be coming. Growth stocks leading."

Toggle via sidebar. Affects any text passed through translate().
"""
from __future__ import annotations

import re
import streamlit as st

# ── Term translations ─────────────────────────────────────────────────────────

_TERMS: dict[str, str] = {
    # Market regimes
    "TRENDING_BULL":          "Strong Uptrend 📈",
    "TRENDING_BEAR":          "Falling Market 📉",
    "CHOPPY":                 "Going Sideways — No Clear Direction",
    "COMPRESSION":            "Quiet Market — Big Move Building",
    "EXPANSION":              "Market Breaking Out 🚀",
    "DISTRIBUTION":           "Market Topping Out ⚠️",

    # Volatility regimes
    "LOW_VOL_COMPRESSION":    "Very Calm — Good Time to Buy Setups",
    "NORMAL":                 "Normal Conditions",
    "TREND_VOLATILITY":       "Some Volatility — Trade Carefully",
    "ELEVATED":               "High Volatility ⚡ — Use Smaller Positions",
    "PANIC":                  "Market in Fear 🔴 — Stay in Cash",

    # Setup types
    "VCP_BREAKOUT":           "Coiling Breakout (Stock Building Energy)",
    "ACCUMULATION_BREAKOUT":  "Quiet Build-up Breakout",
    "MOMENTUM_PULLBACK":      "Strong Stock Dipping — Potential Buy Zone",
    "MEAN_REVERSION_BOUNCE":  "Beaten-Down Stock Recovering",
    "RANGE_BREAKOUT":         "Breaking Free from Trading Range",
    "BULL_FLAG":              "Brief Pause in Uptrend — May Continue",
    "BASE_BREAKOUT":          "Breaking Out of Long Base",

    # Tiers
    "ELITE_A_PLUS":           "⭐ Best Possible Setup",
    "WATCHLIST":              "Watch — Don't Trade Yet",

    # Breadth
    "STRONG":                 "Most Stocks Rising",
    "WEAK":                   "Most Stocks Falling",
    "NEUTRAL":                "Mixed — Some Up, Some Down",

    # Institutional
    "RISK_ON":                "Investors Taking Risk",
    "RISK_OFF":               "Investors Playing Safe",
    "ACCUMULATION":           "Institutions Quietly Buying",
    "DISTRIBUTION":           "Institutions Selling",
    "OFFENSIVE":              "Growth Sectors in Lead",
    "DEFENSIVE":              "Safety Sectors in Lead",
    "MIXED":                  "No Clear Sector Leadership",

    # Technical abbreviations
    "RSI":                    "RSI (Momentum Indicator 0-100)",
    "ATR":                    "ATR (Daily Range — used for stop placement)",
    "EMA":                    "EMA (Trend Line)",
    "SMA":                    "SMA (Average Price Line)",
    "PCR":                    "PCR (Market Sentiment Ratio)",
    "OI":                     "Open Interest (Active Contracts)",
    "VCP":                    "VCP (Stock Tightening Before Big Move)",
    "R-multiple":             "Risk-Reward Multiple",
    "R multiple":             "Risk-Reward Multiple",
    "quality_multiplier":     "Market Quality Adjustment",
    "regime_score":           "Market Health Score",
    "breadth_strength":       "Market Participation Score",
}

_REGIME_EXPLANATIONS: dict[str, str] = {
    "TRENDING_BULL":   "Market is in a healthy uptrend. Most stocks rising. Good time for breakout trades.",
    "TRENDING_BEAR":   "Market is falling. Preserve capital. Only trade if you're very sure.",
    "CHOPPY":          "Market has no clear direction. Trades are harder. Be patient, reduce trade size.",
    "COMPRESSION":     "Market is very quiet — often precedes a big move. Wait for breakout.",
    "EXPANSION":       "Market is breaking out. Momentum is strong. Good time to be aggressive.",
    "DISTRIBUTION":    "Smart money is quietly selling. The rally may be ending. Protect profits.",
}

_VIX_EXPLANATIONS: dict[tuple, str] = {
    (0, 12):    "VIX very low — market very calm. Good for trading, but be careful of complacency.",
    (12, 16):   "VIX normal — ideal trading conditions. Standard position sizes.",
    (16, 20):   "VIX elevated — some fear in market. Use slightly smaller positions.",
    (20, 25):   "VIX high — significant fear. Reduce position sizes by 30%.",
    (25, 999):  "VIX very high — market fearful. Experienced traders only. Small sizes or cash.",
}


def is_plain_english() -> bool:
    return st.session_state.get("plain_english_mode", False)


def translate(text: str) -> str:
    """Replace technical terms with plain English equivalents."""
    if not is_plain_english():
        return text
    result = text
    for term, plain in _TERMS.items():
        result = result.replace(term, plain)
    return result


def explain_regime(regime: str) -> str:
    """One sentence plain-English explanation of a regime."""
    return _REGIME_EXPLANATIONS.get(regime, "")


def explain_vix(vix: float) -> str:
    """Plain English VIX interpretation."""
    for (lo, hi), text in _VIX_EXPLANATIONS.items():
        if lo <= vix < hi:
            return text
    return ""


def render_mode_toggle() -> None:
    """Render the Plain English toggle in the sidebar."""
    current = st.session_state.get("plain_english_mode", False)
    new_val = st.toggle(
        "🗣️ Plain English Mode",
        value=current,
        key="plain_english_toggle",
        help="Translate all technical terms to simple language",
    )
    st.session_state["plain_english_mode"] = new_val
    if new_val:
        st.caption("All technical terms translated to plain language")


def regime_card_html(regime: str, score: float, vix: float, breadth: str) -> str:
    """Returns a simple, human-readable market status card HTML."""
    if not is_plain_english():
        return ""

    mood_map = {
        "TRENDING_BULL": ("🟢", "BULLISH", "#00d4a0"),
        "EXPANSION":     ("🟢", "BULLISH", "#00d4a0"),
        "CHOPPY":        ("🟡", "MIXED",   "#f59e0b"),
        "COMPRESSION":   ("🟡", "WAITING", "#f59e0b"),
        "DISTRIBUTION":  ("🔴", "CAUTION", "#f97316"),
        "TRENDING_BEAR": ("🔴", "BEARISH", "#ff4b4b"),
    }
    emoji, mood, color = mood_map.get(regime, ("⚪", "UNKNOWN", "#8892a4"))
    explanation = _REGIME_EXPLANATIONS.get(regime, "")
    vix_text = explain_vix(vix)
    breadth_text = {"STRONG": "Most stocks going up", "WEAK": "Most stocks falling", "NEUTRAL": "Mixed market"}.get(breadth, "")

    return (
        f"<div style='background:{color}11;border:1px solid {color}44;border-radius:12px;"
        f"padding:.8rem 1.1rem;margin:.5rem 0'>"
        f"<div style='font-size:1.1rem;font-weight:700;color:{color}'>{emoji} Market is {mood}</div>"
        f"<div style='color:#c8cfe0;font-size:.82rem;margin:.25rem 0'>{explanation}</div>"
        f"<div style='color:#8892a4;font-size:.75rem'>{vix_text}</div>"
        f"<div style='color:#8892a4;font-size:.75rem'>{breadth_text}</div>"
        f"</div>"
    )
