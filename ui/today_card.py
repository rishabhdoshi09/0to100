"""
Today in 30 Seconds — the one card every retail investor needs.

Answers three questions instantly:
  1. Is today a good day to trade?
  2. What should I watch?
  3. What should I avoid?

No jargon. Plain language. Cached 30 minutes.
"""
from __future__ import annotations

import os
from datetime import datetime

import streamlit as st


_MOOD_CONFIG = {
    "TRENDING_BULL": ("🟢", "GOOD DAY TO TRADE",  "#00d4a0", "Market is in a strong uptrend. Momentum is on your side."),
    "EXPANSION":     ("🟢", "GOOD DAY TO TRADE",  "#00d4a0", "Market is breaking out. Good for buying breakouts."),
    "CHOPPY":        ("🟡", "TRADE CAREFULLY",     "#f59e0b", "Market has no clear direction. Trade smaller, be patient."),
    "COMPRESSION":   ("🟡", "WAIT AND WATCH",      "#f59e0b", "Market is quiet. Wait for a clear signal before trading."),
    "DISTRIBUTION":  ("🔴", "PROTECT YOUR MONEY",  "#f97316", "Market may be topping. Tighten stops, don't add new trades."),
    "TRENDING_BEAR": ("🔴", "STAY IN CASH",        "#ff4b4b", "Market is falling. Cash is the best position right now."),
}

_VIX_NOTES = {
    (0, 12):   "Fear is very low — calm market.",
    (12, 16):  "Normal fear levels — ideal conditions.",
    (16, 20):  "Some fear — use smaller trade sizes.",
    (20, 25):  "High fear ⚡ — trade very small.",
    (25, 999): "Extreme fear 🔴 — experienced traders only.",
}


@st.cache_data(ttl=1800, show_spinner=False)
def _generate_brief(regime: str, vix: float, nifty: float, nifty_chg: float,
                    top_symbol: str, top_archetype: str, worst_sector: str,
                    opp_score: float, max_trades: int, size_pct: int) -> dict:
    """Generate plain-English today card via DeepSeek. Cached 30 min."""
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return _fallback_brief(regime, vix, nifty, nifty_chg, top_symbol, worst_sector, opp_score)

    import requests as _req
    mood_label = _MOOD_CONFIG.get(regime, ("⚪", "UNCERTAIN", "#8892a4", ""))[1]
    prompt = (
        f"You are a friendly trading coach speaking to a retail investor in India. "
        f"Give them their daily market brief in PLAIN, SIMPLE English. No jargon.\n\n"
        f"Today's data:\n"
        f"- Market: {regime} ({mood_label})\n"
        f"- Nifty 50: ₹{nifty:,.0f} ({nifty_chg:+.2f}% today)\n"
        f"- Fear index (VIX): {vix:.1f}\n"
        f"- Opportunity score: {opp_score:.0f}/100\n"
        f"- Best setup today: {top_symbol} ({top_archetype})\n"
        f"- Sector to avoid: {worst_sector}\n"
        f"- Today's rules: max {max_trades} trades, {size_pct}% of normal position size\n\n"
        f"Return JSON with these exact keys:\n"
        f'{{"watch": "one stock or sector to watch (10 words max)", '
        f'"avoid": "one thing to avoid today (10 words max)", '
        f'"do_today": "one clear action item (15 words max)", '
        f'"market_in_words": "describe the market in one casual sentence (20 words max)"}}'
    )
    try:
        resp = _req.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 200},
            timeout=12,
        )
        import json, re
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception:
        return _fallback_brief(regime, vix, nifty, nifty_chg, top_symbol, worst_sector, opp_score)


def _fallback_brief(regime, vix, nifty, nifty_chg, top_symbol, worst_sector, opp_score) -> dict:
    watch = top_symbol if top_symbol else "Wait for clear setups"
    avoid = f"{worst_sector} stocks" if worst_sector else "chasing momentum"
    do_today = "No new trades today" if opp_score < 40 else f"Look for entries near support"
    mood_desc = _MOOD_CONFIG.get(regime, ("", "", "", "Market conditions unclear."))[3]
    return {"watch": watch, "avoid": avoid, "do_today": do_today, "market_in_words": mood_desc}


def render_today_card(universe: list[str]) -> None:
    """The 30-second daily brief card. Renders at top of Dashboard."""
    try:
        from core.regime_engine import compute_regime
        from core.intelligence_hub import compute_opportunity_score, get_trading_rules
        from core.adaptive_engine import AdaptiveEngine

        regime_state = compute_regime()
        regime       = regime_state.market_regime
        vix          = regime_state.vix or 16.0
        nifty        = regime_state.nifty_price or 0.0
        nifty_chg    = regime_state.nifty_change_1d or 0.0
        worst_sector = (regime_state.lagging_sectors or ["—"])[0]
        leading      = (regime_state.leading_sectors or [])

        try:
            edge = AdaptiveEngine()
        except Exception:
            edge = None

        setups = st.session_state.get("jarvis_setups", [])
        opp    = compute_opportunity_score(regime_state, setups, edge)
        rules  = get_trading_rules(opp)

        top_symbol   = ""
        top_archetype = ""
        if setups:
            top = setups[0]
            top_symbol   = top.get("symbol", "")
            top_archetype = top.get("archetype", "").replace("_", " ").title()

        brief = _generate_brief(
            regime=regime, vix=vix, nifty=nifty, nifty_chg=nifty_chg,
            top_symbol=top_symbol, top_archetype=top_archetype,
            worst_sector=worst_sector, opp_score=opp.total,
            max_trades=rules.max_positions, size_pct=int(rules.size_multiplier * 100),
        )
    except Exception as e:
        regime = "CHOPPY"
        vix = 16.0
        nifty = 0.0
        nifty_chg = 0.0
        opp_total = 50.0
        rules_label = "Standard"
        rules_color = "#00d4ff"
        brief = {"watch": "—", "avoid": "—", "do_today": "Check back after market opens",
                 "market_in_words": "Market data loading..."}
        _render_card(regime, vix, nifty, nifty_chg, 50.0, "Standard", "#00d4ff", brief)
        return

    _render_card(
        regime, vix, nifty, nifty_chg,
        opp.total, rules.label, rules.color, brief,
    )


def _render_card(regime, vix, nifty, nifty_chg, opp_total, rules_label, rules_color, brief) -> None:
    emoji, mood, color, _ = _MOOD_CONFIG.get(regime, ("⚪", "UNCERTAIN", "#8892a4", ""))

    # VIX note
    vix_note = ""
    for (lo, hi), note in _VIX_NOTES.items():
        if lo <= vix < hi:
            vix_note = note
            break

    nifty_sign = "▲" if nifty_chg >= 0 else "▼"
    nifty_col  = "#00d4a0" if nifty_chg >= 0 else "#ff4b4b"
    now = datetime.now().strftime("%a, %d %b · %H:%M")

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0d1117,#161b22);"
        f"border:1px solid {color}44;border-radius:16px;padding:1.1rem 1.4rem;"
        f"margin-bottom:1rem'>"

        # ── Row 1: date + mood ──
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
        f"<div>"
        f"<div style='color:#4a5568;font-size:.65rem;font-family:JetBrains Mono,monospace'>{now}</div>"
        f"<div style='color:{color};font-size:1.1rem;font-weight:800;margin-top:.1rem'>"
        f"{emoji} {mood}</div>"
        f"<div style='color:#8892a4;font-size:.8rem;margin-top:.1rem'>{brief.get('market_in_words','')}</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='color:#c8cfe0;font-size:.85rem;font-weight:600'>"
        f"Nifty <span style='color:{nifty_col}'>{nifty_sign} {abs(nifty_chg):.1f}%</span></div>"
        f"<div style='color:#4a5568;font-size:.7rem'>VIX {vix:.1f} · {vix_note}</div>"
        f"<div style='color:{rules_color};font-size:.68rem;font-weight:600;margin-top:.2rem'>{rules_label}</div>"
        f"</div></div>"

        # ── Row 2: three action boxes ──
        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem;margin-top:.8rem'>"

        f"<div style='background:#0d1117;border:1px solid #00d4a044;border-radius:10px;padding:.55rem .7rem'>"
        f"<div style='color:#00d4a0;font-size:.6rem;font-weight:700;letter-spacing:.08em;margin-bottom:.2rem'>👁 WATCH TODAY</div>"
        f"<div style='color:#e8eaf0;font-size:.82rem;font-weight:600'>{brief.get('watch','—')}</div>"
        f"</div>"

        f"<div style='background:#0d1117;border:1px solid #ff4b4b44;border-radius:10px;padding:.55rem .7rem'>"
        f"<div style='color:#ff4b4b;font-size:.6rem;font-weight:700;letter-spacing:.08em;margin-bottom:.2rem'>⚠️ AVOID TODAY</div>"
        f"<div style='color:#e8eaf0;font-size:.82rem;font-weight:600'>{brief.get('avoid','—')}</div>"
        f"</div>"

        f"<div style='background:#0d1117;border:1px solid {rules_color}44;border-radius:10px;padding:.55rem .7rem'>"
        f"<div style='color:{rules_color};font-size:.6rem;font-weight:700;letter-spacing:.08em;margin-bottom:.2rem'>📋 YOUR RULE TODAY</div>"
        f"<div style='color:#e8eaf0;font-size:.82rem;font-weight:600'>{brief.get('do_today','—')}</div>"
        f"</div>"

        f"</div></div>",
        unsafe_allow_html=True,
    )
