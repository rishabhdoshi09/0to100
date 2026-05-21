"""
Institutional Terminal — JARVIS Mission Control.

One screen, instant decision. Mission control, not spreadsheet.
Top-to-bottom layout:
  1. Threat Level Bar      — villain defense status
  2. Hero Card             — #1 trade, large and actionable
  3. Market Pulse          — 3 quick-read cards
  4. Setup List            — compact table of all setups
  5. Chart + Intelligence  — shown only when a setup is selected
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# ── Optional feature imports (graceful fallback if unavailable) ───────────────
try:
    from scan.relative_strength import compute_rs_score, get_rs_badge
    _HAS_RS = True
except Exception:
    _HAS_RS = False

try:
    from scan.setup_freshness import compute_freshness, freshness_badge
    _HAS_FRESHNESS = True
except Exception:
    _HAS_FRESHNESS = False

try:
    from scan.breakout_memory import get_false_breakout_rate
    _HAS_BREAKOUT_MEMORY = True
except Exception:
    _HAS_BREAKOUT_MEMORY = False

try:
    from core.market_timing import get_timing_score, render_timing_badge
    _HAS_TIMING = True
except Exception:
    _HAS_TIMING = False

try:
    from risk.smart_stop import format_stop_note, suggest_smart_stop
    _HAS_SMART_STOP = True
except Exception:
    _HAS_SMART_STOP = False

try:
    from data.earnings_calendar import render_earnings_chip, get_earnings_risk
    _HAS_EARNINGS = True
except Exception:
    _HAS_EARNINGS = False


# ── Plain-English translations ─────────────────────────────────────────────────
_ARCHETYPE_PLAIN = {
    "VCP_BREAKOUT":           "Coiling for breakout",
    "MOMENTUM_EXPANSION":     "Strong upward momentum",
    "EARLY_LEADER":           "Early stage leader",
    "ACCUMULATION_BREAKOUT":  "Institutions quietly buying",
    "EARNINGS_CONTINUATION":  "Strong after earnings",
    "FAILED_BREAKOUT":        "Breakout failed – avoid",
    "MEAN_REVERSION":         "Bouncing from low",
    "TREND_CONTINUATION":     "Uptrend continuing",
    "HIGH_TIGHT_FLAG":        "Tight flag — explosive potential",
}
_TIER_PLAIN = {
    "ELITE_A_PLUS": "Best Trade",
    "A":            "Strong Trade",
    "B":            "Good Trade",
    "WATCHLIST":    "Watch Only",
    "AVOID":        "Avoid",
}
_REGIME_PLAIN = {
    "TRENDING_BULL": "Rising strongly",
    "EXPANSION":     "Expanding moderately",
    "CHOPPY":        "Choppy — be cautious",
    "COMPRESSION":   "Consolidating — wait",
    "DISTRIBUTION":  "Smart money selling",
    "TRENDING_BEAR": "Falling — avoid new buys",
}
_REGIME_EMOJI = {
    "TRENDING_BULL": "🟢", "EXPANSION": "🟢",
    "CHOPPY": "🟡", "COMPRESSION": "🟡",
    "DISTRIBUTION": "🔴", "TRENDING_BEAR": "🔴",
}
_TIER_COLORS = {
    "ELITE_A_PLUS": ("#00d4a0", "#001a12"),
    "A":            ("#f59e0b", "#1a1200"),
    "B":            ("#60a5fa", "#001020"),
    "WATCHLIST":    ("#8892a4", "#111827"),
    "AVOID":        ("#4a5568", "#0d1117"),
}
_ARCHETYPE_EMOJI = {
    "VCP_BREAKOUT":           "🌀",
    "MOMENTUM_EXPANSION":     "🚀",
    "EARLY_LEADER":           "⭐",
    "ACCUMULATION_BREAKOUT":  "📦",
    "EARNINGS_CONTINUATION":  "📊",
    "FAILED_BREAKOUT":        "🔴",
    "MEAN_REVERSION":         "🔄",
    "TREND_CONTINUATION":     "📈",
    "HIGH_TIGHT_FLAG":        "🏴",
}
_REGIME_COLORS = {
    "TRENDING_BULL": "#00d4a0", "EXPANSION": "#00d4ff",
    "CHOPPY": "#f59e0b", "COMPRESSION": "#60a5fa",
    "DISTRIBUTION": "#fb923c", "TRENDING_BEAR": "#ff4b4b",
}


# ── Cached pipeline ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _run_pipeline(universe_key: str, top_n: int = 30) -> list[dict]:
    """Run 8-stage scan pipeline. Returns serialisable dicts (not dataclasses)."""
    try:
        universe = [s.strip() for s in universe_key.split(",") if s.strip()]
        from core.regime_engine import compute_regime
        regime = compute_regime()

        from scan.pipeline import ScanPipeline
        setups = ScanPipeline(max_workers=12, min_quality_score=40.0).run(
            universe, top_n=top_n, regime_state=regime,
            skip_liquidity_filter=len(universe) <= 30,
        )
        return [
            {
                "symbol":      s.symbol,
                "archetype":   s.archetype,
                "playbook_id": s.playbook_id,
                "price":       s.price,
                "pivot":       s.pivot_level,
                "stop":        s.stop_level,
                "risk_pct":    s.risk_pct,
                "tier":        s.quality_tier,
                "score":       s.quality_score,
                "adj_score":   s.regime_adjusted_score,
                "ev_r":        s.expected_value_r,
                "win_rate":    s.historical_win_rate,
                "reg_align":   s.regime_alignment,
                "fail_risk":   s.failure_risk,
                "evidence":    s.behavioral_evidence,
                "regime":      s.regime,
                "sector_leader": s.sector_leader,
                "pos_pct":     s.suggested_position_pct,
                "summary":     s.institutional_summary,
            }
            for s in setups
        ]
    except Exception as exc:
        st.session_state["pipeline_error"] = str(exc)
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_chart(symbol: str, timeframe: str) -> pd.DataFrame | None:
    tf_map = {"Daily": (250, "day"), "Weekly": (756, "week"), "Hourly": (30, "60minute")}
    days, kite_interval = tf_map.get(timeframe, (250, "day"))
    # 1. Kite Connect
    try:
        from data.kite_client import KiteClient
        from data.instruments import InstrumentManager
        from data.historical import HistoricalDataFetcher
        from datetime import date, timedelta
        kite = KiteClient()
        if kite.is_connected():
            fetcher = HistoricalDataFetcher(kite, InstrumentManager())
            to_dt = date.today().strftime("%Y-%m-%d")
            from_dt = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            df = fetcher.fetch(symbol, from_dt, to_dt, interval=kite_interval)
            if df is not None and not df.empty:
                if "close" not in df.columns and "Close" in df.columns:
                    df.columns = [c.lower() for c in df.columns]
                return df
    except Exception:
        pass
    # 2. yfinance fallback
    try:
        import yfinance as yf
        yf_map = {"Daily": ("250d", "1d"), "Weekly": ("3y", "1wk"), "Hourly": ("30d", "1h")}
        period, interval = yf_map.get(timeframe, ("250d", "1d"))
        df = yf.Ticker(f"{symbol}.NS").history(period=period, interval=interval)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception:
        pass
    # 3. Demo OHLCV
    try:
        from core.demo_data import make_demo_ohlcv
        rows = make_demo_ohlcv(symbol, bars=min(days, 250))
        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def _auto_verdict(symbol: str, archetype: str, tier: str, ev_r: float,
                  win_rate: float, regime: str) -> str:
    """Quick DeepSeek verdict — called automatically, no button needed."""
    from config import settings
    if not settings.deepseek_api_key:
        return ""
    import requests
    prompt = (
        f"In exactly 2 sentences: Should a retail trader take this setup?\n"
        f"Stock: {symbol} | Pattern: {archetype} | Tier: {tier} | "
        f"EV: {ev_r:+.1f}R | Win rate: {win_rate*100:.0f}% | Regime: {regime}\n"
        f"Sentence 1: Verdict (BUY AT PIVOT / WAIT / SKIP). Sentence 2: The single biggest reason."
    )
    try:
        resp = requests.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            json={"model": settings.deepseek_model, "messages": [
                {"role": "user", "content": prompt}
            ], "temperature": 0.0, "max_tokens": 80},
            timeout=10,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip() if "choices" in data else ""
    except Exception:
        return ""


@st.cache_data(ttl=1800, show_spinner=False)
def _deepseek_analyse(symbol: str, setup: dict | None, regime: dict) -> str:
    import os, requests
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return "⚠️ DEEPSEEK_API_KEY not configured."

    setup_block = ""
    if setup:
        ev_str = f"+{setup['ev_r']:.1f}R" if setup['ev_r'] >= 0 else f"{setup['ev_r']:.1f}R"
        setup_block = (
            f"\nSETUP:\n"
            f"  Archetype: {setup['archetype']}\n"
            f"  Quality Tier: {setup['tier']} ({setup['score']:.0f}/100)\n"
            f"  Pivot: ₹{setup['pivot']:,.0f}  Stop: ₹{setup['stop']:,.0f}\n"
            f"  Risk: {setup['risk_pct']:.1f}%  EV: {ev_str}\n"
            f"  Regime Alignment: {setup['reg_align']}\n"
            f"  Evidence: {'; '.join(setup.get('evidence', [])[:3])}"
        )

    prompt = (
        f"Institutional analysis for {symbol} (NSE India).\n\n"
        f"MARKET REGIME:\n"
        f"  {regime.get('market','?')} · VIX {regime.get('vix',16):.1f} ({regime.get('volatility','?')})\n"
        f"  Breadth: {regime.get('breadth','?')} · Risk: {regime.get('risk_mode','?')}\n"
        f"  Institutional: {regime.get('inst_activity','?')}\n"
        f"  Leaders: {', '.join(regime.get('leaders', ['N/A'])[:3])}"
        f"{setup_block}\n\n"
        f"Output EXACTLY this structure (fill in the blanks, no extra text):\n\n"
        f"SETUP QUALITY: [ELITE / STRONG / SELECTIVE / SKIP]\n\n"
        f"WHY IT MATTERS:\n"
        f"• [key bullish factor 1]\n"
        f"• [key bullish factor 2]\n"
        f"• [optional 3rd factor]\n\n"
        f"RISK:\n"
        f"• [primary risk]\n"
        f"• [secondary risk]\n\n"
        f"EXPECTANCY:\n"
        f"[one sentence on EV and best-case scenario]\n\n"
        f"VERDICT: [BUY AT PIVOT / WAIT FOR CONFIRMATION / SKIP] — [specific trigger]\n\n"
        f"Max 150 words total. Probabilistic language. No retail commentary."
    )

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Senior institutional equity analyst. Terse, probabilistic, no fluff. Terminal output only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 350,
            },
            timeout=25,
        )
        data = resp.json()
        if "choices" not in data:
            return f"API error: {data.get('error',{}).get('message','Unknown')}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Analysis unavailable: {e}"


# ── 1. THREAT LEVEL BAR ────────────────────────────────────────────────────────

def _render_threat_level_bar(universe: list[str]) -> None:
    """Full-width colored banner showing combined villain defense status."""
    threats: list[tuple[str, str]] = []  # (level: CLEAR|CAUTION|DANGER, reason)

    # Flash Crash Guard
    try:
        from risk.flash_crash_guard import is_flash_crash_active
        active, reason = is_flash_crash_active()
        if active:
            threats.append(("DANGER", f"Flash crash detected — {reason}"))
    except Exception:
        pass

    # Whipsaw Guard
    try:
        from risk.whipsaw_guard import compute_whipsaw_score
        report = compute_whipsaw_score()
        if report.score >= 70:
            threats.append(("DANGER", f"Severe whipsaw conditions — score {report.score:.0f}/100"))
        elif report.score >= 50:
            threats.append(("CAUTION", f"Choppy market — whipsaw score {report.score:.0f}/100"))
    except Exception:
        pass

    # Event Risk Scanner
    try:
        from risk.event_risk_scanner import get_event_risk
        ev_report = get_event_risk(universe)
        if ev_report.max_impact in ("EXTREME",):
            events = ", ".join(e["event"] for e in ev_report.today_events[:2])
            threats.append(("DANGER", f"Extreme event today — {events or ev_report.action}"))
        elif ev_report.max_impact in ("HIGH",):
            events = ", ".join(e["event"] for e in (ev_report.today_events + ev_report.tomorrow_events)[:2])
            threats.append(("CAUTION", f"High-impact event — {events or ev_report.action}"))
    except Exception:
        pass

    # Market Microstructure Timing
    if _HAS_TIMING:
        try:
            timing = get_timing_score()
            if timing["score"] < 40:
                threats.append(("CAUTION", f"⏰ Bad timing: {timing['reason']}"))
        except Exception:
            pass

    # Determine aggregate threat level
    if any(lvl == "DANGER" for lvl, _ in threats):
        level = "DANGER"
        reasons = [r for lvl, r in threats if lvl == "DANGER"]
        bg = "#2d0a0a"
        border = "#ff4b4b"
        icon = "🔴"
        text_col = "#ff4b4b"
        label = "DANGER — No new trades"
    elif any(lvl == "CAUTION" for lvl, _ in threats):
        level = "CAUTION"
        reasons = [r for lvl, r in threats if lvl == "CAUTION"]
        bg = "#1f1500"
        border = "#f59e0b"
        icon = "🟡"
        text_col = "#f59e0b"
        label = "CAUTION"
    else:
        level = "CLEAR"
        reasons = ["All defenses nominal — conditions suitable for trading"]
        bg = "#001a12"
        border = "#00d4a0"
        icon = "🟢"
        text_col = "#00d4a0"
        label = "ALL CLEAR"

    reason_str = " &nbsp;·&nbsp; ".join(reasons[:2])

    st.markdown(
        f"<div style='background:{bg};border:1px solid {border}44;"
        f"border-radius:10px;padding:10px 18px;margin-bottom:12px;"
        f"display:flex;align-items:center;justify-content:space-between;gap:12px'>"
        f"<div style='display:flex;align-items:center;gap:10px'>"
        f"<span style='font-size:1.1rem'>{icon}</span>"
        f"<span style='color:{text_col};font-size:.8rem;font-weight:800;"
        f"letter-spacing:.12em;text-transform:uppercase'>{label}</span>"
        f"<span style='color:#4a5568;font-size:.68rem'>|</span>"
        f"<span style='color:#c9d1e0;font-size:.72rem'>{reason_str}</span>"
        f"</div>"
        f"<span style='font-size:.6rem;color:#4a5568;white-space:nowrap'>THREAT MONITOR</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── 2. HERO CARD — THE #1 TRADE ───────────────────────────────────────────────

def _render_hero_card(best: dict, regime: dict) -> None:
    """Large, impressive card showing the single best setup."""
    symbol    = best["symbol"]
    score     = best["adj_score"]
    archetype = _ARCHETYPE_PLAIN.get(best["archetype"], best["archetype"].replace("_", " "))
    tier_plain = _TIER_PLAIN.get(best["tier"], best["tier"].replace("_", " "))
    win_rate  = best["win_rate"] * 100
    pivot     = best["pivot"]
    stop      = best["stop"]
    risk_pct  = best["risk_pct"]
    target_1r = pivot * (1 + risk_pct / 100)
    target_2r = pivot * (1 + risk_pct * 2 / 100)
    risk_amt  = pivot - stop
    rew_1     = target_1r - pivot
    rew_2     = target_2r - pivot
    regime_plain = _REGIME_PLAIN.get(regime.get("market", ""), "Unknown")

    # Border color based on confidence
    if score >= 75:
        border_col = "#00d4a0"
        badge_bg   = "#00d4a022"
        glow       = "0 0 24px #00d4a033"
    elif score >= 55:
        border_col = "#f59e0b"
        badge_bg   = "#f59e0b22"
        glow       = "0 0 20px #f59e0b22"
    else:
        border_col = "#4a5568"
        badge_bg   = "#4a556822"
        glow       = "none"

    emoji = _ARCHETYPE_EMOJI.get(best["archetype"], "⭐")

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0d1117 0%,#111827 100%);"
        f"border:1.5px solid {border_col};border-radius:14px;padding:20px 24px;"
        f"box-shadow:{glow};margin-bottom:14px'>"

        # Header row
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px'>"
        f"<div>"
        f"<div style='display:flex;align-items:center;gap:10px'>"
        f"<span style='font-size:1.6rem'>{emoji}</span>"
        f"<span style='color:#ffffff;font-size:1.5rem;font-weight:900;"
        f"letter-spacing:.04em;font-family:JetBrains Mono,monospace'>{symbol}</span>"
        f"<span style='background:{badge_bg};color:{border_col};font-size:.65rem;"
        f"padding:3px 10px;border-radius:6px;font-weight:800;border:1px solid {border_col}44'>"
        f"{tier_plain}</span>"
        f"</div>"
        f"<div style='color:#8892a4;font-size:.75rem;margin-top:4px'>"
        f"{archetype} &nbsp;·&nbsp; Market: {regime_plain}"
        f"</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='color:#4a5568;font-size:.6rem;text-transform:uppercase;letter-spacing:.1em'>Confidence</div>"
        f"<div style='color:{border_col};font-size:1.8rem;font-weight:900;"
        f"font-family:JetBrains Mono,monospace;line-height:1.1'>{score:.0f}"
        f"<span style='font-size:.9rem;color:#4a5568'>/100</span></div>"
        f"</div>"
        f"</div>"

        # Trade levels grid
        f"<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:14px 0'>"

        f"<div style='background:#0d1117;border:1px solid #1e293b;border-radius:8px;padding:10px;text-align:center'>"
        f"<div style='font-size:.58rem;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Buy Above</div>"
        f"<div style='font-size:1rem;color:#00d4ff;font-weight:800;font-family:JetBrains Mono,monospace'>₹{pivot:,.0f}</div>"
        f"</div>"

        f"<div style='background:#0d1117;border:1px solid #2d1010;border-radius:8px;padding:10px;text-align:center'>"
        f"<div style='font-size:.58rem;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Stop Loss</div>"
        f"<div style='font-size:1rem;color:#ff4b4b;font-weight:800;font-family:JetBrains Mono,monospace'>₹{stop:,.0f}</div>"
        f"</div>"

        f"<div style='background:#0d1117;border:1px solid #1e2d10;border-radius:8px;padding:10px;text-align:center'>"
        f"<div style='font-size:.58rem;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Target 1</div>"
        f"<div style='font-size:1rem;color:#f59e0b;font-weight:800;font-family:JetBrains Mono,monospace'>₹{target_1r:,.0f}</div>"
        f"</div>"

        f"<div style='background:#0d1117;border:1px solid #001a12;border-radius:8px;padding:10px;text-align:center'>"
        f"<div style='font-size:.58rem;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Target 2</div>"
        f"<div style='font-size:1rem;color:#00d4a0;font-weight:800;font-family:JetBrains Mono,monospace'>₹{target_2r:,.0f}</div>"
        f"</div>"

        f"<div style='background:#0d1117;border:1px solid #1e1a10;border-radius:8px;padding:10px;text-align:center'>"
        f"<div style='font-size:.58rem;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Win Rate</div>"
        f"<div style='font-size:1rem;color:#a78bfa;font-weight:800;font-family:JetBrains Mono,monospace'>{win_rate:.0f}%</div>"
        f"</div>"

        f"</div>"

        # Plain-English risk/reward
        f"<div style='background:#0d1117;border-radius:8px;padding:10px 14px;margin-bottom:14px;"
        f"border-left:3px solid {border_col}'>"
        f"<span style='font-size:.8rem;color:#c9d1e0'>💡 Risk "
        f"<span style='color:#ff4b4b;font-weight:700'>₹{risk_amt:,.0f}/share</span>"
        f" to potentially make "
        f"<span style='color:#00d4a0;font-weight:700'>₹{rew_1:,.0f}–₹{rew_2:,.0f}/share</span>"
        f"</span>"
        f"</div>"

        f"</div>",
        unsafe_allow_html=True,
    )

    # ── AUTO-VERDICT ──────────────────────────────────────────────────────────
    verdict = _auto_verdict(
        symbol=symbol,
        archetype=best["archetype"],
        tier=best["tier"],
        ev_r=best["ev_r"],
        win_rate=best["win_rate"],
        regime=regime.get("market", ""),
    )
    if verdict:
        verdict_upper = verdict.upper()
        if verdict_upper.startswith("BUY"):
            verdict_bg     = "#001a0e"
            verdict_border = "#00d4a0"
            verdict_col    = "#00d4a0"
        elif verdict_upper.startswith("WAIT"):
            verdict_bg     = "#1f1500"
            verdict_border = "#f59e0b"
            verdict_col    = "#f59e0b"
        else:  # SKIP or anything else
            verdict_bg     = "#1a0505"
            verdict_border = "#ff4b4b"
            verdict_col    = "#ff4b4b"

        st.markdown(
            f"<div style='background:{verdict_bg};border:1px solid {verdict_border}55;"
            f"border-radius:10px;padding:10px 16px;margin-bottom:10px'>"
            f"<div style='font-size:.58rem;color:#4a5568;text-transform:uppercase;"
            f"letter-spacing:.1em;margin-bottom:4px'>AUTO-VERDICT</div>"
            f"<div style='font-size:.78rem;color:{verdict_col};font-weight:700;"
            f"line-height:1.6'>{verdict}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Personal edge note below the verdict
        try:
            from core.learning_loop import get_personal_edge
            edge = get_personal_edge()
            archetype_edge = edge.get(best["archetype"], {})
            if archetype_edge.get("trades", 0) >= 5:
                personal_wr = archetype_edge["win_rate"] * 100
                vs_base     = archetype_edge["vs_baseline"] * 100
                color       = "#00d4a0" if vs_base > 0 else "#ff4b4b"
                arch_label  = best["archetype"].replace("_", " ")
                st.markdown(
                    f"<span style='color:{color};font-size:.65rem'>"
                    f"Your personal edge in {arch_label}: {personal_wr:.0f}% WR "
                    f"({vs_base:+.0f}% vs baseline)</span>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

    # Action buttons
    b1, b2, b3, b4 = st.columns([2, 2, 2, 2])
    with b1:
        if st.button("🤖 Ask JARVIS →", key="hero_analyse", use_container_width=True):
            ev_str = f"+{best['ev_r']:.1f}R" if best['ev_r'] >= 0 else f"{best['ev_r']:.1f}R"
            st.session_state["jarvis_prefill"] = (
                f"Institutional setup for {symbol}: {archetype} ({tier_plain}). "
                f"Confidence {score:.0f}/100. "
                f"Expected value {ev_str}. "
                f"Win rate {win_rate:.0f}%. "
                f"Entry ₹{pivot:,.0f}, Stop ₹{stop:,.0f}. "
                f"Should I take this trade?"
            )
            st.session_state["_nav_pending"] = "🤖  JARVIS"
            st.rerun()
    with b2:
        if st.button("📈 View Chart", key="hero_chart", use_container_width=True):
            st.session_state["iq2_selected"]    = symbol
            st.session_state["iq2_setup_data"]  = best
            st.session_state["iq2_show_chart"]  = True
            st.rerun()
    with b3:
        if st.button("⚡ Trade This", key="hero_trade", use_container_width=True, type="primary"):
            st.session_state["terminal_symbol"]  = symbol
            st.session_state["selected_symbol"]  = symbol
            st.session_state["_nav_pending"]     = "⚡  Terminal"
            st.rerun()
    with b4:
        if st.button("🔄 Re-scan", key="hero_rescan", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Timing badge — subtle row below action buttons
    if _HAS_TIMING:
        try:
            badge_html = render_timing_badge()
            if badge_html:
                st.markdown(
                    f"<div style='margin-top:6px;display:flex;align-items:center;gap:8px'>"
                    f"<span style='font-size:.58rem;color:#4a5568;text-transform:uppercase;"
                    f"letter-spacing:.08em'>Entry timing</span>{badge_html}</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass


# ── 3. MARKET PULSE ───────────────────────────────────────────────────────────

def _render_market_pulse(setups: list[dict], regime: dict) -> None:
    """Three quick-read cards: Market Mood, Your Edge, Active Scans."""
    market    = regime.get("market", "UNKNOWN")
    vix       = regime.get("vix", 16.0)
    breadth   = regime.get("breadth", "NEUTRAL")
    regime_plain = _REGIME_PLAIN.get(market, market.replace("_", " "))
    regime_emoji = _REGIME_EMOJI.get(market, "⚪")
    regime_col   = _REGIME_COLORS.get(market, "#8892a4")

    # Edge stats from setups
    if setups:
        avg_wr  = np.mean([s["win_rate"] for s in setups[:10]]) * 100
        avg_ev  = np.mean([s["ev_r"] for s in setups[:10]])
        top_tiers = [s for s in setups if s["tier"] in ("ELITE_A_PLUS", "A")]
    else:
        avg_wr, avg_ev, top_tiers = 0.0, 0.0, []

    # Sector breakdown
    sector_map: dict[str, int] = {}
    for s in setups:
        arch = s.get("archetype", "OTHER")
        sector_map[arch] = sector_map.get(arch, 0) + 1
    top_arch = max(sector_map, key=lambda k: sector_map[k]) if sector_map else "—"
    top_arch_plain = _ARCHETYPE_PLAIN.get(top_arch, top_arch.replace("_", " "))

    c1, c2, c3 = st.columns(3)

    def _card(col, icon: str, title: str, main_text: str, main_col: str,
              sub_text: str, border_col: str) -> None:
        with col:
            st.markdown(
                f"<div style='background:#0d1117;border:1px solid {border_col}44;"
                f"border-radius:10px;padding:14px 16px;height:100%'>"
                f"<div style='font-size:.62rem;color:#4a5568;text-transform:uppercase;"
                f"letter-spacing:.1em;margin-bottom:6px'>{icon} {title}</div>"
                f"<div style='font-size:1.1rem;color:{main_col};font-weight:800;"
                f"font-family:JetBrains Mono,monospace;margin-bottom:4px;line-height:1.2'>"
                f"{main_text}</div>"
                f"<div style='font-size:.68rem;color:#8892a4'>{sub_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _card(
        c1, regime_emoji, "Market Mood",
        regime_plain,
        regime_col,
        f"VIX {vix:.1f} · Breadth: {breadth.replace('_',' ')}",
        regime_col,
    )
    _card(
        c2, "🎯", "Your Edge Today",
        f"{avg_wr:.0f}% win rate",
        "#f59e0b" if avg_wr >= 55 else "#8892a4",
        f"Avg expected profit: +{avg_ev:.1f}× risk · {len(top_tiers)} elite setups",
        "#f59e0b",
    )
    _card(
        c3, "🔭", "Active Scans",
        f"{len(setups)} setups found",
        "#60a5fa",
        f"Top pattern: {top_arch_plain[:28]}",
        "#60a5fa",
    )

    st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)


# ── 4. SETUP LIST ─────────────────────────────────────────────────────────────

def _render_setup_list(setups: list[dict], universe_key: str) -> None:
    """Compact table of all setups with inline action buttons."""

    # Filter controls in one tight row
    fc1, fc2, fc3 = st.columns([3, 2, 1])
    with fc1:
        min_score = st.slider(
            "Min confidence", 40, 90, 55, 5,
            key="iq2_min", label_visibility="collapsed",
            help="Minimum confidence score to show"
        )
    with fc2:
        only_elite = st.toggle("Elite only", key="iq2_elite")
    with fc3:
        st.markdown(
            f"<div style='padding-top:6px;font-size:.7rem;color:#4a5568'>"
            f"{len(setups)} total</div>",
            unsafe_allow_html=True,
        )

    filtered = [
        s for s in setups
        if s["adj_score"] >= min_score
        and (not only_elite or s["tier"] == "ELITE_A_PLUS")
    ]

    if not filtered:
        st.markdown(
            "<div style='text-align:center;padding:2rem 1rem;background:#0d1117;"
            "border:1px solid #1e293b;border-radius:10px;color:#4a5568;font-size:.82rem'>"
            "No setups match filters — try lowering the minimum confidence score."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Table header
    st.markdown(
        "<div style='display:grid;grid-template-columns:28px 90px 1fr 90px 80px 100px 80px;"
        "gap:6px;padding:6px 10px;border-bottom:1px solid #1e293b;"
        "font-size:.58rem;color:#4a5568;text-transform:uppercase;letter-spacing:.08em;"
        "font-weight:700'>"
        "<span>#</span><span>Symbol</span><span>Pattern</span>"
        "<span style='text-align:right'>Price</span>"
        "<span style='text-align:center'>Confidence</span>"
        "<span style='text-align:center'>Expected Profit</span>"
        "<span style='text-align:center'>Action</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    for rank, s in enumerate(filtered[:25], 1):
        sym       = s["symbol"]
        score     = s["adj_score"]
        ev_r      = s["ev_r"]
        price     = s["price"]
        pattern   = _ARCHETYPE_PLAIN.get(s["archetype"], s["archetype"].replace("_", " "))
        emoji_pat = _ARCHETYPE_EMOJI.get(s["archetype"], "◆")
        sl_badge  = " 🏆" if s.get("sector_leader") else ""

        # ── RS badge (next to symbol name) ───────────────────────────────────
        rs_badge_html = ""
        if _HAS_RS:
            try:
                rs_score = compute_rs_score(sym, pd.DataFrame())  # score from pipeline cache
                rs_badge_html = "&nbsp;" + get_rs_badge(rs_score)
            except Exception:
                pass

        # ── Freshness badge (next to archetype) ───────────────────────────────
        freshness_badge_html = ""
        if _HAS_FRESHNESS:
            try:
                # Use a minimal proxy df — freshness badge is best-effort
                freshness_result = compute_freshness(sym, s.get("archetype", "VCP_BREAKOUT"), pd.DataFrame())
                freshness_badge_html = "&nbsp;" + freshness_badge(freshness_result)
            except Exception:
                pass

        # ── Breakout memory warning ───────────────────────────────────────────
        breakout_warning_html = ""
        if _HAS_BREAKOUT_MEMORY:
            try:
                bk_stats = get_false_breakout_rate(sym)
                if bk_stats.get("warning"):
                    breakout_warning_html = (
                        f"<div style='font-size:.58rem;color:#ff4b4b;margin-top:2px;"
                        f"font-weight:600'>⚠ {bk_stats['warning']}</div>"
                    )
            except Exception:
                pass

        # Confidence color
        if score >= 70:
            conf_col = "#00d4a0"
            conf_bg  = "#00d4a015"
        elif score >= 50:
            conf_col = "#f59e0b"
            conf_bg  = "#f59e0b15"
        else:
            conf_col = "#ff4b4b"
            conf_bg  = "#ff4b4b15"

        ev_col = "#00d4a0" if ev_r > 0 else "#ff4b4b"
        ev_str = f"+{ev_r:.1f}×" if ev_r >= 0 else f"{ev_r:.1f}×"

        selected = st.session_state.get("iq2_selected") == sym
        row_bg   = "#131c2e" if selected else "#0d1117"
        row_border = "#00d4ff44" if selected else "#1e293b"

        st.markdown(
            f"<div style='display:grid;grid-template-columns:28px 90px 1fr 90px 80px 100px 80px;"
            f"gap:6px;padding:7px 10px;background:{row_bg};"
            f"border:1px solid {row_border};border-radius:8px;margin-bottom:3px;"
            f"align-items:center'>"

            f"<span style='font-size:.65rem;color:#4a5568;font-weight:700'>{rank}</span>"

            f"<div>"
            f"<span style='font-size:.82rem;color:#ffffff;font-weight:800;"
            f"font-family:JetBrains Mono,monospace'>{sym}{sl_badge}</span>"
            f"{rs_badge_html}"
            f"{breakout_warning_html}"
            f"</div>"

            f"<span style='font-size:.68rem;color:#8892a4'>"
            f"{emoji_pat} {pattern[:30]}{freshness_badge_html}</span>"

            f"<span style='font-size:.75rem;color:#c9d1e0;font-family:JetBrains Mono,monospace;"
            f"text-align:right'>₹{price:,.0f}</span>"

            f"<div style='background:{conf_bg};border:1px solid {conf_col}33;"
            f"border-radius:5px;padding:2px 6px;text-align:center'>"
            f"<span style='font-size:.72rem;color:{conf_col};font-weight:800'>{score:.0f}</span>"
            f"</div>"

            f"<div style='text-align:center'>"
            f"<span style='font-size:.75rem;color:{ev_col};font-weight:700'>{ev_str} risk</span>"
            f"</div>"

            f"<div style='text-align:center'></div>"  # placeholder for button

            f"</div>",
            unsafe_allow_html=True,
        )

        # Button overlay — Streamlit buttons can't go inside HTML grid, so we
        # place them right after each row using negative margin trick
        btn_col = st.columns([28, 90, 200, 90, 80, 100, 80])[6]
        # Actually render a small button using native streamlit
        with st.container():
            row_cols = st.columns([28+90+200+90+80+100+6, 80])
            with row_cols[1]:
                if st.button("Select", key=f"sel_{sym}_{rank}", use_container_width=True):
                    st.session_state["iq2_selected"]   = sym
                    st.session_state["iq2_setup_data"] = s
                    st.session_state["iq2_show_chart"] = True
                    st.rerun()

    if len(filtered) > 25:
        st.caption(f"Showing top 25 of {len(filtered)} setups")


# ── 5. CHART PANEL ────────────────────────────────────────────────────────────

def _render_chart_workspace(symbol: str, setup: dict | None, regime: dict) -> None:
    """Candlestick chart with pivot/stop levels. Clean, no clutter."""
    import plotly.graph_objects as go

    reg_col = _REGIME_COLORS.get(regime.get("market", ""), "#00d4ff")
    archetype_plain = _ARCHETYPE_PLAIN.get(setup.get("archetype", ""), "") if setup else ""

    # Header
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:8px'>"
        f"<span style='color:{reg_col};font-size:.9rem;font-weight:800;"
        f"font-family:JetBrains Mono,monospace;letter-spacing:.06em'>📊 {symbol}</span>"
        + (f"<span style='font-size:.65rem;color:#4a5568'>{archetype_plain}</span>" if archetype_plain else "") +
        f"</div>",
        unsafe_allow_html=True,
    )

    tf_col, ov1_col, ov2_col, close_col = st.columns([3, 2, 2, 1])
    with tf_col:
        tf = st.radio("TF", ["Daily", "Weekly", "Hourly"], horizontal=True,
                      key="iq2_tf", label_visibility="collapsed")
    with ov1_col:
        show_sma = st.checkbox("SMA 50/200", value=True, key="iq2_sma")
    with ov2_col:
        show_ema = st.checkbox("EMA 10/21", value=False, key="iq2_ema")
    with close_col:
        if st.button("✕", key="close_chart", help="Close chart"):
            st.session_state["iq2_show_chart"] = False
            st.rerun()

    df = _fetch_chart(symbol, tf)
    if df is None or df.empty:
        st.info(f"No chart data for {symbol}. Try connecting to Kite or check the symbol.")
        return

    close = df["close"]
    n     = len(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=close,
        name=symbol,
        increasing_line_color="#00d4a0", decreasing_line_color="#ff4b4b",
        increasing_fillcolor="rgba(0,212,160,.25)", decreasing_fillcolor="rgba(255,75,75,.20)",
        line=dict(width=1),
    ))

    if show_sma and n >= 50:
        fig.add_trace(go.Scatter(x=df.index, y=close.rolling(50).mean(), name="SMA50",
                                  line=dict(color="#f59e0b", width=1.2, dash="dot")))
    if show_sma and n >= 200:
        fig.add_trace(go.Scatter(x=df.index, y=close.rolling(200).mean(), name="SMA200",
                                  line=dict(color="#8892a4", width=1, dash="dash")))
    if show_ema and n >= 21:
        fig.add_trace(go.Scatter(x=df.index, y=close.ewm(span=10).mean(), name="EMA10",
                                  line=dict(color="#a78bfa", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=close.ewm(span=21).mean(), name="EMA21",
                                  line=dict(color="#60a5fa", width=1)))

    if setup:
        pivot = setup.get("pivot", 0)
        stop  = setup.get("stop", 0)
        if pivot > 0:
            fig.add_hline(y=pivot, line_color="#00d4ff", line_dash="dash", line_width=1.5,
                          annotation_text=f"Entry ₹{pivot:,.0f}",
                          annotation_font_color="#00d4ff",
                          annotation_position="right")
        if stop > 0:
            fig.add_hline(y=stop, line_color="#ff4b4b", line_dash="dot", line_width=1,
                          annotation_text=f"Stop ₹{stop:,.0f}",
                          annotation_font_color="#ff4b4b",
                          annotation_position="right")
            if pivot > stop > 0:
                fig.add_hrect(y0=stop, y1=pivot, fillcolor="rgba(255,75,75,.04)", line_width=0)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=420, margin=dict(l=10, r=80, t=5, b=5),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, color="#4a5568", showspikes=True),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#4a5568", showspikes=True),
        legend=dict(orientation="h", y=1.01, x=0, font=dict(size=9, color="#8892a4"),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"iq2_c_{symbol}_{tf}")

    if "volume" in df.columns:
        vols  = df["volume"]
        vcols = ["rgba(0,212,160,.35)" if close.iloc[i] >= df["open"].iloc[i]
                 else "rgba(255,75,75,.3)" for i in range(n)]
        vfig  = go.Figure(go.Bar(x=df.index, y=vols, marker_color=vcols, name="Vol"))
        vfig.add_trace(go.Scatter(x=df.index, y=vols.rolling(20).mean(),
                                   line=dict(color="#f59e0b", width=1)))
        vfig.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=90, margin=dict(l=10, r=80, t=0, b=5), showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
        )
        st.plotly_chart(vfig, use_container_width=True, key=f"iq2_v_{symbol}_{tf}")


# ── 5. INTELLIGENCE PANEL ────────────────────────────────────────────────────

def _render_intelligence_panel(symbol: str, setup: dict | None, regime: dict) -> None:
    """AI analysis in plain English."""
    st.markdown(
        "<div style='background:#0d1117;border:1px solid rgba(167,139,250,.2);"
        "border-radius:10px;padding:12px 14px;margin-bottom:10px'>"
        "<div style='color:#a78bfa;font-size:.72rem;font-weight:800;"
        "letter-spacing:.1em;text-transform:uppercase'>🧠 AI Intelligence</div>"
        "<div style='color:#4a5568;font-size:.63rem;margin-top:2px'>"
        "DeepSeek institutional analysis — click to generate</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Setup quick stats
    if setup:
        price  = setup.get("price", 0)
        pivot  = setup.get("pivot", price * 1.02)
        stop   = setup.get("stop", price * 0.95)
        risk   = setup.get("risk_pct", 5.0)
        pos    = setup.get("pos_pct", 2.0)
        t1     = pivot * (1 + risk / 100)
        t2     = pivot * (1 + risk * 2 / 100)
        ev_r   = setup["ev_r"]
        wr_pct = setup["win_rate"] * 100
        ev_col = "#00d4a0" if ev_r > 0 else "#ff4b4b"
        ev_str = f"+{ev_r:.1f}R" if ev_r >= 0 else f"{ev_r:.1f}R"
        fail_col = "#ff4b4b" if setup["fail_risk"] == "HIGH" else \
                   "#f59e0b" if setup["fail_risk"] == "MODERATE" else "#00d4a0"

        st.markdown(
            f"<div style='background:#111827;border:1px solid #1e293b;border-radius:8px;"
            f"padding:10px 12px;margin-bottom:10px;font-family:JetBrains Mono,monospace'>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px'>"

            f"<div style='background:#0d1117;border-radius:5px;padding:6px 8px'>"
            f"<div style='font-size:.55rem;color:#4a5568'>Expected Profit</div>"
            f"<div style='font-size:.85rem;color:{ev_col};font-weight:800'>{ev_str}</div>"
            f"</div>"

            f"<div style='background:#0d1117;border-radius:5px;padding:6px 8px'>"
            f"<div style='font-size:.55rem;color:#4a5568'>Win Rate</div>"
            f"<div style='font-size:.85rem;color:#f59e0b;font-weight:800'>{wr_pct:.0f}%</div>"
            f"</div>"

            f"<div style='background:#0d1117;border-radius:5px;padding:6px 8px'>"
            f"<div style='font-size:.55rem;color:#4a5568'>Entry / Stop</div>"
            f"<div style='font-size:.75rem;color:#c9d1e0;font-weight:700'>"
            f"₹{pivot:,.0f} / <span style='color:#ff4b4b'>₹{stop:,.0f}</span></div>"
            f"</div>"

            f"<div style='background:#0d1117;border-radius:5px;padding:6px 8px'>"
            f"<div style='font-size:.55rem;color:#4a5568'>Targets</div>"
            f"<div style='font-size:.72rem;color:#00d4a0;font-weight:700'>"
            f"₹{t1:,.0f} / ₹{t2:,.0f}</div>"
            f"</div>"

            f"</div>"

            f"<div style='display:flex;justify-content:space-between;padding-top:4px;"
            f"border-top:1px solid #1e293b;font-size:.63rem'>"
            f"<span style='color:#4a5568'>Risk: {risk:.1f}% &nbsp;·&nbsp; "
            f"Pos: {pos:.1f}%</span>"
            f"<span>Fail risk: <span style='color:{fail_col};font-weight:700'>"
            f"{setup['fail_risk']}</span></span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Evidence bullets
        evidence = setup.get("evidence", [])
        if evidence:
            bullets = "".join(
                f"<div style='font-size:.67rem;color:#8892a4;padding:2px 0'>• {e}</div>"
                for e in evidence[:4]
            )
            st.markdown(
                f"<div style='margin-bottom:8px'>"
                f"<div style='font-size:.58rem;color:#4a5568;text-transform:uppercase;"
                f"letter-spacing:.08em;margin-bottom:4px'>Evidence</div>"
                f"{bullets}</div>",
                unsafe_allow_html=True,
            )

        # Smart stop note
        if _HAS_SMART_STOP:
            try:
                df_chart = _fetch_chart(symbol, "Daily")
                smart_result = suggest_smart_stop(
                    symbol=symbol,
                    df=df_chart,
                    entry_price=float(pivot),
                    archetype=setup.get("archetype", "VCP_BREAKOUT"),
                )
                stop_note = format_stop_note(smart_result)
                st.markdown(
                    f"<div style='font-size:.65rem;color:#8892a4;margin-bottom:6px;"
                    f"padding:4px 8px;background:#111827;border-radius:5px;"
                    f"border-left:2px solid #ff4b4b44'>🛡 {stop_note}</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        # Earnings proximity chip
        if _HAS_EARNINGS:
            try:
                chip_html = render_earnings_chip(symbol)
                if chip_html:
                    st.markdown(
                        f"<div style='margin-bottom:8px'>{chip_html}</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

    # AI Analysis button + output
    cache_key = f"iq2_analysis_{symbol}"

    # Auto-trigger if hero card "Analyse with AI" was clicked
    if st.session_state.pop("iq2_trigger_ai", False):
        with st.spinner("DeepSeek reasoning…"):
            analysis = _deepseek_analyse(symbol, setup, regime)
        st.session_state[cache_key] = analysis

    col_ai, col_jarvis = st.columns(2)
    with col_ai:
        if st.button("🧠 Analyse with AI", key=f"iq2_btn_{symbol}", use_container_width=True):
            with st.spinner("DeepSeek reasoning…"):
                analysis = _deepseek_analyse(symbol, setup, regime)
            st.session_state[cache_key] = analysis
    with col_jarvis:
        if st.button("🤖 Ask JARVIS →", key=f"iq2_jarvis_{symbol}", use_container_width=True):
            if setup:
                ev_r_val = setup.get("ev_r", 0.0)
                wr_val   = setup.get("win_rate", 0.0) * 100
                tier_v   = _TIER_PLAIN.get(setup.get("tier", ""), setup.get("tier", ""))
                arch_v   = _ARCHETYPE_PLAIN.get(setup.get("archetype", ""), setup.get("archetype", ""))
                ev_s     = f"+{ev_r_val:.1f}R" if ev_r_val >= 0 else f"{ev_r_val:.1f}R"
                jarvis_q = (
                    f"Institutional setup for {symbol}: {arch_v} ({tier_v}). "
                    f"Expected value {ev_s}. "
                    f"Win rate {wr_val:.0f}%. "
                    f"Should I take this trade?"
                )
            else:
                jarvis_q = f"Analyse {symbol} for me. Is it a good trade right now?"
            st.session_state["jarvis_prefill"] = jarvis_q
            st.session_state["_nav_pending"]   = "🤖  JARVIS"
            st.rerun()

    analysis = st.session_state.get(cache_key)
    if analysis:
        st.markdown(
            f"<div style='background:#111827;border:1px solid rgba(167,139,250,.2);"
            f"border-radius:8px;padding:12px 14px;font-size:.72rem;color:#c9d1e0;"
            f"line-height:1.7;font-family:JetBrains Mono,monospace;white-space:pre-wrap'>"
            f"{analysis}</div>",
            unsafe_allow_html=True,
        )
    elif not setup:
        st.markdown(
            "<div style='color:#4a5568;font-size:.72rem;text-align:center;padding:20px'>"
            "Select a setup from the list to see AI analysis.</div>",
            unsafe_allow_html=True,
        )

    # Playbooks
    st.markdown(
        "<div style='margin-top:12px;font-size:.62rem;color:#4a5568;"
        "text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>"
        "Today's Playbooks</div>",
        unsafe_allow_html=True,
    )
    try:
        from playbooks import get_playbooks_for_regime
        pbs = get_playbooks_for_regime(
            regime.get("market", "CHOPPY"),
            regime.get("volatility", "NORMAL"),
            regime.get("breadth", "NEUTRAL"),
        )[:3]
        for pb in pbs:
            dot = "🟢" if pb.regime_aligned else "🟡"
            ev  = f"{pb.baseline_expectancy*100:+.1f}%"
            wr  = f"{pb.baseline_win_rate*100:.0f}%"
            st.markdown(
                f"<div style='background:#111827;border:1px solid rgba(167,139,250,.1);"
                f"border-radius:6px;padding:7px 10px;margin-bottom:4px'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:#a78bfa;font-size:.7rem;font-weight:700'>"
                f"{dot} {pb.emoji} {pb.name}</span>"
                f"<span style='font-size:.6rem;color:#4a5568'>EV {ev} · WR {wr}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("Playbook data unavailable")


# ── Main Entry Point ───────────────────────────────────────────────────────────

def render_institutional_terminal(universe: list[str]) -> None:
    from ui.regime_bar import render_regime_bar, get_regime

    # Persistent regime bar at top
    render_regime_bar()
    try:
        from ui.ironlock_widget import render_ironlock_status
        render_ironlock_status()
    except Exception:
        pass
    regime = get_regime()

    # ── Page Title ──────────────────────────────────────────────────────────────
    title_col, search_col = st.columns([5, 1])
    with title_col:
        st.markdown(
            "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
            "font-size:1.15rem;letter-spacing:.12em;margin:0 0 2px;text-transform:uppercase'>"
            "⚡ Mission Control</h2>"
            "<span style='color:#4a5568;font-size:.65rem'>"
            "8-Stage Pipeline · Regime-Aware · Expectancy-Based · "
            "Real-time Threat Detection</span>",
            unsafe_allow_html=True,
        )
    with search_col:
        manual = st.text_input(
            "Jump to symbol", value="", placeholder="e.g. INFY",
            key="iq2_manual", label_visibility="collapsed"
        )
        if manual.strip().upper():
            st.session_state["iq2_selected"]   = manual.strip().upper()
            st.session_state["iq2_setup_data"] = None
            st.session_state["iq2_show_chart"] = True

    universe_key = ",".join(sorted(universe))

    # ── 1. THREAT LEVEL BAR ─────────────────────────────────────────────────────
    _render_threat_level_bar(universe)

    # ── Load pipeline data ──────────────────────────────────────────────────────
    with st.spinner("Running 8-stage scan pipeline…"):
        setups = _run_pipeline(universe_key, top_n=30)

    if err := st.session_state.pop("pipeline_error", None):
        st.warning(f"Pipeline error: {err}")

    # ── 2. HERO CARD ────────────────────────────────────────────────────────────
    if setups:
        best = setups[0]
        _render_hero_card(best, regime)
    else:
        st.info("No setups found. Check universe, market connectivity, or lower min score.")

    # ── 3. MARKET PULSE ─────────────────────────────────────────────────────────
    _render_market_pulse(setups, regime)

    # ── Divider ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='border-top:1px solid #1e293b;margin:6px 0 12px'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#8892a4;font-size:.68rem;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px'>"
        "⚡ All Setups</div>",
        unsafe_allow_html=True,
    )

    # ── 4. SETUP LIST ───────────────────────────────────────────────────────────
    _render_setup_list(setups, universe_key)

    # ── 5. CHART + INTELLIGENCE (only when selected) ────────────────────────────
    selected   = st.session_state.get("iq2_selected")
    setup_data = st.session_state.get("iq2_setup_data")
    show_chart = st.session_state.get("iq2_show_chart", False)

    if selected and show_chart:
        st.markdown(
            "<div style='border-top:1px solid #1e293b;margin:14px 0 12px'></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#00d4ff;font-size:.72rem;font-weight:800;"
            f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>"
            f"📊 Analysis — {selected}</div>",
            unsafe_allow_html=True,
        )

        chart_col, intel_col = st.columns([6, 4])
        with chart_col:
            _render_chart_workspace(selected, setup_data, regime)
        with intel_col:
            _render_intelligence_panel(selected, setup_data, regime)
