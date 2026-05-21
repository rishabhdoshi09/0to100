"""
sitrep.py — Pre-Market Situation Report (SITREP) dashboard.

Auto-generates every morning during the pre-market window (08:30–09:30 IST).
Provides a full-screen intelligence briefing before the trading session opens.

Public API:
    should_show_sitrep() -> bool
    render_sitrep()
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Module-level regime cache (5-minute TTL, no Streamlit dependency)
# ---------------------------------------------------------------------------
_REGIME_CACHE: dict = {}
_REGIME_CACHE_TTL = 5 * 60  # 5 minutes


def _get_ist_now() -> datetime:
    try:
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        return datetime.now(ist)
    except Exception:
        # Fallback: UTC + 5:30
        from datetime import timezone
        utc_now = datetime.now(timezone.utc)
        return utc_now.replace(tzinfo=None) + timedelta(hours=5, minutes=30)


# ---------------------------------------------------------------------------
# Playbook plain-English mapping
# ---------------------------------------------------------------------------
_PLAYBOOK_PLAIN: dict[str, str] = {
    "VCP_BREAKOUT": "Tight coil breakouts",
    "MOMENTUM_CONTINUATION": "Strong stocks getting stronger",
    "EARNINGS_CONTINUATION": "Post-earnings winners",
    "MEAN_REVERSION": "Oversold bounces",
    "HIGH_TIGHT_FLAG": "Explosive flag patterns",
    "ACCUMULATION_BREAKOUT": "Institutional accumulation breakouts",
    "MOMENTUM_EXPANSION": "Strong stocks getting stronger",
    "EARLY_LEADER": "Early-stage leaders before the crowd",
    "SECTOR_BREAKOUT": "Sector-wide breakout momentum",
    "RANGE_FADE": "Fade the edges of choppy ranges",
    "DEFENSIVE_POSITIONING": "Shift to defensive holdings",
    "CASH_CONSERVATION": "Raise cash, limit new buys",
    "SECTOR_MOMENTUM": "Ride leading sector rotations",
    "DEFENSIVE_ROTATION": "Rotate into defensive sectors",
    "BROAD_MARKET_LONG": "Broad-based market uptrend plays",
    "FAILED_BREAKOUT_REVERSAL": "Short failed breakout traps",
    "VOLATILITY_MEAN_REVERSION": "Mean-revert in panic spikes",
}

_AVOID_PLAIN: dict[str, str] = {
    "MEAN_REVERSION": "Oversold bounces (trend too strong)",
    "COUNTER_TREND_SHORT": "Shorting into an uptrend",
    "BREAKOUT_BUY": "Buying breakouts (high failure rate now)",
    "MOMENTUM_EXPANSION": "Chasing momentum (conditions weak)",
    "MOMENTUM_CHASING": "Chasing recent movers",
    "LATE_MOMENTUM": "Late-stage momentum entries",
    "HIGH_BETA_MOMENTUM": "High-beta names in defensive rotation",
    "VCP_BREAKOUT": "Tight coil breakouts (regime unfavorable)",
    "EARLY_LEADER": "Early leaders (distribution risk)",
}


def _plain(pid: str, mapping: dict[str, str]) -> str:
    return mapping.get(pid, pid.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_show_sitrep() -> bool:
    """
    Returns True if:
    - Current IST time is between 08:30 and 09:30 (pre-market window)
    - OR st.session_state.get("force_sitrep") is True
    """
    if st.session_state.get("force_sitrep"):
        return True
    try:
        now_ist = _get_ist_now()
        t = now_ist.time()
        from datetime import time as _time
        return _time(8, 30) <= t <= _time(9, 30)
    except Exception:
        return False


def _cached_regime():
    """Return regime with 5-minute module-level cache."""
    now = time.time()
    cached = _REGIME_CACHE.get("state")
    cached_ts = _REGIME_CACHE.get("ts", 0.0)
    if cached is not None and (now - cached_ts) < _REGIME_CACHE_TTL:
        return cached
    try:
        from core.regime_engine import compute_regime
        state = compute_regime()
        _REGIME_CACHE["state"] = state
        _REGIME_CACHE["ts"] = now
        return state
    except Exception:
        return None


def _vix_interpretation(vix: float) -> str:
    if vix <= 0:
        return "VIX unavailable"
    if vix < 14:
        return f"VIX {vix:.1f} = low fear, ideal breakout conditions"
    if vix < 18:
        return f"VIX {vix:.1f} = normal, standard position sizes"
    if vix < 24:
        return f"VIX {vix:.1f} = elevated, use smaller positions"
    if vix < 30:
        return f"VIX {vix:.1f} = high fear, reduce exposure significantly"
    return f"VIX {vix:.1f} = panic, consider cash / hedges only"


def _regime_color(regime: str) -> str:
    if regime in ("TRENDING_BULL", "EXPANSION"):
        return "green"
    if regime in ("TRENDING_BEAR", "DISTRIBUTION"):
        return "red"
    return "#f59e0b"  # yellow/amber for choppy/compression


def _inst_color(inst: str) -> str:
    if inst in ("ACCUMULATION", "RISK_ON"):
        return "green"
    if inst in ("DISTRIBUTION", "RISK_OFF"):
        return "red"
    return "#8892a4"


def _inst_plain(inst: str) -> str:
    return {
        "ACCUMULATION": "Smart money buying — ride with institutions",
        "DISTRIBUTION": "Smart money selling — don't fight it",
        "RISK_ON": "Risk appetite is elevated — markets in buy mode",
        "RISK_OFF": "Risk appetite is low — institutions de-risking",
        "NEUTRAL": "No clear institutional conviction",
    }.get(inst, inst)


def _risk_mode_plain(mode: str) -> str:
    return {
        "OFFENSIVE": "Conditions favor aggressive sizing",
        "DEFENSIVE": "Reduce size, widen stops",
        "RISK_ON": "Conditions favor aggressive sizing",
        "RISK_OFF": "Reduce size, widen stops",
        "NEUTRAL": "Balanced approach — standard sizing",
    }.get(mode, mode)


def _vol_plain(vol: str) -> str:
    return {
        "LOW_VOL_COMPRESSION": "Compressed volatility — potential breakout imminent",
        "NORMAL": "Normal volatility — standard conditions",
        "TREND_VOLATILITY": "Moderate volatility — trending environment",
        "ELEVATED": "Elevated volatility — reduce position size",
        "PANIC": "Panic volatility — extreme caution required",
    }.get(vol, vol)


def _breadth_plain(label: str) -> str:
    return {
        "STRONG": "Most sectors participating — broad rally",
        "NEUTRAL": "Mixed sector participation — selective approach",
        "WEAK": "Narrow breadth — few stocks leading",
    }.get(label, label)


def _gate_badge(label: str, open_: bool, reason: str = "") -> None:
    if open_:
        st.markdown(
            f'<span style="background:#22c55e;color:#fff;padding:4px 10px;'
            f'border-radius:12px;font-size:13px;font-weight:600;">🟢 {label}</span>',
            unsafe_allow_html=True,
        )
    else:
        tip = f" — {reason}" if reason else ""
        st.markdown(
            f'<span style="background:#ef4444;color:#fff;padding:4px 10px;'
            f'border-radius:12px;font-size:13px;font-weight:600;">🔴 {label}{tip}</span>',
            unsafe_allow_html=True,
        )


def _impact_color(impact: str) -> str:
    return {
        "EXTREME": "#ff4b4b",
        "HIGH": "#f97316",
        "MEDIUM": "#f59e0b",
        "LOW": "#8892a4",
    }.get(impact, "#8892a4")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_sitrep() -> None:
    """
    Renders the full-screen pre-market SITREP dashboard using Streamlit.
    Call this from the main app when should_show_sitrep() returns True and
    the SITREP has not been dismissed.
    """

    # ── Header ───────────────────────────────────────────────────────────────
    now_ist = _get_ist_now()
    date_str = now_ist.strftime("%A, %d %B %Y")

    col_title, col_dismiss = st.columns([8, 1])
    with col_title:
        st.markdown(
            f"## ⚡ SITREP — {date_str} — Pre-Market Intelligence",
        )
    with col_dismiss:
        st.write("")
        if st.button("Dismiss", key="sitrep_dismiss_btn", type="secondary"):
            st.session_state["sitrep_dismissed"] = True
            st.rerun()

    st.divider()

    # ── Section 1: Global Overnight ──────────────────────────────────────────
    st.markdown("### 🌍 Global Overnight")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Index Levels**")
        try:
            import yfinance as yf
            nsei = yf.Ticker("^NSEI")
            nsei_hist = nsei.history(period="2d")
            if nsei_hist is not None and not nsei_hist.empty and len(nsei_hist) >= 1:
                last_close = float(nsei_hist["Close"].iloc[-1])
                prev_close = float(nsei_hist["Close"].iloc[-2]) if len(nsei_hist) >= 2 else last_close
                chg = last_close - prev_close
                chg_pct = (chg / prev_close * 100) if prev_close else 0.0
                arrow = "▲" if chg >= 0 else "▼"
                color = "green" if chg >= 0 else "red"
                st.markdown(
                    f"**Nifty 50** — "
                    f'<span style="color:{color}">{arrow} ₹{last_close:,.2f} ({chg_pct:+.2f}%)</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("Nifty 50: Data unavailable")
        except Exception:
            st.info("Nifty 50: Data unavailable")

        try:
            import yfinance as yf
            # SGX Nifty proxy — use Nifty futures or SGX-listed product
            sgx = yf.Ticker("^NSEI")
            sgx_hist = sgx.history(period="1d", interval="1h")
            if sgx_hist is not None and not sgx_hist.empty:
                sgx_last = float(sgx_hist["Close"].iloc[-1])
                st.markdown(
                    f"**SGX Nifty proxy** (latest intraday) — ₹{sgx_last:,.2f}",
                )
            else:
                st.caption("SGX Nifty proxy: Data unavailable")
        except Exception:
            st.caption("SGX Nifty proxy: Data unavailable")

    with col_right:
        st.markdown("**India VIX**")
        try:
            import yfinance as yf
            vix_ticker = yf.Ticker("^INDIAVIX")
            vix_hist = vix_ticker.history(period="2d")
            if vix_hist is not None and not vix_hist.empty:
                vix_val = float(vix_hist["Close"].iloc[-1])
                st.metric("India VIX", f"{vix_val:.2f}")
                st.caption(_vix_interpretation(vix_val))
            else:
                st.info("India VIX: Data unavailable")
        except Exception:
            st.info("India VIX: Data unavailable")

    st.divider()

    # ── Section 2: Today's Full Regime ───────────────────────────────────────
    st.markdown("### 🧭 Today's Market Regime")

    regime = _cached_regime()

    if regime is None:
        st.warning("Regime data unavailable — check network connection.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)

        # 1. Market Regime
        with c1:
            color = _regime_color(regime.market_regime)
            st.markdown(
                f'<p style="font-size:11px;color:#8892a4;margin-bottom:2px;">Market Regime</p>'
                f'<p style="font-size:18px;font-weight:700;color:{color};margin:0;">'
                f'{regime.market_regime.replace("_", " ")}</p>',
                unsafe_allow_html=True,
            )
            plain_regime = {
                "TRENDING_BULL": "Strong uptrend — ride the wave",
                "TRENDING_BEAR": "Downtrend in force — stay defensive",
                "CHOPPY": "Range-bound chop — trade selectively",
                "COMPRESSION": "Coiling for a move — wait for breakout",
                "EXPANSION": "Volatility expanding — breakouts active",
                "DISTRIBUTION": "Top forming — distribution pressure",
            }.get(regime.market_regime, regime.market_regime)
            st.caption(plain_regime)

        # 2. Volatility
        with c2:
            vol_colors = {
                "LOW_VOL_COMPRESSION": "green",
                "NORMAL": "#22c55e",
                "TREND_VOLATILITY": "#f59e0b",
                "ELEVATED": "#f97316",
                "PANIC": "#ef4444",
            }
            vol_labels = {
                "LOW_VOL_COMPRESSION": "LOW",
                "NORMAL": "NORMAL",
                "TREND_VOLATILITY": "HIGH",
                "ELEVATED": "ELEVATED",
                "PANIC": "EXTREME",
            }
            vol_color = vol_colors.get(regime.volatility_regime, "#8892a4")
            vol_label = vol_labels.get(regime.volatility_regime, regime.volatility_regime)
            st.markdown(
                f'<p style="font-size:11px;color:#8892a4;margin-bottom:2px;">Volatility</p>'
                f'<p style="font-size:18px;font-weight:700;color:{vol_color};margin:0;">'
                f'{vol_label}</p>',
                unsafe_allow_html=True,
            )
            st.caption(_vol_plain(regime.volatility_regime))

        # 3. Breadth
        with c3:
            breadth_colors = {"STRONG": "green", "NEUTRAL": "#f59e0b", "WEAK": "#ef4444"}
            b_color = breadth_colors.get(regime.breadth_label, "#8892a4")
            st.markdown(
                f'<p style="font-size:11px;color:#8892a4;margin-bottom:2px;">Breadth</p>'
                f'<p style="font-size:18px;font-weight:700;color:{b_color};margin:0;">'
                f'{regime.breadth_label}</p>',
                unsafe_allow_html=True,
            )
            st.caption(_breadth_plain(regime.breadth_label))

        # 4. Institutional Activity
        with c4:
            inst_color = _inst_color(regime.institutional_activity)
            st.markdown(
                f'<p style="font-size:11px;color:#8892a4;margin-bottom:2px;">Institutional</p>'
                f'<p style="font-size:18px;font-weight:700;color:{inst_color};margin:0;">'
                f'{regime.institutional_activity.replace("_", " ")}</p>',
                unsafe_allow_html=True,
            )
            st.caption(_inst_plain(regime.institutional_activity))

        # 5. Risk Mode
        with c5:
            rm_colors = {
                "RISK_ON": "green", "OFFENSIVE": "green",
                "RISK_OFF": "#ef4444", "DEFENSIVE": "#ef4444",
                "NEUTRAL": "#8892a4",
            }
            rm_color = rm_colors.get(regime.risk_mode, "#8892a4")
            st.markdown(
                f'<p style="font-size:11px;color:#8892a4;margin-bottom:2px;">Risk Mode</p>'
                f'<p style="font-size:18px;font-weight:700;color:{rm_color};margin:0;">'
                f'{regime.risk_mode.replace("_", " ")}</p>',
                unsafe_allow_html=True,
            )
            st.caption(_risk_mode_plain(regime.risk_mode))

    st.divider()

    # ── Section 3: Today's Playbooks ─────────────────────────────────────────
    st.markdown("### 🎯 Today's Playbooks")

    if regime is not None:
        pb_col, av_col = st.columns(2)
        with pb_col:
            st.markdown("**Play today:**")
            plays = regime.recommended_playbooks
            if plays:
                for pid in plays[:6]:
                    st.markdown(f"✅ **PLAY:** {_plain(pid, _PLAYBOOK_PLAIN)}")
            else:
                st.caption("No playbooks recommended — stay in cash.")

        with av_col:
            st.markdown("**Avoid today:**")
            avoids = regime.avoid_patterns
            if avoids:
                for pid in avoids[:6]:
                    st.markdown(f"❌ **AVOID:** {_plain(pid, _AVOID_PLAIN)}")
            else:
                st.caption("No explicit avoids — regime is clear.")
    else:
        st.info("Playbook data unavailable.")

    st.divider()

    # ── Section 4: IronLock Gate Status ──────────────────────────────────────
    st.markdown("### 🔒 IronLock Gate Status")

    # Gather values from session state / regime with safe defaults
    kill_switch_active = st.session_state.get("kill_switch_active", False)
    today_drawdown = float(st.session_state.get("today_drawdown_pct", 0.0))
    fno_banned = st.session_state.get("fno_banned", False)
    consecutive_losses = int(st.session_state.get("consecutive_losses", 0))
    open_positions_list = st.session_state.get("open_positions", [])
    open_positions_count = len(open_positions_list) if isinstance(open_positions_list, list) else 0
    max_positions = int(st.session_state.get("max_open_positions", 5))
    regime_confidence = float(regime.regime_confidence) if regime else 0.0

    gates = [
        ("Kill Switch", not kill_switch_active, "Kill switch is active — all trading halted"),
        ("Regime", regime_confidence >= 40, f"Confidence {regime_confidence:.0f}% < 40% — unreliable"),
        ("Daily Loss", today_drawdown < 5.0, f"Down {today_drawdown:.1f}% today — limit breached"),
        ("F&O Ban", not fno_banned, "Symbol is in F&O ban list"),
        ("Consecutive Losses", consecutive_losses < 3, f"{consecutive_losses} losses in a row — tilt risk"),
        ("Positions", open_positions_count < max_positions,
         f"{open_positions_count}/{max_positions} positions open"),
    ]

    gate_cols = st.columns(6)
    for i, (label, is_open, reason) in enumerate(gates):
        with gate_cols[i]:
            _gate_badge(label, is_open, "" if is_open else reason)

    # Count blocked gates
    blocked = sum(1 for _, is_open, _ in gates if not is_open)
    if blocked == 0:
        st.success("All gates clear — system is armed and ready to trade.")
    elif blocked == 1:
        st.warning(f"{blocked} gate blocked — review before trading.")
    else:
        st.error(f"{blocked} gates blocked — do NOT enter new trades.")

    st.divider()

    # ── Section 5: Events Today + Tomorrow ───────────────────────────────────
    st.markdown("### 📅 Economic Events")

    try:
        from risk.event_risk_scanner import get_event_risk
        event_report = get_event_risk()

        today_events = event_report.today_events
        tomorrow_events = event_report.tomorrow_events

        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            today_label = now_ist.strftime("%d %b") + " (Today)"
            st.markdown(f"**{today_label}**")
            if today_events:
                for ev in today_events:
                    impact = ev.get("impact", "LOW")
                    color = _impact_color(impact)
                    event_name = ev.get("event", "Unknown event")
                    ev_type = ev.get("type", "")
                    st.markdown(
                        f'<span style="color:{color};font-weight:600;">[{impact}]</span> '
                        f'{event_name} <span style="color:#8892a4;font-size:12px;">({ev_type})</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No major events today.")

        with ev_col2:
            tomorrow_label = (now_ist + timedelta(days=1)).strftime("%d %b") + " (Tomorrow)"
            st.markdown(f"**{tomorrow_label}**")
            if tomorrow_events:
                for ev in tomorrow_events:
                    impact = ev.get("impact", "LOW")
                    color = _impact_color(impact)
                    event_name = ev.get("event", "Unknown event")
                    ev_type = ev.get("type", "")
                    st.markdown(
                        f'<span style="color:{color};font-weight:600;">[{impact}]</span> '
                        f'{event_name} <span style="color:#8892a4;font-size:12px;">({ev_type})</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No major events tomorrow.")

        if event_report.max_impact in ("EXTREME", "HIGH"):
            action_color = _impact_color(event_report.max_impact)
            st.markdown(
                f'<div style="background:#1e1e2e;border-left:4px solid {action_color};'
                f'padding:10px 14px;border-radius:4px;margin-top:8px;">'
                f'⚠️ <b>Action required:</b> {event_report.action}</div>',
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.info(f"Event calendar unavailable ({type(exc).__name__}).")

    st.divider()

    # ── Section 6: Positions at Risk ─────────────────────────────────────────
    st.markdown("### ⚠️ Positions at Risk")

    portfolio_path = Path("logs/portfolio_state.json")
    positions_at_risk = []

    if portfolio_path.exists():
        try:
            with open(portfolio_path) as f:
                portfolio_data = json.load(f)

            positions = portfolio_data.get("positions", {})
            if isinstance(positions, dict):
                positions = list(positions.values())

            for pos in positions:
                symbol = pos.get("symbol", pos.get("ticker", "UNKNOWN"))
                stop_loss = float(pos.get("stop_loss", pos.get("stop", 0)) or 0)
                current_price = float(pos.get("current_price", pos.get("ltp", pos.get("entry_price", 0))) or 0)

                if stop_loss <= 0 or current_price <= 0:
                    continue

                # Within 1% of stop loss?
                gap_pct = ((current_price - stop_loss) / stop_loss * 100) if stop_loss > 0 else 100
                if 0 < gap_pct < 1.0:
                    positions_at_risk.append({
                        "symbol": symbol,
                        "stop_loss": stop_loss,
                        "current_price": current_price,
                        "gap_pct": gap_pct,
                    })

        except Exception as exc:
            st.caption(f"Could not read portfolio state: {type(exc).__name__}")

    if positions_at_risk:
        for pos in positions_at_risk:
            st.warning(
                f"⚠️ **{pos['symbol']}** — near stop loss "
                f"(₹{pos['stop_loss']:,.2f}). "
                f"Current: ₹{pos['current_price']:,.2f} "
                f"({pos['gap_pct']:.2f}% gap)."
            )
    else:
        if portfolio_path.exists():
            st.success("No positions near stop loss. All positions are safe.")
        else:
            st.caption("No open positions found (portfolio_state.json not present).")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.caption("Refresh anytime · Dismisses automatically at 9:30 AM")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import ast
    import sys

    src = Path(__file__).read_text()
    try:
        ast.parse(src)
        print("✓ sitrep.py syntax OK")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        sys.exit(1)

    print(f"should_show_sitrep() would return: <depends on IST time>")
    print("Module loaded successfully.")
