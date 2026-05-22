"""
Session Debrief — auto-generated end-of-day review.
Answers: What happened today? What worked? What didn't? What to watch tomorrow?
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

IST = timezone(timedelta(hours=5, minutes=30))


def should_show_debrief() -> bool:
    """Returns True if IST time is between 15:25 and 16:00 AND not already dismissed today."""
    now = datetime.now(IST)
    today_key = f"debrief_dismissed_{now.date()}"
    return (now.hour == 15 and now.minute >= 25) and not st.session_state.get(today_key)


@st.cache_data(ttl=3600, show_spinner=False)
def _generate_debrief_text(trades_json: str, regime: str, nifty_move: float) -> str:
    """Call DeepSeek to generate the end-of-day debrief."""
    try:
        from config import settings
    except Exception:
        return ""

    if not settings.deepseek_api_key:
        return ""

    import requests

    prompt = f"""End of trading day debrief for a retail NSE trader.

Trades today: {trades_json}
Market regime: {regime}
Nifty move: {nifty_move:+.2f}%

Write a crisp debrief in exactly 4 bullet points:
• What the market did and whether it matched the regime call
• What worked today (be specific about setup types)
• What to avoid tomorrow based on today
• Top 2 stocks to watch tomorrow and why

Max 120 words. No fluff."""

    try:
        resp = requests.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            json={
                "model": settings.deepseek_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 200,
            },
            timeout=15,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"] if "choices" in data else ""
    except Exception:
        return ""


def _load_portfolio_state() -> dict[str, Any]:
    """Load portfolio state from logs/portfolio_state.json."""
    try:
        path = Path("logs/portfolio_state.json")
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _get_nifty_data() -> tuple[float, float]:
    """Return (nifty_move_pct, vix_change). Uses yfinance."""
    try:
        import yfinance as yf
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="2d")
        if len(hist) >= 2:
            prev_close = hist["Close"].iloc[-2]
            last_close = hist["Close"].iloc[-1]
            move_pct = (last_close - prev_close) / prev_close * 100
        else:
            move_pct = 0.0

        vix = yf.Ticker("^INDIAVIX")
        vh = vix.history(period="2d")
        if len(vh) >= 2:
            vix_change = vh["Close"].iloc[-1] - vh["Close"].iloc[-2]
        else:
            vix_change = 0.0

        return move_pct, vix_change
    except Exception:
        return 0.0, 0.0


def _get_regime_info() -> tuple[str, str]:
    """Return (opening_regime, current_regime) strings."""
    try:
        from ui.regime_bar import get_regime
        current = get_regime().get("regime", "UNKNOWN")
    except Exception:
        current = "UNKNOWN"

    opening = st.session_state.get("session_opening_regime", current)
    return opening, current


def _compute_scorecard(state: dict[str, Any]) -> dict[str, Any]:
    """Parse portfolio state into scorecard metrics."""
    now_date = datetime.now(IST).date().isoformat()
    trades = state.get("trades", [])
    today_trades = [t for t in trades if str(t.get("date", "")).startswith(now_date)]

    winners = [t for t in today_trades if t.get("pnl", 0) > 0]
    losers = [t for t in today_trades if t.get("pnl", 0) < 0]
    total_pnl = sum(t.get("pnl", 0) for t in today_trades)
    starting_capital = state.get("starting_capital", state.get("initial_capital", 1_000_000))

    best_trade: dict | None = None
    worst_trade: dict | None = None
    if today_trades:
        best_trade = max(today_trades, key=lambda t: t.get("r_multiple", t.get("pnl", 0)))
        worst_trade = min(today_trades, key=lambda t: t.get("r_multiple", t.get("pnl", 0)))

    return {
        "count": len(today_trades),
        "winners": len(winners),
        "losers": len(losers),
        "total_pnl": total_pnl,
        "pnl_pct": (total_pnl / starting_capital * 100) if starting_capital else 0.0,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }


def render_session_debrief() -> None:
    """Builds and displays the end-of-day session debrief."""
    now = datetime.now(IST)
    today_key = f"debrief_dismissed_{now.date()}"

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_dismiss = st.columns([5, 1])
    with col_title:
        st.markdown(
            f"<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
            f"font-size:1.25rem;letter-spacing:2px;margin-bottom:0'>📋 SESSION DEBRIEF — "
            f"{now.strftime('%d %b %Y')}</h2>",
            unsafe_allow_html=True,
        )
    with col_dismiss:
        if st.button("Dismiss", key="debrief_dismiss_btn", width="stretch"):
            st.session_state[today_key] = True
            st.rerun()

    st.markdown(
        "<div style='height:.25rem'></div>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Today's Scorecard ─────────────────────────────────────────
    st.markdown(
        "<div style='color:#8892a4;text-transform:uppercase;font-size:.72rem;"
        "letter-spacing:.07em;margin:.8rem 0 .4rem'>1 · Today's Scorecard</div>",
        unsafe_allow_html=True,
    )

    state = _load_portfolio_state()
    sc = _compute_scorecard(state)

    pnl_color = "#00d4a0" if sc["total_pnl"] >= 0 else "#ff4466"
    pnl_sign = "+" if sc["total_pnl"] >= 0 else ""
    pct_sign = "+" if sc["pnl_pct"] >= 0 else ""

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Trades Today", sc["count"])
    sc2.metric("Winners", sc["winners"])
    sc3.metric("Losers", sc["losers"])
    sc4.metric(
        "Day P&L",
        f"{pnl_sign}₹{abs(sc['total_pnl']):,.0f}",
        delta=f"{pct_sign}{sc['pnl_pct']:.2f}% of capital",
    )

    if sc["best_trade"]:
        bt = sc["best_trade"]
        bt_sym = bt.get("symbol", bt.get("tradingsymbol", "—"))
        bt_r = bt.get("r_multiple", "")
        bt_r_str = f" | R: {bt_r:.2f}x" if bt_r != "" else ""
        st.markdown(
            f"<div style='font-size:.8rem;color:#6b7280;margin-top:.4rem'>"
            f"Best: <span style='color:#00d4a0;font-weight:700'>{bt_sym}</span>{bt_r_str} &nbsp;|&nbsp; "
            f"Worst: <span style='color:#ff4466;font-weight:700'>"
            f"{sc['worst_trade'].get('symbol', sc['worst_trade'].get('tradingsymbol', '—'))}</span>"
            + (f" | R: {sc['worst_trade'].get('r_multiple', ''):.2f}x"
               if sc.get("worst_trade") and sc["worst_trade"].get("r_multiple", "") != "" else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    # ── Section 2: What the Market Did ───────────────────────────────────────
    st.markdown(
        "<div style='color:#8892a4;text-transform:uppercase;font-size:.72rem;"
        "letter-spacing:.07em;margin:1rem 0 .4rem'>2 · What the Market Did</div>",
        unsafe_allow_html=True,
    )

    opening_regime, current_regime = _get_regime_info()
    nifty_move, vix_change = _get_nifty_data()

    regime_changed = opening_regime != current_regime
    regime_label = (
        f"<span style='color:#f59e0b'>Changed: {opening_regime} → {current_regime}</span>"
        if regime_changed
        else f"<span style='color:#00d4a0'>Stable: {current_regime}</span>"
    )
    nifty_color = "#00d4a0" if nifty_move >= 0 else "#ff4466"
    nifty_sign = "+" if nifty_move >= 0 else ""
    vix_color = "#ff4466" if vix_change > 0 else "#00d4a0"
    vix_sign = "+" if vix_change >= 0 else ""

    st.markdown(
        f"<div style='background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);"
        f"border-radius:10px;padding:12px 16px;font-size:.82rem'>"
        f"<div style='margin-bottom:.35rem'>Regime: {regime_label}</div>"
        f"<div style='color:#c8cfe0'>Nifty: "
        f"<span style='color:{nifty_color};font-weight:700'>{nifty_sign}{nifty_move:.2f}%</span>"
        f" &nbsp;|&nbsp; VIX change: "
        f"<span style='color:{vix_color};font-weight:700'>{vix_sign}{vix_change:.2f}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Section 3: Setups that Triggered ─────────────────────────────────────
    st.markdown(
        "<div style='color:#8892a4;text-transform:uppercase;font-size:.72rem;"
        "letter-spacing:.07em;margin:1rem 0 .4rem'>3 · Setups that Triggered</div>",
        unsafe_allow_html=True,
    )

    setups = st.session_state.get("iq2_pipeline_cache", [])
    if setups:
        setup_rows = []
        for s in setups[:8]:
            sym_s = s.get("symbol", s.get("tradingsymbol", ""))
            pivot = s.get("pivot", s.get("entry", 0))
            setup_type = s.get("setup", s.get("archetype", "Setup"))
            try:
                import yfinance as yf
                ticker = yf.Ticker(f"{sym_s}.NS")
                info = ticker.fast_info
                cur_px = getattr(info, "last_price", None) or 0
            except Exception:
                cur_px = 0

            if pivot and cur_px:
                diff_pct = (cur_px - pivot) / pivot * 100
                worked_str = (
                    f"<span style='color:#00d4a0'>+{diff_pct:.1f}% vs pivot ✓</span>"
                    if diff_pct > 0
                    else f"<span style='color:#ff4466'>{diff_pct:.1f}% vs pivot ✗</span>"
                )
            else:
                worked_str = "<span style='color:#6b7280'>—</span>"

            setup_rows.append(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:.8rem;padding:.25rem 0;border-bottom:1px solid rgba(255,255,255,.05)'>"
                f"<span style='color:#e8eaf0;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>{sym_s}</span>"
                f"<span style='color:#8892a4'>{setup_type}</span>"
                f"{worked_str}</div>"
            )

        st.markdown(
            "<div style='background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);"
            "border-radius:10px;padding:12px 16px'>"
            + "".join(setup_rows)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#6b7280;font-size:.8rem;font-style:italic'>"
            "No setups cached from today's session.</div>",
            unsafe_allow_html=True,
        )

    # ── Section 4: JARVIS Debrief ────────────────────────────────────────────
    st.markdown(
        "<div style='color:#8892a4;text-transform:uppercase;font-size:.72rem;"
        "letter-spacing:.07em;margin:1rem 0 .4rem'>4 · JARVIS Debrief</div>",
        unsafe_allow_html=True,
    )

    trades_snapshot = json.dumps(
        [
            {
                "symbol": t.get("symbol", t.get("tradingsymbol", "")),
                "pnl": t.get("pnl", 0),
                "r_multiple": t.get("r_multiple", ""),
                "setup": t.get("setup", t.get("archetype", "")),
            }
            for t in state.get("trades", [])
            if str(t.get("date", "")).startswith(datetime.now(IST).date().isoformat())
        ][:10],
        default=str,
    )

    with st.spinner("JARVIS is reviewing the session..."):
        debrief_text = _generate_debrief_text(trades_snapshot, current_regime, nifty_move)

    if debrief_text:
        st.markdown(
            f"<div style='background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.2);"
            f"border-radius:10px;padding:14px 18px;font-size:.84rem;color:#c8cfe0;"
            f"line-height:1.7;white-space:pre-wrap'>{debrief_text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("JARVIS debrief unavailable — add DEEPSEEK_API_KEY to .env to enable.")

    # ── Section 5: Tomorrow's Prep ───────────────────────────────────────────
    st.markdown(
        "<div style='color:#8892a4;text-transform:uppercase;font-size:.72rem;"
        "letter-spacing:.07em;margin:1rem 0 .4rem'>5 · Tomorrow's Prep</div>",
        unsafe_allow_html=True,
    )

    # Events tomorrow
    try:
        from risk.event_risk_scanner import get_event_risk
        event_report = get_event_risk()
        events = getattr(event_report, "events", []) or []
        if events:
            event_items = "".join(
                f"<li style='color:#f59e0b;font-size:.8rem'>{e}</li>"
                for e in events[:5]
            )
            st.markdown(
                f"<div style='background:rgba(245,158,11,.06);"
                f"border:1px solid rgba(245,158,11,.2);border-radius:8px;"
                f"padding:10px 14px;margin-bottom:.8rem'>"
                f"<div style='color:#f59e0b;font-size:.72rem;font-weight:700;"
                f"text-transform:uppercase;margin-bottom:.4rem'>Events Tomorrow</div>"
                f"<ul style='margin:0;padding-left:1.2rem'>{event_items}</ul></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='color:#6b7280;font-size:.78rem'>No major events flagged for tomorrow.</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            "<div style='color:#6b7280;font-size:.78rem'>Event scanner unavailable.</div>",
            unsafe_allow_html=True,
        )

    # Notes input
    st.markdown(
        "<div style='font-size:.78rem;color:#8892a4;margin:.6rem 0 .3rem'>"
        "Set your alerts — notes for tomorrow:</div>",
        unsafe_allow_html=True,
    )
    notes_key = f"tomorrow_prep_notes_{now.date()}"
    existing_notes = st.session_state.get(notes_key, "")
    notes = st.text_area(
        "Notes for tomorrow",
        value=existing_notes,
        height=90,
        key=f"tomorrow_prep_input_{now.date()}",
        label_visibility="collapsed",
        placeholder="Symbols to watch, levels, setups, reminders...",
    )
    if st.button("Save notes", key="save_tomorrow_prep"):
        st.session_state[notes_key] = notes
        try:
            Path("logs").mkdir(exist_ok=True)
            Path("logs/tomorrow_prep.txt").write_text(
                f"[{now.strftime('%Y-%m-%d')}]\n{notes}\n"
            )
            st.success("Notes saved to logs/tomorrow_prep.txt")
        except Exception as exc:
            st.error(f"Could not save notes: {exc}")
