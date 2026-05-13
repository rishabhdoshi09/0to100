"""Auto-journal, equity curve dashboard, Kelly criterion, and cognitive bias detector."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

JOURNAL_DB = Path("journal.db")


def _init_journal_db():
    conn = sqlite3.connect(JOURNAL_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            symbol      TEXT,
            action      TEXT,
            price       REAL,
            qty         INTEGER,
            indicators  TEXT,
            news_snap   TEXT,
            voice_note  TEXT,
            chart_state TEXT,
            ai_verdict  TEXT
        )
    """)
    # Non-destructive migration: add ai_verdict column to existing DBs
    try:
        conn.execute("ALTER TABLE journal_entries ADD COLUMN ai_verdict TEXT")
    except Exception:
        pass  # column already exists
    conn.commit()
    conn.close()


def log_trade_to_journal(
    symbol: str,
    action: str,
    price: float,
    qty: int,
    indicators: dict | None = None,
    news_snap: str = "",
    chart_state: dict | None = None,
    ai_verdict: str = "",
):
    """Called automatically when a trade is placed. Snapshots context + AI verdict."""
    _init_journal_db()
    conn = sqlite3.connect(JOURNAL_DB)
    conn.execute(
        "INSERT INTO journal_entries "
        "(timestamp,symbol,action,price,qty,indicators,news_snap,chart_state,ai_verdict) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            datetime.now().isoformat(),
            symbol,
            action,
            price,
            qty,
            json.dumps(indicators or {}),
            news_snap[:500],
            json.dumps(chart_state or {}),
            ai_verdict,
        ),
    )
    conn.commit()
    conn.close()


def _load_entries(limit: int = 200) -> pd.DataFrame:
    _init_journal_db()
    conn = sqlite3.connect(JOURNAL_DB)
    df = pd.read_sql("SELECT * FROM journal_entries ORDER BY timestamp DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df


def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Full Kelly criterion: f = (p*b - q) / b  where b = avg_win/avg_loss."""
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    p = win_rate / 100
    q = 1 - p
    b = avg_win / avg_loss
    return max(0.0, (p * b - q) / b)


def _run_bias_detector(entries_json: str) -> str:
    """Use dual-LLM (DeepSeek → Claude) to detect cognitive biases in the trade journal."""
    from ai.dual_llm_service import get_service
    svc = get_service()
    prompt = (
        "Analyse this trading journal for cognitive biases. "
        "Look for: revenge trading, FOMO entries, premature exits, anchoring, overconfidence. "
        "Be specific and reference actual trades. Suggest one concrete fix for each bias found.\n\n"
        "Journal (JSON):\n" + entries_json[:3000]
    )
    text, dm, detail = svc.ask(prompt, max_tokens=600)
    from ai.dual_llm_service import get_service as _gs
    badge = _gs().badge(dm, detail)
    return f"{badge}<br>{text}"


def render_journal():
    """Full journal & performance analytics tab."""
    st.markdown("### 📓 Journal & Performance Analytics")
    _init_journal_db()

    entries = _load_entries()

    # ── Stats header ──────────────────────────────────────────────────────────
    from paper_trading import get_trading_summary, get_equity_curve, get_closed_positions, init_db
    init_db()

    summary = get_trading_summary()
    closed  = get_closed_positions()

    if summary:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Trades",   summary.get("total_trades", 0))
        m2.metric("Win Rate",       f"{summary.get('win_rate', 0):.1f}%")
        m3.metric("Total P&L",      f"₹{summary.get('total_pnl', 0):,.0f}")
        m4.metric("Avg Win",        f"₹{summary.get('avg_win', 0):,.0f}")
        m5.metric("Avg Loss",       f"₹{summary.get('avg_loss', 0):,.0f}")

        # Kelly criterion
        wr   = summary.get("win_rate", 50)
        aw   = abs(summary.get("avg_win", 1))
        al   = abs(summary.get("avg_loss", 1))
        kelly = _kelly_fraction(wr, aw, al)
        half_kelly = kelly / 2
        st.markdown(
            f"<div class='devbloom-card' style='padding:.6rem 1rem'>"
            f"<span style='color:#8892a4;font-size:.72rem'>Kelly Criterion: </span>"
            f"<span style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-weight:600'>"
            f"Full Kelly {kelly*100:.1f}% · Half Kelly {half_kelly*100:.1f}% of capital per trade</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Equity curve vs Nifty ─────────────────────────────────────────────────
    eq_df = get_equity_curve()
    if eq_df is not None and not eq_df.empty:
        st.markdown("#### Equity Curve vs Nifty 50")

        # Fetch benchmark
        try:
            import yfinance as yf
            n50 = yf.download("^NSEI", period="1y", interval="1d", progress=False, auto_adjust=True)
            if isinstance(n50.columns, pd.MultiIndex):
                n50.columns = [c[0] for c in n50.columns]
            n50 = n50["Close"].reset_index()
            n50.columns = ["date", "nifty"]
            n50["date"] = n50["date"].astype(str)
            has_bench = True
        except Exception:
            has_bench = False

        eq_col = "equity" if "equity" in eq_df.columns else eq_df.columns[-1]
        dt_col = "date"   if "date"   in eq_df.columns else eq_df.columns[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df[dt_col], y=eq_df[eq_col],
            name="My Portfolio", mode="lines",
            line=dict(color="#00d4ff", width=2),
        ))

        if has_bench and not n50.empty:
            base_nifty = n50["nifty"].iloc[0]
            base_eq    = eq_df[eq_col].iloc[0]
            n50["scaled"] = n50["nifty"] / base_nifty * base_eq
            fig.add_trace(go.Scatter(
                x=n50["date"], y=n50["scaled"],
                name="Nifty 50 (scaled)", mode="lines",
                line=dict(color="#ffb800", width=1.5, dash="dot"),
            ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,12,28,0.6)",
            margin=dict(l=0, r=0, t=8, b=0), height=240,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
            xaxis=dict(color="#8892a4", showgrid=False),
            yaxis=dict(color="#8892a4", tickprefix="₹", gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="journal_equity_chart")

    # ── Auto-journal entries ───────────────────────────────────────────────────
    st.markdown("#### Auto-Journal")
    if entries.empty:
        st.info("No journal entries yet. Entries are auto-logged when trades are placed.")
    else:
        for _, row in entries.head(20).iterrows():
            ind = json.loads(row.get("indicators") or "{}")
            ts  = row.get("timestamp", "")[:16]
            action_color = "#00d4ff" if row["action"] == "BUY" else "#ff4466"
            rsi_txt = f"RSI {ind.get('rsi_14', '—'):.1f}" if "rsi_14" in ind else ""
            ai_v = row.get("ai_verdict", "") or ""
            ai_v_color = {"GO": "#00ff88", "CAUTION": "#ffb800", "ABORT": "#ff4466"}.get(ai_v, "")
            ai_v_html = (
                f"<span style='font-size:.65rem;color:{ai_v_color};font-weight:600;"
                f"font-family:JetBrains Mono,monospace;margin-left:.4rem'>{ai_v}</span>"
                if ai_v else ""
            )

            st.markdown(
                f"<div class='devbloom-card' style='padding:.55rem 1rem;margin-bottom:.35rem'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"  <span style='color:{action_color};font-weight:700;font-size:.8rem'>"
                f"    {row['action']} {row['symbol']} — ₹{row['price']:,.2f} × {row['qty']}"
                f"    {ai_v_html}"
                f"  </span>"
                f"  <span style='color:#8892a4;font-size:.7rem'>{ts}</span>"
                f"</div>"
                f"<div style='color:#8892a4;font-size:.72rem;margin-top:.15rem'>{rsi_txt}</div>"
                + (f"<div style='color:#c8cfe0;font-size:.75rem;margin-top:.2rem'>{row['news_snap']}</div>" if row.get('news_snap') else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    # ── Cognitive bias detector ────────────────────────────────────────────────
    st.markdown("#### Cognitive Bias Detector")
    st.caption("AI-powered analysis of your trade patterns. Requires ANTHROPIC_API_KEY.")
    if st.button("🧠 Analyse My Journal for Biases", key="bias_detect"):
        if entries.empty:
            st.warning("No journal entries to analyse yet.")
        else:
            sample = entries.head(30)[["timestamp","symbol","action","price"]].to_json(orient="records")
            with st.spinner("Analysing…"):
                result = _run_bias_detector(sample)
            st.markdown(
                f"<div class='devbloom-card'><div style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{result}</div></div>",
                unsafe_allow_html=True,
            )

    # ── Closed positions table ────────────────────────────────────────────────
    if closed is not None and not closed.empty:
        with st.expander("Closed Trades"):
            st.dataframe(closed, use_container_width=True)
