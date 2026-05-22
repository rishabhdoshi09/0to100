"""
Alpha Decay Tracker — is your trading edge eroding?
Reads from logs/journal.db (table: trades). Falls back to demo data.
"""
from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

_DB_PATH   = Path("logs/journal.db")
_ROLL_WIN  = 10   # rolling window for win rate / expectancy
_DECAY_WIN = 30   # longer window for decay comparison

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,28,0.6)",
    font=dict(color="#8892a4", size=11),
    margin=dict(l=0, r=0, t=32, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
    xaxis=dict(color="#8892a4", showgrid=False, zeroline=False),
    yaxis=dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)", zeroline=False),
)

GREEN = "#00ff88"
RED   = "#ff4466"
CYAN  = "#00d4ff"
AMBER = "#ffb800"
WHITE = "#e8eaf0"

_SETUP_TYPES = ["Breakout", "Pullback", "Reversal", "Momentum", "Mean-Revert", "Earnings"]


# ── Demo data generator ───────────────────────────────────────────────────────

def _generate_demo_trades(n: int = 60) -> pd.DataFrame:
    rng     = np.random.default_rng(2024)
    try:
        from data.nse_universe import NIFTY500
        symbols = NIFTY500
    except Exception:
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                   "SBIN", "WIPRO", "LT", "AXISBANK", "BAJFINANCE"]

    # Recent trades degrade slightly to simulate alpha decay
    win_probs = np.linspace(0.62, 0.38, n)  # decays from 62% → 38%

    rows = []
    date = datetime.date.today() - datetime.timedelta(days=n * 2)
    for i in range(n):
        date += datetime.timedelta(days=int(rng.integers(1, 4)))
        won   = rng.random() < win_probs[i]
        ep    = rng.uniform(200, 3000)
        if won:
            pct = rng.uniform(0.5, 5.0)
        else:
            pct = -rng.uniform(0.3, 3.5)
        xp   = ep * (1 + pct / 100)
        rows.append({
            "id":          i + 1,
            "date":        str(date),
            "symbol":      rng.choice(symbols),
            "action":      "BUY",
            "entry_price": round(ep, 2),
            "exit_price":  round(xp, 2),
            "pnl":         round((xp - ep), 2),
            "pnl_pct":     round(pct, 2),
            "setup_type":  rng.choice(_SETUP_TYPES),
            "notes":       rng.choice(["trending", "ranging", "breakout failed",
                                       "strong trend", "choppy", "gap fill"]),
        })
    return pd.DataFrame(rows)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_trades() -> tuple[pd.DataFrame, bool]:
    """
    Returns (df, is_demo).
    Tries logs/journal.db → table trades. Falls back to demo data.
    """
    if _DB_PATH.exists():
        try:
            conn = sqlite3.connect(_DB_PATH)
            df   = pd.read_sql("SELECT * FROM trades ORDER BY date ASC", conn)
            conn.close()
            if not df.empty and len(df) >= 5:
                df["date"]    = pd.to_datetime(df["date"], errors="coerce")
                df["pnl"]     = pd.to_numeric(df["pnl"],     errors="coerce")
                df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
                df["win"]     = df["pnl"] > 0
                return df.dropna(subset=["date", "pnl"]).reset_index(drop=True), False
        except Exception:
            pass

    demo = _generate_demo_trades(60)
    demo["date"]    = pd.to_datetime(demo["date"])
    demo["pnl"]     = pd.to_numeric(demo["pnl"])
    demo["pnl_pct"] = pd.to_numeric(demo["pnl_pct"])
    demo["win"]     = demo["pnl"] > 0
    return demo, True


# ── Rolling metrics ───────────────────────────────────────────────────────────

def _rolling_winrate(df: pd.DataFrame, window: int = _ROLL_WIN) -> pd.Series:
    return df["win"].astype(float).rolling(window, min_periods=window).mean()


def _rolling_expectancy(df: pd.DataFrame, window: int = _ROLL_WIN) -> pd.Series:
    """Rolling expectancy = win_rate * avg_win - loss_rate * avg_loss (all in pct)."""
    results = []
    for i in range(len(df)):
        start = max(0, i - window + 1)
        chunk = df.iloc[start : i + 1]
        if len(chunk) < window:
            results.append(np.nan)
            continue
        wins   = chunk.loc[chunk["win"], "pnl_pct"]
        losses = chunk.loc[~chunk["win"], "pnl_pct"]
        wr     = len(wins) / len(chunk)
        lr     = 1 - wr
        avg_w  = wins.mean()   if len(wins)   > 0 else 0.0
        avg_l  = abs(losses.mean()) if len(losses) > 0 else 0.0
        results.append(wr * avg_w - lr * avg_l)
    return pd.Series(results, index=df.index)


def _detect_decay(df: pd.DataFrame) -> Optional[str]:
    if len(df) < _DECAY_WIN:
        return None
    last10   = df.tail(_ROLL_WIN)
    last30   = df.tail(_DECAY_WIN)
    wr_last10 = last10["win"].mean()
    wr_last30 = last30["win"].mean()
    if wr_last10 < 0.40 and wr_last30 > 0.55:
        return (
            f"Last 10 trades win rate: **{wr_last10:.0%}** — "
            f"vs last 30: **{wr_last30:.0%}**. Edge appears to be deteriorating."
        )
    return None


# ── Chart builders ────────────────────────────────────────────────────────────

def _win_rate_chart(df: pd.DataFrame) -> go.Figure:
    rwr = _rolling_winrate(df)
    fig = go.Figure()
    fig.add_hline(y=0.5, line_color="rgba(255,255,255,0.15)", line_dash="dash", line_width=1)
    fig.add_hline(y=0.55, line_color="rgba(0,255,136,0.2)",   line_dash="dot",  line_width=1)
    fig.add_hline(y=0.40, line_color="rgba(255,68,102,0.2)",  line_dash="dot",  line_width=1)
    fig.add_trace(go.Scatter(
        x=df["date"], y=rwr,
        mode="lines+markers",
        name=f"{_ROLL_WIN}-trade Rolling Win Rate",
        line=dict(color=CYAN, width=2),
        marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
    ))
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text=f"Rolling {_ROLL_WIN}-Trade Win Rate", font=dict(color=WHITE, size=13))
    layout["height"]  = 240
    layout["yaxis"]   = dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)",
                              tickformat=".0%", range=[0, 1])
    fig.update_layout(**layout)
    return fig


def _expectancy_chart(df: pd.DataFrame) -> go.Figure:
    exp = _rolling_expectancy(df)
    colors = [GREEN if v >= 0 else RED for v in exp.fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=exp,
        marker_color=colors, opacity=0.85, name="Expectancy (%)",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text=f"Rolling {_ROLL_WIN}-Trade Expectancy (%)", font=dict(color=WHITE, size=13))
    layout["height"]  = 240
    layout["yaxis"]   = dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)",
                              zeroline=True, zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_layout(**layout)
    return fig


def _setup_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    if "setup_type" not in df.columns or df["setup_type"].isna().all():
        return go.Figure()
    grp = (
        df.groupby("setup_type")
        .agg(trades=("win", "count"), win_rate=("win", "mean"), avg_pnl_pct=("pnl_pct", "mean"))
        .reset_index()
        .sort_values("win_rate", ascending=False)
    )
    colors = [GREEN if wr >= 0.5 else RED for wr in grp["win_rate"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["setup_type"], y=grp["win_rate"],
        marker_color=colors, opacity=0.85, name="Win Rate",
        text=[f"{v:.0%}<br>{t} trades" for v, t in zip(grp["win_rate"], grp["trades"])],
        textposition="outside",
        textfont=dict(color=WHITE, size=10),
    ))
    fig.add_hline(y=0.5, line_color="rgba(255,255,255,0.2)", line_dash="dash", line_width=1)
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text="Win Rate by Setup Type", font=dict(color=WHITE, size=13))
    layout["height"]  = 280
    layout["yaxis"]   = dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)",
                              tickformat=".0%", range=[0, 1.15])
    fig.update_layout(**layout)
    return fig


def _monthly_pnl_heatmap(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["year"]  = df2["date"].dt.year
    df2["month"] = df2["date"].dt.month
    monthly = df2.groupby(["year", "month"])["pnl_pct"].sum().reset_index()

    years  = sorted(monthly["year"].unique())
    months = list(range(1, 13))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    z = []
    for yr in years:
        row = []
        for mo in months:
            val = monthly.loc[(monthly["year"] == yr) & (monthly["month"] == mo), "pnl_pct"]
            row.append(float(val.iloc[0]) if not val.empty else None)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=month_names,
        y=[str(y) for y in years],
        colorscale=[[0, "#ff4466"], [0.5, "#1a1f35"], [1, "#00ff88"]],
        zmid=0,
        text=[[f"{v:.1f}%" if v is not None else "—" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10, color=WHITE),
        showscale=True,
        colorbar=dict(tickfont=dict(color="#8892a4", size=9)),
    ))
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text="Monthly PnL Heatmap (%)", font=dict(color=WHITE, size=13))
    layout["height"]  = max(160, 80 + len(years) * 50)
    layout["xaxis"]   = dict(color="#8892a4")
    layout["yaxis"]   = dict(color="#8892a4", showgrid=False)
    fig.update_layout(**layout)
    return fig


def _regime_chart(df: pd.DataFrame) -> go.Figure:
    """Correlate underperformance with regime heuristics from notes column."""
    if "notes" not in df.columns or df["notes"].isna().all():
        return go.Figure()

    df2 = df.copy()
    df2["regime"] = df2["notes"].str.lower().apply(
        lambda n: "Trending" if any(k in str(n) for k in ["trend", "breakout", "strong"])
        else ("Ranging" if any(k in str(n) for k in ["rang", "chop", "gap"]) else "Other")
    )
    grp = (
        df2.groupby("regime")
        .agg(trades=("win","count"), win_rate=("win","mean"), total_pnl=("pnl","sum"))
        .reset_index()
    )
    colors = [GREEN if wr >= 0.5 else RED for wr in grp["win_rate"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["regime"], y=grp["win_rate"],
        marker_color=colors, opacity=0.8, name="Win Rate",
        text=[f"{wr:.0%}" for wr in grp["win_rate"]],
        textposition="outside", textfont=dict(color=WHITE, size=11),
    ))
    fig.add_hline(y=0.5, line_color="rgba(255,255,255,0.2)", line_dash="dash", line_width=1)
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text="Win Rate by Market Regime (from Notes)", font=dict(color=WHITE, size=13))
    layout["height"]  = 260
    layout["yaxis"]   = dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)",
                              tickformat=".0%", range=[0, 1.2])
    fig.update_layout(**layout)
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render_alpha_decay() -> None:
    st.markdown("## 🧬 Alpha Decay Tracker")

    with st.spinner("Loading trade journal…"):
        df, is_demo = _load_trades()

    if is_demo:
        st.info(
            "ℹ️ **Demo mode** — using synthetic trade data (60 trades with simulated alpha decay). "
            "To use real data, create `logs/journal.db` with a `trades` table.",
            icon="🧪",
        )
    else:
        st.caption(f"Loaded **{len(df)}** trades from `logs/journal.db`")

    if df.empty:
        st.warning("No trade data available.")
        return

    # ── Alpha decay banner ───────────────────────────────────────────────────
    decay_msg = _detect_decay(df)
    if decay_msg:
        st.error(f"🚨 **EDGE DECAYING ⚠️** — {decay_msg}", icon="⚠️")
    else:
        last10_wr = df.tail(_ROLL_WIN)["win"].mean()
        if last10_wr >= 0.55:
            st.success(f"✅ Edge healthy — last {_ROLL_WIN} trades win rate: **{last10_wr:.0%}**")
        else:
            st.warning(f"📉 Win rate softer recently — last {_ROLL_WIN} trades: **{last10_wr:.0%}**")

    # ── Summary metrics ──────────────────────────────────────────────────────
    total_trades  = len(df)
    overall_wr    = df["win"].mean()
    total_pnl     = df["pnl"].sum()
    avg_win       = df.loc[df["win"],  "pnl_pct"].mean()
    avg_loss      = df.loc[~df["win"], "pnl_pct"].mean()
    expectancy    = (overall_wr * avg_win) - ((1 - overall_wr) * abs(avg_loss))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades",    f"{total_trades}")
    c2.metric("Overall Win Rate", f"{overall_wr:.1%}")
    c3.metric("Total PnL",       f"₹{total_pnl:,.0f}")
    c4.metric("Avg Win / Loss",  f"{avg_win:.1f}% / {avg_loss:.1f}%")
    c5.metric("Expectancy",      f"{expectancy:.2f}%")

    st.markdown("---")

    # ── Rolling charts ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_win_rate_chart(df), width="stretch")
    with col2:
        st.plotly_chart(_expectancy_chart(df), width="stretch")

    # ── Setup breakdown ──────────────────────────────────────────────────────
    st.plotly_chart(_setup_breakdown_chart(df), width="stretch")

    # ── Monthly heatmap + regime ─────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        fig_heat = _monthly_pnl_heatmap(df)
        if fig_heat.data:
            st.plotly_chart(fig_heat, width="stretch")
        else:
            st.caption("Insufficient data for monthly heatmap.")
    with col4:
        fig_reg = _regime_chart(df)
        if fig_reg.data:
            st.plotly_chart(fig_reg, width="stretch")
        else:
            st.caption("No regime data in notes column.")

    # ── Raw trade log ────────────────────────────────────────────────────────
    with st.expander("📋 Raw trade log (last 20)"):
        show_cols = [c for c in ["date","symbol","action","pnl","pnl_pct","setup_type","notes","win"]
                     if c in df.columns]
        st.dataframe(df[show_cols].tail(20)[::-1].reset_index(drop=True),
                     width="stretch")
