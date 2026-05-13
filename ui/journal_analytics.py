"""Journal analytics — win rate, R:R, sector breakdown, drawdown."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── DB path — same DB used by paper_trading.py ────────────────────────────────
_DB_PATH = Path("logs/paper_trading.db")
_FALLBACK_DB = Path("paper_trading.db")  # legacy root-level location

# ── Shared plotly theme ────────────────────────────────────────────────────────
_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,28,0.6)",
    font=dict(color="#8892a4", size=11),
    margin=dict(l=0, r=0, t=28, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=10)),
    xaxis=dict(color="#8892a4", showgrid=False, zeroline=False),
    yaxis=dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)", zeroline=False),
)

_GREEN = "#00ff88"
_RED   = "#ff4466"
_CYAN  = "#00d4ff"
_AMBER = "#ffb800"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_closed_trades() -> pd.DataFrame:
    """Read CLOSED positions from the paper-trading SQLite DB.

    Returns a DataFrame with computed columns:
        return_pct  — % return relative to entry (sign-adjusted for direction)
        duration    — holding period in days
    Returns an empty DataFrame if the DB does not exist or has no closed rows.
    """
    db = _DB_PATH if _DB_PATH.exists() else (_FALLBACK_DB if _FALLBACK_DB.exists() else None)
    if db is None:
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db)
        df = pd.read_sql(
            """
            SELECT symbol, entry_price, exit_price, quantity, direction,
                   entry_date, exit_date, pnl
            FROM   positions
            WHERE  status = 'CLOSED'
               OR  exit_date IS NOT NULL
            ORDER  BY exit_date ASC
            """,
            conn,
        )
        conn.close()
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    # Coerce types
    for col in ("entry_price", "exit_price", "quantity", "pnl"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"]  = pd.to_datetime(df["exit_date"],  errors="coerce")

    # Return % — flip sign for SELL/SHORT trades
    raw_ret = (df["exit_price"] - df["entry_price"]) / df["entry_price"].replace(0, np.nan) * 100
    df["return_pct"] = np.where(
        df["direction"].str.upper().isin(["SELL", "SHORT"]),
        -raw_ret,
        raw_ret,
    )

    # Holding duration in days
    df["duration"] = (df["exit_date"] - df["entry_date"]).dt.days.fillna(0).astype(int)

    return df.dropna(subset=["pnl"])


# ── Helper: empty-state placeholder ───────────────────────────────────────────

def _render_empty_state() -> None:
    st.info("No closed trades yet. Paper trade to build your analytics.")

    # Flat ₹10L equity line as placeholder
    import datetime
    dates = pd.date_range(end=datetime.date.today(), periods=30, freq="D")
    fig = go.Figure(go.Scatter(
        x=dates, y=[1_000_000] * 30,
        mode="lines",
        line=dict(color=_CYAN, width=2, dash="dot"),
        name="Starting Capital ₹10L",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=200,
        title=dict(text="Equity Curve (placeholder)", font=dict(color="#8892a4", size=12)),
        yaxis=dict(tickprefix="₹", **{k: v for k, v in _LAYOUT_BASE["yaxis"].items()}),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="ja_empty_equity")


# ── Section renderers ──────────────────────────────────────────────────────────

def _section_kpis(df: pd.DataFrame) -> None:
    """Section 1 — Summary KPIs."""
    total   = len(df)
    wins    = df[df["pnl"] > 0]
    losses  = df[df["pnl"] <= 0]
    win_rate = len(wins) / total * 100 if total else 0.0

    avg_win  = wins["pnl"].mean()  if not wins.empty  else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0

    gross_profit = wins["pnl"].sum()
    gross_loss   = abs(losses["pnl"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    cumulative = df["pnl"].cumsum()
    peak       = cumulative.cummax()
    drawdown   = (cumulative - peak) / peak.replace(0, np.nan) * 100
    max_dd     = drawdown.min() if not drawdown.empty else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trades",   total)
    c2.metric("Win Rate",       f"{win_rate:.1f}%")

    avg_win_str  = f"₹{avg_win:,.0f}"  if avg_win  else "—"
    avg_loss_str = f"₹{avg_loss:,.0f}" if avg_loss else "—"
    c3.metric("Avg Win / Avg Loss", f"{avg_win_str} / {avg_loss_str}")

    pf_str = f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"
    c4.metric("Profit Factor",  pf_str)
    c5.metric("Max Drawdown",   f"{max_dd:.2f}%")


def _section_equity_curve(df: pd.DataFrame) -> None:
    """Section 2 — Equity Curve with peak & drawdown annotation."""
    st.markdown("#### Cumulative P&L")

    df_sorted  = df.sort_values("exit_date").reset_index(drop=True)
    cum_pnl    = df_sorted["pnl"].cumsum()
    dates      = df_sorted["exit_date"]

    peak_idx   = cum_pnl.idxmax()
    trough_val = cum_pnl.min()
    peak_val   = cum_pnl.max()

    # Colour: green above 0, red below 0
    line_color = _GREEN if cum_pnl.iloc[-1] >= 0 else _RED

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cum_pnl,
        mode="lines", name="Cumulative PnL",
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({'0,255,136' if line_color == _GREEN else '255,68,102'},0.07)",
    ))

    # Peak annotation
    fig.add_annotation(
        x=dates.iloc[peak_idx], y=peak_val,
        text=f"Peak ₹{peak_val:,.0f}",
        showarrow=True, arrowhead=2, arrowcolor=_AMBER,
        font=dict(color=_AMBER, size=10),
        bgcolor="rgba(0,0,0,0.5)", bordercolor=_AMBER, borderwidth=1,
    )

    # Max drawdown annotation (trough)
    trough_idx = cum_pnl.idxmin()
    if trough_idx != peak_idx:
        dd_pct = (trough_val - peak_val) / abs(peak_val) * 100 if peak_val != 0 else 0
        fig.add_annotation(
            x=dates.iloc[trough_idx], y=trough_val,
            text=f"DD {dd_pct:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor=_RED,
            font=dict(color=_RED, size=10),
            bgcolor="rgba(0,0,0,0.5)", bordercolor=_RED, borderwidth=1,
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        yaxis=dict(tickprefix="₹", **{k: v for k, v in _LAYOUT_BASE["yaxis"].items()}),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="ja_equity_curve")


def _section_win_loss_dist(df: pd.DataFrame) -> None:
    """Section 3 — Win/Loss Return % Distribution."""
    st.markdown("#### Return % Distribution")

    wins   = df[df["return_pct"] > 0]["return_pct"]
    losses = df[df["return_pct"] <= 0]["return_pct"]

    fig = go.Figure()
    if not wins.empty:
        fig.add_trace(go.Histogram(
            x=wins, name="Wins",
            marker_color=_GREEN, opacity=0.75,
            xbins=dict(size=0.5),
        ))
    if not losses.empty:
        fig.add_trace(go.Histogram(
            x=losses, name="Losses",
            marker_color=_RED, opacity=0.75,
            xbins=dict(size=0.5),
        ))

    fig.add_vline(x=0, line_color="#8892a4", line_dash="dash", line_width=1)

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="overlay",
        height=240,
        xaxis=dict(title="Return %", **{k: v for k, v in _LAYOUT_BASE["xaxis"].items()}),
        yaxis=dict(title="Trades", **{k: v for k, v in _LAYOUT_BASE["yaxis"].items()}),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="ja_win_loss_dist")


def _section_by_symbol(df: pd.DataFrame) -> None:
    """Section 4 — Top 10 Symbols by Total PnL."""
    st.markdown("#### P&L by Symbol (Top 10)")

    sym_df = (
        df.groupby("symbol")
        .agg(total_pnl=("pnl", "sum"), trade_count=("pnl", "count"))
        .reset_index()
        .sort_values("total_pnl", ascending=False)
        .head(10)
    )

    colors = [_GREEN if v >= 0 else _RED for v in sym_df["total_pnl"]]

    fig = go.Figure(go.Bar(
        x=sym_df["symbol"],
        y=sym_df["total_pnl"],
        marker_color=colors,
        text=[f"{c} trades" for c in sym_df["trade_count"]],
        textposition="outside",
        textfont=dict(color="#8892a4", size=9),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        yaxis=dict(tickprefix="₹", title="Total PnL",
                   **{k: v for k, v in _LAYOUT_BASE["yaxis"].items()}),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="ja_by_symbol")


def _section_monthly_pnl(df: pd.DataFrame) -> None:
    """Section 5 — Monthly PnL Heatmap (bar)."""
    st.markdown("#### Monthly P&L")

    df2 = df.copy()
    df2["ym"] = df2["exit_date"].dt.to_period("M").astype(str)
    monthly = df2.groupby("ym")["pnl"].sum().reset_index()
    monthly.columns = ["Month", "PnL"]
    monthly = monthly.sort_values("Month")

    colors = [_GREEN if v >= 0 else _RED for v in monthly["PnL"]]

    fig = go.Figure(go.Bar(
        x=monthly["Month"],
        y=monthly["PnL"],
        marker_color=colors,
        text=[f"₹{v:,.0f}" for v in monthly["PnL"]],
        textposition="outside",
        textfont=dict(color="#8892a4", size=9),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=240,
        yaxis=dict(tickprefix="₹", title="Net PnL",
                   **{k: v for k, v in _LAYOUT_BASE["yaxis"].items()}),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="ja_monthly_pnl")


def _section_duration_analysis(df: pd.DataFrame) -> None:
    """Section 6 — Trade Duration Analysis."""
    st.markdown("#### Holding Period (Days)")

    wins   = df[df["pnl"] > 0]["duration"]
    losses = df[df["pnl"] <= 0]["duration"]

    avg_win_dur  = wins.mean()   if not wins.empty   else 0.0
    avg_loss_dur = losses.mean() if not losses.empty else 0.0

    categories = []
    values     = []
    bar_colors = []

    if not wins.empty:
        categories.append("Winning Trades")
        values.append(avg_win_dur)
        bar_colors.append(_GREEN)
    if not losses.empty:
        categories.append("Losing Trades")
        values.append(avg_loss_dur)
        bar_colors.append(_RED)

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=bar_colors,
        text=[f"{v:.1f}d" for v in values],
        textposition="outside",
        textfont=dict(color="#e8eaf0", size=11),
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=220,
        yaxis=dict(title="Avg Days", **{k: v for k, v in _LAYOUT_BASE["yaxis"].items()}),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="ja_duration")


def _section_streak_analysis(df: pd.DataFrame) -> None:
    """Section 7 — Streak Analysis."""
    st.markdown("#### Streak Analysis")

    outcomes = (df.sort_values("exit_date")["pnl"] > 0).tolist()

    max_win_streak  = 0
    max_loss_streak = 0
    cur_win         = 0
    cur_loss        = 0
    current_streak_val  = 0
    current_streak_kind = "—"

    for win in outcomes:
        if win:
            cur_win  += 1
            cur_loss  = 0
        else:
            cur_loss += 1
            cur_win   = 0
        max_win_streak  = max(max_win_streak,  cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Current streak = trailing run
    if outcomes:
        last   = outcomes[-1]
        streak = 0
        for o in reversed(outcomes):
            if o == last:
                streak += 1
            else:
                break
        current_streak_val  = streak
        current_streak_kind = "Win" if last else "Loss"

    c1, c2, c3, c4 = st.columns(4)
    streak_color = _GREEN if current_streak_kind == "Win" else (_RED if current_streak_kind == "Loss" else "#8892a4")
    c1.metric("Current Streak",
              f"{current_streak_val} {current_streak_kind}" if current_streak_kind != "—" else "—")
    c2.metric("Max Win Streak",  max_win_streak)
    c3.metric("Max Loss Streak", max_loss_streak)
    c4.metric("Total Trades",    len(outcomes))


def _section_recent_trades(df: pd.DataFrame) -> None:
    """Section 8 — Recent Trades Table (last 20)."""
    st.markdown("#### Recent Closed Trades")

    recent = (
        df.sort_values("exit_date", ascending=False)
        .head(20)
        .copy()
    )

    recent["Entry"]     = recent["entry_price"].map(lambda x: f"₹{x:,.2f}")
    recent["Exit"]      = recent["exit_price"].map(lambda x: f"₹{x:,.2f}")
    recent["Return %"]  = recent["return_pct"].map(lambda x: f"{x:+.2f}%")
    recent["P&L"]       = recent["pnl"].map(lambda x: f"₹{x:+,.0f}")
    recent["Duration"]  = recent["duration"].map(lambda x: f"{x}d")
    recent["Exit Date"] = recent["exit_date"].dt.strftime("%Y-%m-%d")

    display = recent[["symbol", "direction", "Entry", "Exit", "Return %", "P&L", "Duration", "Exit Date"]].rename(
        columns={"symbol": "Symbol", "direction": "Dir"}
    )

    # Build HTML table with coloured rows
    rows_html = ""
    for _, row in display.iterrows():
        pnl_val  = float(row["P&L"].replace("₹", "").replace(",", "").replace("+", ""))
        bg_color = "rgba(0,255,136,0.05)" if pnl_val >= 0 else "rgba(255,68,102,0.05)"
        pnl_color = _GREEN if pnl_val >= 0 else _RED
        ret_color = _GREEN if "+" in row["Return %"] else _RED

        cells = "".join(
            f"<td style='padding:.4rem .6rem;border-bottom:1px solid rgba(255,255,255,0.05);color:"
            + (pnl_color if col == "P&L" else (ret_color if col == "Return %" else "#e8eaf0"))
            + f";font-family:JetBrains Mono,monospace;font-size:.78rem'>{val}</td>"
            for col, val in row.items()
        )
        rows_html += f"<tr style='background:{bg_color}'>{cells}</tr>"

    headers_html = "".join(
        f"<th style='padding:.4rem .6rem;color:#8892a4;font-size:.7rem;"
        f"text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid rgba(255,255,255,0.1)'>{col}</th>"
        for col in display.columns
    )

    table_html = f"""
    <div style='overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,0.08);
                background:rgba(255,255,255,0.02);'>
      <table style='width:100%;border-collapse:collapse'>
        <thead><tr style='background:rgba(255,255,255,0.04)'>{headers_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ── Main entry point ───────────────────────────────────────────────────────────

def render_journal_analytics() -> None:
    """Render all journal analytics sections below the main journal."""
    st.markdown("---")
    st.markdown("### 📊 Enhanced Trade Analytics")

    df = load_closed_trades()

    if df.empty:
        _render_empty_state()
        return

    # Section 1 — KPIs
    _section_kpis(df)

    st.markdown("")  # spacer

    # Sections 2 & 3 side-by-side
    col_left, col_right = st.columns([3, 2])
    with col_left:
        _section_equity_curve(df)
    with col_right:
        _section_win_loss_dist(df)

    # Sections 4 & 5
    col_a, col_b = st.columns(2)
    with col_a:
        _section_by_symbol(df)
    with col_b:
        _section_monthly_pnl(df)

    # Sections 6 & 7
    col_c, col_d = st.columns([2, 3])
    with col_c:
        _section_duration_analysis(df)
    with col_d:
        _section_streak_analysis(df)

    st.markdown("")

    # Section 8 — Recent trades
    _section_recent_trades(df)
