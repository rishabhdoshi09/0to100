"""Trade Replay Engine — frame-by-frame candlestick replay with P&L tracker."""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Theme ──────────────────────────────────────────────────────────────────────
_NAVY   = "#0a0e1a"
_CYAN   = "#00d4ff"
_GREEN  = "#00ff88"
_RED    = "#ff4466"
_AMBER  = "#ffb800"
_WHITE  = "#e8eaf0"
_MUTED  = "#8892a4"

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,28,0.8)",
    font=dict(color=_MUTED, size=11),
    margin=dict(l=0, r=0, t=36, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED, size=10)),
    xaxis=dict(color=_MUTED, showgrid=False, zeroline=False),
    yaxis=dict(color=_MUTED, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
)

# ── Paths ──────────────────────────────────────────────────────────────────────
_JOURNAL_DB   = Path("logs/journal.db")
_FALLBACK_DB  = Path("journal.db")
_LESSONS_FILE = Path("logs/replay_lessons.json")

# ── NSE suffix ─────────────────────────────────────────────────────────────────
def _nse(sym: str) -> str:
    s = sym.strip().upper()
    return s if s.endswith(".NS") else s + ".NS"


# ── Demo trades ────────────────────────────────────────────────────────────────
def _demo_trades() -> pd.DataFrame:
    random.seed(7)
    base = datetime(2024, 3, 1)
    data = [
        {
            "id": 1, "symbol": "RELIANCE", "action": "BUY",
            "entry_price": 2850.0, "exit_price": 2940.0,
            "pnl": 9000.0, "pnl_pct": 3.16, "setup_type": "Breakout",
            "date": (base + timedelta(days=0)).strftime("%Y-%m-%d"),
            "exit_date": (base + timedelta(days=7)).strftime("%Y-%m-%d"),
            "notes": "Confidence: 78% | RSI breakout above 60",
        },
        {
            "id": 2, "symbol": "INFY", "action": "BUY",
            "entry_price": 1480.0, "exit_price": 1420.0,
            "pnl": -6000.0, "pnl_pct": -4.05, "setup_type": "Mean Reversion",
            "date": (base + timedelta(days=15)).strftime("%Y-%m-%d"),
            "exit_date": (base + timedelta(days=22)).strftime("%Y-%m-%d"),
            "notes": "Confidence: 65% | Oversold bounce failed",
        },
        {
            "id": 3, "symbol": "HDFCBANK", "action": "BUY",
            "entry_price": 1650.0, "exit_price": 1730.0,
            "pnl": 8000.0, "pnl_pct": 4.85, "setup_type": "Momentum",
            "date": (base + timedelta(days=30)).strftime("%Y-%m-%d"),
            "exit_date": (base + timedelta(days=40)).strftime("%Y-%m-%d"),
            "notes": "Confidence: 82% | Strong volume surge",
        },
        {
            "id": 4, "symbol": "TCS", "action": "BUY",
            "entry_price": 3800.0, "exit_price": 3950.0,
            "pnl": 15000.0, "pnl_pct": 3.95, "setup_type": "VWAP Bounce",
            "date": (base + timedelta(days=45)).strftime("%Y-%m-%d"),
            "exit_date": (base + timedelta(days=58)).strftime("%Y-%m-%d"),
            "notes": "Confidence: 74% | VWAP support held",
        },
        {
            "id": 5, "symbol": "SBIN", "action": "BUY",
            "entry_price": 760.0, "exit_price": 800.0,
            "pnl": 4000.0, "pnl_pct": 5.26, "setup_type": "Gap Fill",
            "date": (base + timedelta(days=60)).strftime("%Y-%m-%d"),
            "exit_date": (base + timedelta(days=75)).strftime("%Y-%m-%d"),
            "notes": "Confidence: 71% | Gap up follow-through",
        },
    ]
    return pd.DataFrame(data)


# ── Load trades ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _load_trades() -> tuple[pd.DataFrame, bool]:
    db = _JOURNAL_DB if _JOURNAL_DB.exists() else (
        _FALLBACK_DB if _FALLBACK_DB.exists() else None
    )
    if db is not None:
        try:
            conn = sqlite3.connect(db)
            df = pd.read_sql("SELECT * FROM trades", conn)
            conn.close()
            if not df.empty:
                return df, False
        except Exception:
            pass
    return _demo_trades(), True


# ── Fetch OHLCV ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        ticker = yf.Ticker(_nse(symbol))
        df = ticker.history(start=start, end=end, interval="1d")
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        return df
    except Exception:
        return pd.DataFrame()


def _synthetic_ohlcv(entry_price: float, n_days: int, seed: int = 42) -> pd.DataFrame:
    """Generate plausible synthetic OHLCV when yfinance fails."""
    np.random.seed(seed)
    dates = pd.date_range(datetime.today() - timedelta(days=n_days + 10), periods=n_days)
    prices = [entry_price]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.012)))
    prices = np.array(prices)
    df = pd.DataFrame({
        "Open":   prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        "High":   prices * (1 + np.abs(np.random.normal(0, 0.008, n_days))),
        "Low":    prices * (1 - np.abs(np.random.normal(0, 0.008, n_days))),
        "Close":  prices,
        "Volume": np.random.randint(500_000, 5_000_000, n_days).astype(float),
    }, index=dates)
    return df


# ── EMA helper ────────────────────────────────────────────────────────────────
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ── Lessons persistence ────────────────────────────────────────────────────────
def _load_lessons() -> dict:
    if _LESSONS_FILE.exists():
        try:
            return json.loads(_LESSONS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_lesson(trade_id: str, lesson: str) -> None:
    _LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = _load_lessons()
    data[trade_id] = {
        "lesson": lesson,
        "saved_at": datetime.now().isoformat(),
    }
    _LESSONS_FILE.write_text(json.dumps(data, indent=2))


# ── Candlestick chart builder ─────────────────────────────────────────────────
def _build_chart(
    df_slice: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    symbol: str,
    replay_day: int,
) -> go.Figure:
    """Build a candlestick + EMA + volume subplot for the given slice."""
    df_vis = df_slice.iloc[: replay_day + 1].copy()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_vis.index,
            open=df_vis["Open"],
            high=df_vis["High"],
            low=df_vis["Low"],
            close=df_vis["Close"],
            increasing_line_color=_GREEN,
            decreasing_line_color=_RED,
            increasing_fillcolor=f"rgba(0,255,136,0.25)",
            decreasing_fillcolor=f"rgba(255,68,102,0.25)",
            name=symbol,
            showlegend=False,
        ),
        row=1, col=1,
    )

    # EMA 20
    if len(df_vis) >= 3:
        ema20 = _ema(df_vis["Close"], 20)
        fig.add_trace(
            go.Scatter(
                x=df_vis.index, y=ema20,
                mode="lines", name="EMA 20",
                line=dict(color=_CYAN, width=1.5, dash="solid"),
            ),
            row=1, col=1,
        )

    # EMA 50
    if len(df_vis) >= 5:
        ema50 = _ema(df_vis["Close"], 50)
        fig.add_trace(
            go.Scatter(
                x=df_vis.index, y=ema50,
                mode="lines", name="EMA 50",
                line=dict(color=_AMBER, width=1.5, dash="dot"),
            ),
            row=1, col=1,
        )

    # Entry arrow
    if entry_date in df_vis.index:
        fig.add_trace(
            go.Scatter(
                x=[entry_date],
                y=[df_vis.loc[entry_date, "Low"] * 0.995],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=16,
                    color=_GREEN,
                    line=dict(color=_GREEN, width=2),
                ),
                text=["ENTRY"],
                textposition="bottom center",
                textfont=dict(color=_GREEN, size=9),
                name="Entry",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Exit arrow (only show after exit day)
    if exit_date in df_vis.index:
        fig.add_trace(
            go.Scatter(
                x=[exit_date],
                y=[df_vis.loc[exit_date, "High"] * 1.005],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-down",
                    size=16,
                    color=_RED,
                    line=dict(color=_RED, width=2),
                ),
                text=["EXIT"],
                textposition="top center",
                textfont=dict(color=_RED, size=9),
                name="Exit",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Volume bars
    vol_colors = [
        _GREEN if c >= o else _RED
        for o, c in zip(df_vis["Open"], df_vis["Close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df_vis.index,
            y=df_vis["Volume"],
            marker_color=vol_colors,
            opacity=0.7,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text=f"{symbol} — Trade Replay (Day {replay_day + 1} of {len(df_slice)})",
            font=dict(color=_WHITE, size=14),
        ),
        xaxis_rangeslider_visible=False,
        height=520,
        yaxis=dict(color=_MUTED, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis2=dict(color=_MUTED, gridcolor="rgba(255,255,255,0.03)", zeroline=False),
    )
    return fig


# ── Main render ────────────────────────────────────────────────────────────────
def render_trade_replay() -> None:
    st.markdown(
        """
        <style>
        .replay-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1rem 1.4rem;
            margin-bottom: 0.8rem;
        }
        .replay-label {
            color: #8892a4;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: .06em;
        }
        .replay-val {
            color: #e8eaf0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.2rem;
            font-weight: 700;
        }
        .signal-box {
            background: rgba(0,212,255,0.06);
            border: 1px solid rgba(0,212,255,0.18);
            border-radius: 10px;
            padding: .7rem 1rem;
            margin-bottom: .6rem;
        }
        .whatif-box {
            background: rgba(255,184,0,0.06);
            border: 1px solid rgba(255,184,0,0.2);
            border-radius: 10px;
            padding: .7rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🎬 Trade Replay Engine")
    st.markdown(
        "<span style='color:#8892a4;font-size:.85rem;'>"
        "Frame-by-frame replay · Entry/exit overlays · What-if analysis · Lesson capture"
        "</span>",
        unsafe_allow_html=True,
    )

    trades, is_demo = _load_trades()
    if is_demo:
        st.info("Using demo trades — connect `logs/journal.db` for live data.", icon="ℹ️")

    if trades.empty:
        st.warning("No trades available for replay.")
        return

    # Ensure date cols
    for col in ["entry_price", "exit_price", "pnl", "pnl_pct"]:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")

    # Trade selector
    trade_labels = []
    for _, r in trades.iterrows():
        sym    = r.get("symbol", "?")
        date   = str(r.get("date", ""))[:10]
        pnl_p  = r.get("pnl_pct", 0)
        sign   = "+" if float(pnl_p or 0) >= 0 else ""
        label  = f"#{r.get('id','?')}  {sym}  {date}  ({sign}{float(pnl_p or 0):.2f}%)"
        trade_labels.append(label)

    selected_label = st.selectbox(
        "Select a trade to replay",
        trade_labels,
        help="Choose any past trade to replay frame-by-frame.",
    )
    sel_idx   = trade_labels.index(selected_label)
    trade     = trades.iloc[sel_idx]

    symbol      = str(trade.get("symbol", "RELIANCE"))
    entry_price = float(trade.get("entry_price", 0) or 0)
    exit_price  = float(trade.get("exit_price", 0) or 0)
    pnl_pct     = float(trade.get("pnl_pct", 0) or 0)
    setup_type  = str(trade.get("setup_type", "Unknown"))
    notes       = str(trade.get("notes", ""))
    trade_id    = str(trade.get("id", sel_idx))

    # Parse dates
    date_str      = str(trade.get("date", ""))[:10]
    exit_date_str = str(trade.get("exit_date", ""))[:10]

    try:
        entry_dt = pd.Timestamp(date_str)
    except Exception:
        entry_dt = pd.Timestamp.today() - timedelta(days=30)

    try:
        exit_dt = pd.Timestamp(exit_date_str) if exit_date_str and exit_date_str != "nan" else entry_dt + timedelta(days=10)
    except Exception:
        exit_dt = entry_dt + timedelta(days=10)

    fetch_start = (entry_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    fetch_end   = (exit_dt  + timedelta(days=8)).strftime("%Y-%m-%d")

    # Fetch OHLCV
    with st.spinner(f"Fetching {symbol} OHLCV data…"):
        df_full = _fetch_ohlcv(symbol, fetch_start, fetch_end)

    if df_full.empty:
        st.warning(
            f"Could not fetch data for {symbol} from yfinance — using synthetic OHLCV.",
            icon="⚠️",
        )
        hold_days = max(5, (exit_dt - entry_dt).days)
        df_full = _synthetic_ohlcv(entry_price, hold_days + 14, seed=sel_idx)
        entry_dt = df_full.index[5]
        exit_dt  = df_full.index[5 + hold_days]

    # Align entry / exit to available trading days
    available_dates = df_full.index
    entry_date = min(available_dates, key=lambda d: abs((d - entry_dt).days))
    exit_date  = min(available_dates, key=lambda d: abs((d - exit_dt).days))

    # Build display slice: 5 bars before entry, all hold bars, 5 bars after exit
    entry_pos = list(available_dates).index(entry_date)
    exit_pos  = list(available_dates).index(exit_date)
    start_pos = max(0, entry_pos - 5)
    end_pos   = min(len(df_full), exit_pos + 6)
    df_slice  = df_full.iloc[start_pos:end_pos].copy()
    n_days    = len(df_slice)

    if n_days == 0:
        st.warning("Insufficient OHLCV data for this trade period.")
        return

    # Re-compute local indices after slicing
    slice_dates = list(df_slice.index)
    entry_idx   = slice_dates.index(entry_date) if entry_date in slice_dates else 5
    exit_idx    = slice_dates.index(exit_date)  if exit_date  in slice_dates else n_days - 1

    st.divider()

    # ── Trade metadata ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    pnl_color = _GREEN if pnl_pct >= 0 else _RED
    for col, label, val in [
        (c1, "Symbol",     symbol),
        (c2, "Setup",      setup_type),
        (c3, "Entry",      f"₹{entry_price:,.2f}"),
        (c4, "PnL %",      f"{pnl_pct:+.2f}%"),
    ]:
        color = pnl_color if label == "PnL %" else _CYAN
        col.markdown(
            f"""<div class="replay-card">
            <div class="replay-label">{label}</div>
            <div class="replay-val" style="color:{color};">{val}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Signal box ────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div class="signal-box">
        <span style="color:{_CYAN};font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">
        Signal Annotations</span><br>
        <span style="color:{_WHITE};font-size:.88rem;">{notes or "No notes recorded."}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Replay slider ─────────────────────────────────────────────────────────
    replay_day = st.slider(
        "Replay — Day N of trade",
        min_value=0,
        max_value=n_days - 1,
        value=entry_idx,
        step=1,
        help="Scrub through the trade day-by-day",
    )

    # Running P&L
    current_close = float(df_slice["Close"].iloc[replay_day])
    running_pnl_pct = ((current_close - entry_price) / entry_price * 100) if entry_price else 0
    current_date_label = df_slice.index[replay_day].strftime("%d %b %Y")

    pnl_col = _GREEN if running_pnl_pct >= 0 else _RED
    pnl_sign = "+" if running_pnl_pct >= 0 else ""

    col_pnl, col_price, col_day = st.columns(3)
    col_pnl.markdown(
        f"""<div class="replay-card">
        <div class="replay-label">Running P&L (vs entry)</div>
        <div class="replay-val" style="color:{pnl_col};">{pnl_sign}{running_pnl_pct:.2f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
    col_price.markdown(
        f"""<div class="replay-card">
        <div class="replay-label">Current Close</div>
        <div class="replay-val">₹{current_close:,.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    col_day.markdown(
        f"""<div class="replay-card">
        <div class="replay-label">Date</div>
        <div class="replay-val">{current_date_label}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Candlestick chart ─────────────────────────────────────────────────────
    fig = _build_chart(
        df_slice, entry_date, exit_date,
        entry_price, exit_price,
        symbol, replay_day,
    )
    st.plotly_chart(fig, width="stretch")

    # ── P&L equity curve ──────────────────────────────────────────────────────
    pnl_curve = ((df_slice["Close"] - entry_price) / entry_price * 100).values
    pnl_colors = [_GREEN if v >= 0 else _RED for v in pnl_curve[: replay_day + 1]]

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=list(range(replay_day + 1)),
        y=pnl_curve[: replay_day + 1],
        mode="lines",
        fill="tozeroy",
        line=dict(color=_CYAN, width=2),
        fillcolor="rgba(0,212,255,0.08)",
        name="Running PnL %",
    ))
    fig_pnl.add_hline(y=0, line_color=_WHITE, line_dash="dash", line_width=1)
    if replay_day < len(pnl_curve):
        fig_pnl.add_trace(go.Scatter(
            x=[replay_day],
            y=[pnl_curve[replay_day]],
            mode="markers",
            marker=dict(size=10, color=pnl_col),
            name="Current",
            showlegend=False,
        ))
    fig_pnl.update_layout(
        **_LAYOUT,
        title=dict(text="Running P&L % curve", font=dict(color=_WHITE, size=13)),
        xaxis_title="Day",
        yaxis_title="PnL %",
        height=220,
    )
    st.plotly_chart(fig_pnl, width="stretch")

    # ── What-if panel ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🤔 What-If Analysis")

    wi_col1, wi_col2 = st.columns(2)
    for days_shift, col, label in [(-2, wi_col1, "2 days earlier"), (+2, wi_col2, "2 days later")]:
        alt_exit_idx = max(0, min(n_days - 1, exit_idx + days_shift))
        alt_exit_price = float(df_slice["Close"].iloc[alt_exit_idx])
        alt_pnl = ((alt_exit_price - entry_price) / entry_price * 100) if entry_price else 0
        diff = alt_pnl - pnl_pct
        diff_color = _GREEN if diff >= 0 else _RED
        col.markdown(
            f"""<div class="whatif-box">
            <span style="color:{_AMBER};font-size:.75rem;font-weight:600;
            text-transform:uppercase;">Exit {label}</span><br>
            <span style="color:{_WHITE};font-size:1.1rem;font-weight:700;">
            ₹{alt_exit_price:,.2f}</span><br>
            <span style="color:{_MUTED};font-size:.8rem;">PnL: </span>
            <span style="color:{diff_color};font-size:.9rem;font-weight:700;">
            {alt_pnl:+.2f}%</span>
            <span style="color:{diff_color};font-size:.78rem;"> ({diff:+.2f}% vs actual)</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Custom what-if
    st.markdown("##### Custom exit day")
    custom_day = st.slider(
        "What if you exited on day…",
        min_value=0,
        max_value=n_days - 1,
        value=exit_idx,
        key="whatif_slider",
    )
    custom_price = float(df_slice["Close"].iloc[custom_day])
    custom_pnl   = ((custom_price - entry_price) / entry_price * 100) if entry_price else 0
    custom_diff  = custom_pnl - pnl_pct
    custom_color = _GREEN if custom_diff >= 0 else _RED
    st.markdown(
        f"""<div class="whatif-box">
        <span style="color:{_AMBER};font-weight:600;">Day {custom_day} exit:</span>
        ₹{custom_price:,.2f} →
        <span style="color:{custom_color};font-weight:700;">{custom_pnl:+.2f}%</span>
        <span style="color:{_MUTED};font-size:.82rem;">(Δ {custom_diff:+.2f}% vs actual exit)</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Lesson capture ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📝 Capture a Lesson")

    lessons = _load_lessons()
    existing_lesson = lessons.get(trade_id, {}).get("lesson", "")

    lesson_text = st.text_area(
        "What did you learn from this trade?",
        value=existing_lesson,
        placeholder="e.g. I chased the breakout instead of waiting for a pullback…",
        height=120,
        key=f"lesson_{trade_id}",
    )
    save_col, status_col = st.columns([1, 3])
    with save_col:
        if st.button("💾 Save Lesson", key=f"save_lesson_{trade_id}"):
            if lesson_text.strip():
                _save_lesson(trade_id, lesson_text.strip())
                status_col.success("Lesson saved to `logs/replay_lessons.json`")
            else:
                status_col.warning("Write a lesson first.")

    # Show all saved lessons
    if lessons:
        with st.expander("📚 All Saved Lessons"):
            for tid, entry in sorted(lessons.items(), key=lambda x: x[1].get("saved_at",""), reverse=True):
                saved_at = entry.get("saved_at","")[:10]
                st.markdown(
                    f"""<div style="background:rgba(255,255,255,0.03);border-left:3px solid
                    {_CYAN};border-radius:0 8px 8px 0;padding:.5rem .9rem;margin:.3rem 0;">
                    <span style="color:{_MUTED};font-size:.72rem;">Trade #{tid} · {saved_at}</span><br>
                    <span style="color:{_WHITE};font-size:.85rem;">{entry.get('lesson','')}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
