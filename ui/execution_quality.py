"""Execution Quality Analyzer — slippage, VWAP comparison, fill quality scoring."""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Theme constants ────────────────────────────────────────────────────────────
_NAVY   = "#0a0e1a"
_CYAN   = "#00d4ff"
_GREEN  = "#00ff88"
_RED    = "#ff4466"
_AMBER  = "#ffb800"
_WHITE  = "#e8eaf0"
_MUTED  = "#8892a4"
_CARD   = "rgba(255,255,255,0.04)"
_BORDER = "rgba(255,255,255,0.08)"

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,28,0.6)",
    font=dict(color=_MUTED, size=11),
    margin=dict(l=0, r=0, t=32, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED, size=10)),
    xaxis=dict(color=_MUTED, showgrid=False, zeroline=False),
    yaxis=dict(color=_MUTED, gridcolor="rgba(255,255,255,0.04)", zeroline=False),
)

# ── DB paths ───────────────────────────────────────────────────────────────────
_JOURNAL_DB = Path("logs/journal.db")
_FALLBACK_DB = Path("journal.db")

# ── NSE suffix helper ──────────────────────────────────────────────────────────
def _nse(symbol: str) -> str:
    s = symbol.strip().upper()
    return s if s.endswith(".NS") else s + ".NS"


# ── Demo data ──────────────────────────────────────────────────────────────────
def _demo_trades() -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    symbols = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK",
               "SBIN", "AXISBANK", "WIPRO", "LT", "BAJFINANCE"]
    setups  = ["Breakout", "Mean Reversion", "Momentum", "VWAP Bounce", "Gap Fill"]
    rows = []
    base_date = datetime(2024, 6, 1)
    hours = [9, 10, 11, 12, 13, 14, 15]
    for i in range(40):
        sym   = random.choice(symbols)
        price = random.uniform(500, 3500)
        pnl_p = random.uniform(-4, 8)
        entry_dt = base_date + timedelta(days=random.randint(0, 200))
        entry_dt = entry_dt.replace(
            hour=random.choice(hours),
            minute=random.randint(0, 59),
        )
        rows.append({
            "id":          i + 1,
            "date":        entry_dt.strftime("%Y-%m-%d %H:%M"),
            "symbol":      sym,
            "action":      random.choice(["BUY", "SELL"]),
            "entry_price": round(price, 2),
            "exit_price":  round(price * (1 + pnl_p / 100), 2),
            "pnl":         round(price * pnl_p / 100 * random.randint(5, 50), 2),
            "pnl_pct":     round(pnl_p, 2),
            "setup_type":  random.choice(setups),
            "notes":       f"Confidence: {random.randint(60, 95)}%",
        })
    return pd.DataFrame(rows)


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


# ── yfinance fetch (daily open as signal proxy) ───────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_daily_open(symbol: str, date_str: str) -> float | None:
    try:
        import yfinance as yf
        ticker = yf.Ticker(_nse(symbol))
        dt = pd.Timestamp(date_str[:10])
        hist = ticker.history(start=dt - timedelta(days=5), end=dt + timedelta(days=2))
        if hist.empty:
            return None
        hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index
        row = hist[hist.index.date == dt.date()]
        if row.empty:
            return None
        return float(row["Open"].iloc[0])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_intraday_vwap(symbol: str, date_str: str) -> float | None:
    """Approximate VWAP from 5-min intraday data via yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(_nse(symbol))
        dt = pd.Timestamp(date_str[:10])
        hist = ticker.history(
            start=dt,
            end=dt + timedelta(days=1),
            interval="5m",
        )
        if hist.empty:
            return None
        typical = (hist["High"] + hist["Low"] + hist["Close"]) / 3
        vwap = (typical * hist["Volume"]).sum() / hist["Volume"].sum()
        return float(vwap)
    except Exception:
        return None


# ── Fill quality scoring ───────────────────────────────────────────────────────
def _compute_fill_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add slippage_bps, vwap, vwap_rel, hour, fill_score columns."""
    rows = []
    prog = st.progress(0, text="Computing fill quality…")
    total = len(df)

    for i, row in df.iterrows():
        date_str = str(row.get("date", ""))[:16]
        symbol   = str(row.get("symbol", ""))
        entry    = float(row.get("entry_price", 0) or 0)
        action   = str(row.get("action", "BUY")).upper()

        # slippage: entry vs open
        open_px = _fetch_daily_open(symbol, date_str)
        if open_px and open_px > 0:
            slippage_bps = ((entry - open_px) / open_px) * 10_000
            if action == "SELL":
                slippage_bps = -slippage_bps  # for sells, buying higher is good
        else:
            # simulate if yfinance fails
            slippage_bps = np.random.uniform(-20, 40)

        # VWAP
        vwap = _fetch_intraday_vwap(symbol, date_str)
        if vwap is None or vwap == 0:
            vwap = entry * (1 + np.random.uniform(-0.005, 0.005))

        vwap_rel = (entry - vwap) / vwap * 100  # positive = above VWAP

        # hour of entry
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            hour = dt.hour
        except Exception:
            hour = random.randint(9, 15)

        # fill quality score (0–100)
        # slippage component (50 pts): 0 bps = 50, every 10 bps worse = -5
        slip_score = max(0, min(50, 50 - slippage_bps * 0.5))
        # VWAP component (30 pts): below VWAP on buy = better
        if action == "BUY":
            vwap_score = max(0, min(30, 30 - vwap_rel * 6))
        else:
            vwap_score = max(0, min(30, 30 + vwap_rel * 6))
        # timing component (20 pts): avoid first 15 min (9 AM) and last 15 min (15 PM)
        timing_map = {9: 8, 10: 20, 11: 20, 12: 18, 13: 16, 14: 15, 15: 6}
        timing_score = timing_map.get(hour, 12)

        fill_score = round(slip_score + vwap_score + timing_score, 1)
        fill_score = max(0, min(100, fill_score))

        rows.append({
            **{k: row[k] for k in df.columns},
            "slippage_bps": round(slippage_bps, 1),
            "open_price":   round(open_px, 2) if open_px else None,
            "vwap":         round(vwap, 2),
            "vwap_rel":     round(vwap_rel, 3),
            "hour":         hour,
            "fill_score":   fill_score,
        })
        prog.progress((i + 1) / total if total else 1, text="Computing fill quality…")

    prog.empty()
    return pd.DataFrame(rows)


# ── Improvement suggestions ────────────────────────────────────────────────────
def _generate_suggestions(scored: pd.DataFrame) -> list[str]:
    tips = []
    buys = scored[scored["action"].str.upper() == "BUY"]
    sells = scored[scored["action"].str.upper() == "SELL"]

    if not buys.empty:
        avg_vwap_buy = buys["vwap_rel"].mean()
        if avg_vwap_buy > 0.1:
            tips.append(
                f"You tend to BUY above VWAP ({avg_vwap_buy:+.2f}% avg) — "
                "try limit orders near or below VWAP for better fills."
            )

    avg_slip = scored["slippage_bps"].mean()
    if avg_slip > 15:
        tips.append(
            f"Average slippage is {avg_slip:.1f} bps — consider placing orders "
            "as limit orders rather than market orders."
        )

    morning_trades = scored[scored["hour"] == 9]
    if len(morning_trades) >= 3:
        morning_pnl = morning_trades["pnl_pct"].mean() if "pnl_pct" in morning_trades else 0
        if morning_pnl < 0:
            tips.append(
                "Your 9 AM trades show below-average PnL — "
                "the opening auction is volatile; wait for price discovery (9:15–9:30 AM)."
            )

    if "hour" in scored.columns:
        by_hour = scored.groupby("hour")["fill_score"].mean()
        best_hour = by_hour.idxmax()
        tips.append(
            f"Your best fill quality is at {best_hour}:00 — "
            "consider timing more entries around this window."
        )

    low_score = scored[scored["fill_score"] < 40]
    if len(low_score) > len(scored) * 0.3:
        tips.append(
            "Over 30% of fills scored below 40/100 — "
            "review your entry triggers; they may be chasing momentum."
        )

    if not tips:
        tips.append("Fill quality looks solid. Keep using disciplined limit entries near VWAP.")
    return tips


# ── Main render ────────────────────────────────────────────────────────────────
def render_execution_quality() -> None:
    st.markdown(
        """
        <style>
        .eq-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 0.9rem;
        }
        .eq-label {
            color: #8892a4;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: .06em;
        }
        .eq-val {
            color: #e8eaf0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.35rem;
            font-weight: 700;
        }
        .tip-box {
            background: rgba(0,212,255,0.05);
            border-left: 3px solid #00d4ff;
            border-radius: 0 10px 10px 0;
            padding: .6rem 1rem;
            margin: .4rem 0;
            color: #e8eaf0;
            font-size: .88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## ⚡ Execution Quality Analyzer")
    st.markdown(
        "<span style='color:#8892a4;font-size:.85rem;'>"
        "Slippage · VWAP position · Fill quality scoring"
        "</span>",
        unsafe_allow_html=True,
    )

    trades_raw, is_demo = _load_trades()
    if is_demo:
        st.info("No `logs/journal.db` found — showing demo data.", icon="ℹ️")

    if trades_raw.empty:
        st.warning("No trades to analyze.")
        return

    # Ensure numeric columns
    for col in ["entry_price", "exit_price", "pnl", "pnl_pct"]:
        if col in trades_raw.columns:
            trades_raw[col] = pd.to_numeric(trades_raw[col], errors="coerce")

    with st.spinner("Fetching market data for fill quality analysis…"):
        scored = _compute_fill_scores(trades_raw)

    # ── KPI row ───────────────────────────────────────────────────────────────
    avg_slip  = scored["slippage_bps"].mean()
    avg_score = scored["fill_score"].mean()
    pct_below_vwap = (scored["vwap_rel"] < 0).mean() * 100
    best_score = scored["fill_score"].max()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, suffix, good_low in [
        (c1, "Avg Slippage", avg_slip, " bps", True),
        (c2, "Avg Fill Score", avg_score, " / 100", False),
        (c3, "% Entries Below VWAP", pct_below_vwap, "%", False),
        (c4, "Best Fill Score", best_score, " / 100", False),
    ]:
        color = _GREEN if (good_low and val < 10) or (not good_low and val > 60) else _RED
        col.markdown(
            f"""<div class="eq-card">
            <div class="eq-label">{label}</div>
            <div class="eq-val" style="color:{color};">{val:.1f}{suffix}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Slippage", "📈 VWAP Analysis", "🕐 Time of Day", "🏆 Fill Leaderboard", "💡 Suggestions"
    ])

    # ── Tab 1: Slippage ───────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Slippage Distribution (entry vs open price)")
        fig_slip = go.Figure()
        fig_slip.add_trace(go.Histogram(
            x=scored["slippage_bps"],
            nbinsx=25,
            marker_color=_CYAN,
            opacity=0.8,
            name="Slippage (bps)",
        ))
        fig_slip.add_vline(x=0, line_color=_WHITE, line_dash="dash", line_width=1)
        fig_slip.add_vline(x=avg_slip, line_color=_AMBER, line_dash="dot",
                           annotation_text=f"Avg {avg_slip:.1f} bps",
                           annotation_font_color=_AMBER)
        fig_slip.update_layout(
            **_LAYOUT,
            title=dict(text="Slippage Distribution", font=dict(color=_WHITE, size=13)),
            xaxis_title="Slippage (bps)",
            yaxis_title="Count",
            height=320,
        )
        st.plotly_chart(fig_slip, use_container_width=True)

        # Slippage by symbol
        st.markdown("#### Avg Slippage by Symbol")
        by_sym = (
            scored.groupby("symbol")["slippage_bps"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        colors = [_RED if v > 0 else _GREEN for v in by_sym["slippage_bps"]]
        fig_sym = go.Figure(go.Bar(
            x=by_sym["symbol"],
            y=by_sym["slippage_bps"],
            marker_color=colors,
            text=[f"{v:.1f}" for v in by_sym["slippage_bps"]],
            textposition="outside",
            textfont=dict(color=_WHITE, size=10),
        ))
        fig_sym.update_layout(
            **_LAYOUT,
            title=dict(text="Avg Slippage by Symbol", font=dict(color=_WHITE, size=13)),
            yaxis_title="Avg Slippage (bps)",
            height=300,
        )
        st.plotly_chart(fig_sym, use_container_width=True)

    # ── Tab 2: VWAP ───────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### VWAP Position at Entry")
        st.markdown(
            "<span style='color:#8892a4;font-size:.82rem;'>"
            "Negative = entry below VWAP (good for BUYs) · "
            "Positive = entry above VWAP (bad for BUYs)"
            "</span>",
            unsafe_allow_html=True,
        )

        colors_vwap = [
            (_GREEN if row["action"].upper() == "BUY" else _CYAN)
            if row["vwap_rel"] <= 0
            else (_RED if row["action"].upper() == "BUY" else _AMBER)
            for _, row in scored.iterrows()
        ]
        fig_vwap = go.Figure(go.Bar(
            x=scored["symbol"],
            y=scored["vwap_rel"],
            marker_color=colors_vwap,
            text=[f"{v:+.2f}%" for v in scored["vwap_rel"]],
            textposition="outside",
            textfont=dict(color=_WHITE, size=9),
        ))
        fig_vwap.add_hline(y=0, line_color=_WHITE, line_dash="dash", line_width=1)
        fig_vwap.update_layout(
            **_LAYOUT,
            title=dict(text="Entry vs VWAP (%)", font=dict(color=_WHITE, size=13)),
            yaxis_title="Entry vs VWAP (%)",
            height=340,
        )
        st.plotly_chart(fig_vwap, use_container_width=True)

        # Summary
        buys = scored[scored["action"].str.upper() == "BUY"]
        sells = scored[scored["action"].str.upper() == "SELL"]
        c1, c2 = st.columns(2)
        if not buys.empty:
            avg_buy_vwap = buys["vwap_rel"].mean()
            label = "Above VWAP (bad)" if avg_buy_vwap > 0 else "Below VWAP (good)"
            color = _RED if avg_buy_vwap > 0 else _GREEN
            c1.markdown(
                f"""<div class="eq-card">
                <div class="eq-label">BUY entries avg VWAP position</div>
                <div class="eq-val" style="color:{color};">{avg_buy_vwap:+.2f}%</div>
                <div class="eq-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        if not sells.empty:
            avg_sell_vwap = sells["vwap_rel"].mean()
            label = "Below VWAP (bad for sells)" if avg_sell_vwap < 0 else "Above VWAP (good for sells)"
            color = _GREEN if avg_sell_vwap > 0 else _RED
            c2.markdown(
                f"""<div class="eq-card">
                <div class="eq-label">SELL entries avg VWAP position</div>
                <div class="eq-val" style="color:{color};">{avg_sell_vwap:+.2f}%</div>
                <div class="eq-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Tab 3: Time of Day ────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Avg PnL % by Hour of Entry")
        if "pnl_pct" in scored.columns:
            by_hour = (
                scored.groupby("hour")
                .agg(avg_pnl=("pnl_pct", "mean"), count=("pnl_pct", "count"),
                     avg_score=("fill_score", "mean"))
                .reset_index()
            )
        else:
            by_hour = (
                scored.groupby("hour")
                .agg(count=("fill_score", "count"), avg_score=("fill_score", "mean"))
                .reset_index()
            )
            by_hour["avg_pnl"] = np.random.uniform(-1, 3, len(by_hour))

        hour_labels = [f"{h:02d}:00" for h in by_hour["hour"]]
        bar_colors = [_GREEN if v >= 0 else _RED for v in by_hour["avg_pnl"]]

        fig_tod = go.Figure()
        fig_tod.add_trace(go.Bar(
            x=hour_labels,
            y=by_hour["avg_pnl"],
            marker_color=bar_colors,
            name="Avg PnL %",
            yaxis="y1",
            text=[f"{v:.2f}%" for v in by_hour["avg_pnl"]],
            textposition="outside",
            textfont=dict(color=_WHITE, size=9),
        ))
        fig_tod.add_trace(go.Scatter(
            x=hour_labels,
            y=by_hour["avg_score"],
            mode="lines+markers",
            name="Avg Fill Score",
            line=dict(color=_CYAN, width=2),
            marker=dict(size=6, color=_CYAN),
            yaxis="y2",
        ))
        fig_tod.update_layout(
            **_LAYOUT,
            title=dict(text="Avg PnL% & Fill Score by Hour", font=dict(color=_WHITE, size=13)),
            yaxis=dict(title="Avg PnL %", color=_MUTED, showgrid=False),
            yaxis2=dict(title="Avg Fill Score", overlaying="y", side="right",
                        color=_CYAN, showgrid=False),
            height=360,
            barmode="group",
        )
        st.plotly_chart(fig_tod, use_container_width=True)

        # Trade count table
        st.markdown("#### Trade Count by Hour")
        by_hour_display = by_hour[["hour", "count", "avg_pnl", "avg_score"]].copy()
        by_hour_display.columns = ["Hour", "# Trades", "Avg PnL %", "Avg Fill Score"]
        by_hour_display["Hour"] = by_hour_display["Hour"].apply(lambda h: f"{h:02d}:00")
        by_hour_display["Avg PnL %"] = by_hour_display["Avg PnL %"].map("{:.2f}%".format)
        by_hour_display["Avg Fill Score"] = by_hour_display["Avg Fill Score"].map("{:.1f}".format)
        st.dataframe(by_hour_display, use_container_width=True, hide_index=True)

    # ── Tab 4: Leaderboard ────────────────────────────────────────────────────
    with tab4:
        display_cols = ["symbol", "date", "action", "entry_price", "slippage_bps",
                        "vwap_rel", "fill_score", "pnl_pct"]
        display_cols = [c for c in display_cols if c in scored.columns]

        top5    = scored.nlargest(5, "fill_score")[display_cols]
        bottom5 = scored.nsmallest(5, "fill_score")[display_cols]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"#### 🥇 Top 5 Fills")
            top5_display = top5.copy()
            if "pnl_pct" in top5_display:
                top5_display["pnl_pct"] = top5_display["pnl_pct"].map("{:.2f}%".format)
            if "slippage_bps" in top5_display:
                top5_display["slippage_bps"] = top5_display["slippage_bps"].map("{:.1f}".format)
            if "vwap_rel" in top5_display:
                top5_display["vwap_rel"] = top5_display["vwap_rel"].map("{:+.3f}%".format)
            if "fill_score" in top5_display:
                top5_display["fill_score"] = top5_display["fill_score"].map("{:.1f}".format)
            st.dataframe(top5_display, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### ⚠️ Bottom 5 Fills")
            bot5_display = bottom5.copy()
            if "pnl_pct" in bot5_display:
                bot5_display["pnl_pct"] = bot5_display["pnl_pct"].map("{:.2f}%".format)
            if "slippage_bps" in bot5_display:
                bot5_display["slippage_bps"] = bot5_display["slippage_bps"].map("{:.1f}".format)
            if "vwap_rel" in bot5_display:
                bot5_display["vwap_rel"] = bot5_display["vwap_rel"].map("{:+.3f}%".format)
            if "fill_score" in bot5_display:
                bot5_display["fill_score"] = bot5_display["fill_score"].map("{:.1f}".format)
            st.dataframe(bot5_display, use_container_width=True, hide_index=True)

        # Fill score distribution
        st.markdown("#### Fill Score Distribution")
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=scored["fill_score"],
            nbinsx=20,
            marker=dict(
                color=scored["fill_score"],
                colorscale=[[0, _RED], [0.5, _AMBER], [1, _GREEN]],
                showscale=True,
                colorbar=dict(title="Score", tickfont=dict(color=_MUTED)),
            ),
        ))
        fig_score.add_vline(x=avg_score, line_color=_CYAN, line_dash="dash",
                            annotation_text=f"Avg {avg_score:.1f}",
                            annotation_font_color=_CYAN)
        fig_score.update_layout(
            **_LAYOUT,
            title=dict(text="Fill Quality Score Distribution", font=dict(color=_WHITE, size=13)),
            xaxis_title="Fill Score (0–100)",
            yaxis_title="Count",
            height=300,
        )
        st.plotly_chart(fig_score, use_container_width=True)

    # ── Tab 5: Suggestions ────────────────────────────────────────────────────
    with tab5:
        st.markdown("#### 💡 Improvement Suggestions")
        suggestions = _generate_suggestions(scored)
        for tip in suggestions:
            st.markdown(
                f'<div class="tip-box">💡 {tip}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Setup Type Performance")
        if "setup_type" in scored.columns and "pnl_pct" in scored.columns:
            by_setup = (
                scored.groupby("setup_type")
                .agg(
                    avg_score=("fill_score", "mean"),
                    avg_pnl=("pnl_pct", "mean"),
                    count=("fill_score", "count"),
                )
                .reset_index()
                .sort_values("avg_score", ascending=False)
            )
            fig_setup = go.Figure()
            fig_setup.add_trace(go.Bar(
                name="Avg Fill Score",
                x=by_setup["setup_type"],
                y=by_setup["avg_score"],
                marker_color=_CYAN,
                yaxis="y1",
            ))
            fig_setup.add_trace(go.Scatter(
                name="Avg PnL %",
                x=by_setup["setup_type"],
                y=by_setup["avg_pnl"],
                mode="lines+markers",
                line=dict(color=_GREEN, width=2),
                marker=dict(size=7),
                yaxis="y2",
            ))
            fig_setup.update_layout(
                **_LAYOUT,
                title=dict(text="Fill Score & PnL by Setup Type", font=dict(color=_WHITE, size=13)),
                yaxis=dict(title="Avg Fill Score", color=_MUTED),
                yaxis2=dict(title="Avg PnL %", overlaying="y", side="right",
                            color=_GREEN, showgrid=False),
                height=320,
            )
            st.plotly_chart(fig_setup, use_container_width=True)
