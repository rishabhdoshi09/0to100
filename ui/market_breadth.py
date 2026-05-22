"""
Market Breadth Dashboard — NSE / Nifty50
Shows A/D line, 52-week highs/lows, above/below 200 SMA, McClellan Oscillator,
breadth divergence warning, and a composite Breadth Score.
"""
from __future__ import annotations

import datetime
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "KOTAKBANK.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "NESTLEIND.NS", "ULTRACEMCO.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "SUNPHARMA.NS",
]

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


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_breadth_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (daily_close_df, today_stats_df).
    daily_close_df  — 252 trading days of adjusted close prices, columns = symbols
    today_stats_df  — per-symbol row: last, prev_close, high52, low52, sma200
    Falls back to synthetic data if yfinance is unavailable or rate-limited.
    """
    try:
        import yfinance as yf  # lazy import
        end   = datetime.datetime.today()
        start = end - datetime.timedelta(days=400)  # enough for 252 td + 200 SMA warmup
        raw   = yf.download(
            NIFTY50_SYMBOLS,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise ValueError("Empty download")

        close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
        close = close.dropna(how="all").ffill().tail(252)

        stats_rows = []
        for sym in close.columns:
            s = close[sym].dropna()
            if len(s) < 2:
                continue
            last       = float(s.iloc[-1])
            prev       = float(s.iloc[-2])
            high52     = float(s.tail(252).max())
            low52      = float(s.tail(252).min())
            sma200_val = float(s.tail(200).mean()) if len(s) >= 200 else float(s.mean())
            stats_rows.append({
                "symbol": sym,
                "last":    last,
                "prev":    prev,
                "high52":  high52,
                "low52":   low52,
                "sma200":  sma200_val,
            })
        stats = pd.DataFrame(stats_rows)
        return close, stats

    except Exception:
        # ── Synthetic fallback ──────────────────────────────────────────────
        rng  = np.random.default_rng(42)
        days = 252
        syms = NIFTY50_SYMBOLS
        prices = {}
        for sym in syms:
            drift  = rng.uniform(-0.0002, 0.0005)
            vol    = rng.uniform(0.010, 0.022)
            base   = rng.uniform(300, 4000)
            log_r  = rng.normal(drift, vol, days)
            prices[sym] = base * np.exp(np.cumsum(log_r))

        idx   = pd.bdate_range(end=datetime.date.today(), periods=days)
        close = pd.DataFrame(prices, index=idx)

        stats_rows = []
        for sym in syms:
            s          = close[sym]
            last       = float(s.iloc[-1])
            prev       = float(s.iloc[-2])
            high52     = float(s.tail(252).max())
            low52      = float(s.tail(252).min())
            sma200_val = float(s.tail(200).mean())
            stats_rows.append({
                "symbol": sym, "last": last, "prev": prev,
                "high52": high52, "low52": low52, "sma200": sma200_val,
            })
        stats = pd.DataFrame(stats_rows)
        return close, stats


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_nifty_index() -> pd.Series:
    """Fetch ^NSEI index last 60 days."""
    try:
        import yfinance as yf
        end   = datetime.datetime.today()
        start = end - datetime.timedelta(days=90)
        raw   = yf.download("^NSEI", start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError
        close = raw["Close"].dropna().tail(60)
        return close
    except Exception:
        rng   = np.random.default_rng(7)
        idx   = pd.bdate_range(end=datetime.date.today(), periods=60)
        vals  = 22000 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, 60)))
        return pd.Series(vals, index=idx, name="^NSEI")


# ── Indicator computation ─────────────────────────────────────────────────────

def _compute_breadth_series(close: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading day (last 30), compute:
      advances, declines, unchanged, ad_ratio, pct_above200, new_highs, new_lows
    """
    rows = []
    dates = close.index[-31:]  # one extra for day-over-day change
    for i in range(1, len(dates)):
        d      = dates[i]
        d_prev = dates[i - 1]
        adv = dec = unch = 0
        n_highs = n_lows = above200 = 0
        n_valid = 0
        for sym in close.columns:
            s    = close[sym]
            last = s.get(d, np.nan)
            prev = s.get(d_prev, np.nan)
            if np.isnan(last) or np.isnan(prev):
                continue
            n_valid += 1
            chg = last - prev
            if chg > 0:   adv  += 1
            elif chg < 0: dec  += 1
            else:         unch += 1
            # 52-week high/low check (rolling 252 days up to d)
            hist = s[:d].dropna().tail(252)
            if len(hist) >= 2:
                if last >= hist.max():  n_highs += 1
                if last <= hist.min():  n_lows  += 1
            # above 200 SMA
            hist200 = s[:d].dropna().tail(200)
            sma200  = hist200.mean() if len(hist200) >= 50 else np.nan
            if not np.isnan(sma200) and last > sma200:
                above200 += 1

        ad_ratio     = adv / max(adv + dec, 1)
        pct_above200 = above200 / max(n_valid, 1) * 100
        rows.append({
            "date":         d,
            "advances":     adv,
            "declines":     dec,
            "unchanged":    unch,
            "ad_ratio":     ad_ratio,
            "pct_above200": pct_above200,
            "new_highs":    n_highs,
            "new_lows":     n_lows,
            "net_ad":       adv - dec,
        })
    df = pd.DataFrame(rows).set_index("date")
    # Cumulative A/D line
    df["ad_line"] = df["net_ad"].cumsum()
    return df


def _mclellan(breadth_df: pd.DataFrame) -> pd.Series:
    """
    Simplified McClellan Oscillator = EMA19(ratio_adj) - EMA39(ratio_adj)
    ratio_adj = (advances - declines) / (advances + declines + unchanged)
    """
    ratio_adj = breadth_df["net_ad"] / (
        breadth_df["advances"] + breadth_df["declines"] + breadth_df["unchanged"].replace(0, 1)
    )
    ema19 = ratio_adj.ewm(span=19, adjust=False).mean()
    ema39 = ratio_adj.ewm(span=39, adjust=False).mean()
    return (ema19 - ema39) * 1000  # scale to readable range


def _breadth_score(stats: pd.DataFrame, breadth_df: pd.DataFrame) -> int:
    """
    Composite 0-100 score:
      - A/D ratio today  (0-40 pts)
      - % above 200 SMA  (0-40 pts)
      - highs vs lows    (0-20 pts)
    """
    today = breadth_df.iloc[-1]
    ad_pts    = int(today["ad_ratio"] * 40)
    sma_pts   = int(today["pct_above200"] / 100 * 40)
    hl_total  = today["new_highs"] + today["new_lows"]
    hl_pts    = int(today["new_highs"] / max(hl_total, 1) * 20)
    return min(100, ad_pts + sma_pts + hl_pts)


# ── Chart builders ────────────────────────────────────────────────────────────

def _gauge_chart(value: float, title: str, max_val: float = 1.0) -> go.Figure:
    pct = value / max_val
    color = GREEN if pct > 0.6 else (AMBER if pct > 0.4 else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100 if max_val == 1.0 else value, 1),
        number={"suffix": "%" if max_val == 1.0 else "", "font": {"color": WHITE, "size": 22}},
        title={"text": title, "font": {"color": "#8892a4", "size": 12}},
        gauge={
            "axis": {"range": [0, 100 if max_val == 1.0 else max_val],
                     "tickfont": {"color": "#8892a4", "size": 9}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.04)",
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0,  40 if max_val == 1.0 else max_val*0.4], "color": "rgba(255,68,102,0.10)"},
                {"range": [40 if max_val == 1.0 else max_val*0.4,
                           60 if max_val == 1.0 else max_val*0.6], "color": "rgba(255,184,0,0.10)"},
                {"range": [60 if max_val == 1.0 else max_val*0.6,
                           100 if max_val == 1.0 else max_val],       "color": "rgba(0,255,136,0.08)"},
            ],
        },
    ))
    fig.update_layout(**_LAYOUT, height=200)
    return fig


def _ad_line_chart(breadth_df: pd.DataFrame, nifty: pd.Series) -> go.Figure:
    fig = go.Figure()
    # A/D cumulative line
    fig.add_trace(go.Scatter(
        x=breadth_df.index, y=breadth_df["ad_line"],
        mode="lines", name="A/D Line",
        line=dict(color=CYAN, width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
    ))
    # Nifty normalised overlay (secondary y)
    if not nifty.empty:
        nifty_aligned = nifty.reindex(breadth_df.index, method="ffill").dropna()
        if not nifty_aligned.empty:
            nifty_norm = (nifty_aligned - nifty_aligned.min()) / (
                nifty_aligned.max() - nifty_aligned.min() + 1e-9
            ) * (breadth_df["ad_line"].max() - breadth_df["ad_line"].min()) + breadth_df["ad_line"].min()
            fig.add_trace(go.Scatter(
                x=nifty_norm.index, y=nifty_norm,
                mode="lines", name="Nifty50 (norm)",
                line=dict(color=AMBER, width=1.5, dash="dot"),
            ))
    layout = dict(**_LAYOUT)
    layout["title"] = dict(text="Cumulative A/D Line vs Nifty50", font=dict(color=WHITE, size=13))
    layout["height"] = 280
    fig.update_layout(**layout)
    return fig


def _highs_lows_chart(breadth_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=breadth_df.index, y=breadth_df["new_highs"],
        name="52W Highs", marker_color=GREEN, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=breadth_df.index, y=-breadth_df["new_lows"],
        name="52W Lows", marker_color=RED, opacity=0.85,
    ))
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text="New 52-Week Highs / Lows", font=dict(color=WHITE, size=13))
    layout["barmode"] = "relative"
    layout["height"]  = 220
    layout["yaxis"]   = dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)",
                              zeroline=True, zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_layout(**layout)
    return fig


def _mclellan_chart(breadth_df: pd.DataFrame) -> go.Figure:
    osc = _mclellan(breadth_df)
    colors = [GREEN if v >= 0 else RED for v in osc]
    fig = go.Figure(go.Bar(
        x=osc.index, y=osc.values,
        marker_color=colors, opacity=0.85, name="McClellan Osc",
    ))
    fig.add_hline(y=0,   line_color="rgba(255,255,255,0.2)",  line_width=1)
    fig.add_hline(y=100, line_color="rgba(0,255,136,0.25)",   line_dash="dot", line_width=1)
    fig.add_hline(y=-100,line_color="rgba(255,68,102,0.25)",  line_dash="dot", line_width=1)
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text="McClellan Oscillator", font=dict(color=WHITE, size=13))
    layout["height"]  = 220
    layout["yaxis"]   = dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)",
                              zeroline=True, zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_layout(**layout)
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render_market_breadth() -> None:
    st.markdown("## 📊 Market Breadth Dashboard")
    st.caption("NSE Nifty50 universe · data cached 15 min")

    with st.spinner("Fetching breadth data…"):
        try:
            close, stats = _fetch_breadth_data()
            nifty        = _fetch_nifty_index()
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            return

    # Compute breadth series (last 30 days)
    close_30 = close.tail(31)
    try:
        breadth_df = _compute_breadth_series(close_30)
    except Exception as exc:
        st.warning(f"Could not compute breadth indicators: {exc}")
        return

    if breadth_df.empty:
        st.warning("Insufficient data to compute breadth metrics.")
        return

    today_b  = breadth_df.iloc[-1]
    score    = _breadth_score(stats, breadth_df)

    # ── Breadth Score banner ─────────────────────────────────────────────────
    score_color = GREEN if score > 60 else (AMBER if score >= 40 else RED)
    score_label = "BULLISH" if score > 60 else ("NEUTRAL" if score >= 40 else "BEARISH")
    st.markdown(
        f"""
        <div style="
          background: rgba(255,255,255,0.04);
          border: 1px solid {score_color}44;
          border-radius: 16px; padding: 1.2rem 1.8rem;
          display: flex; align-items: center; gap: 1.5rem;
          margin-bottom: 1rem;
        ">
          <div style="font-size:3rem; font-weight:800; color:{score_color};
                      font-family:'JetBrains Mono',monospace; line-height:1;">
            {score}
          </div>
          <div>
            <div style="font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:.08em; color:#8892a4;">Breadth Score</div>
            <div style="font-size:1.2rem; font-weight:700; color:{score_color};">
              {score_label}
            </div>
          </div>
          <div style="margin-left:auto; font-size:0.78rem; color:#8892a4; text-align:right;">
            A/D Ratio: <span style="color:{WHITE}; font-family:monospace;">
              {today_b['advances']:.0f} / {today_b['declines']:.0f}</span><br>
            Above 200 SMA: <span style="color:{WHITE}; font-family:monospace;">
              {today_b['pct_above200']:.1f}%</span><br>
            52W Highs/Lows: <span style="color:{WHITE}; font-family:monospace;">
              {today_b['new_highs']:.0f} / {today_b['new_lows']:.0f}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Divergence warning ───────────────────────────────────────────────────
    if not nifty.empty and len(nifty) >= 2:
        nifty_chg = (nifty.iloc[-1] - nifty.iloc[-2]) / nifty.iloc[-2] * 100
        ad_ratio_today = float(today_b["ad_ratio"])
        if nifty_chg > 0.3 and ad_ratio_today < 0.45:
            st.warning(
                "⚠️  **Breadth Divergence Detected** — Nifty50 is rising "
                f"({nifty_chg:+.2f}%) but market breadth is weak "
                f"(A/D ratio {ad_ratio_today:.1%}). Rally may be narrow / unsustainable."
            )
        elif nifty_chg < -0.3 and ad_ratio_today > 0.55:
            st.info(
                "ℹ️  **Positive Divergence** — Nifty50 is falling "
                f"({nifty_chg:+.2f}%) but breadth remains healthy "
                f"(A/D ratio {ad_ratio_today:.1%}). Potential support."
            )

    # ── Gauge row ────────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            _gauge_chart(float(today_b["ad_ratio"]), "A/D Ratio", max_val=1.0),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            _gauge_chart(float(today_b["pct_above200"]) / 100, "Above 200 SMA", max_val=1.0),
            use_container_width=True,
        )
    with g3:
        hl_total = today_b["new_highs"] + today_b["new_lows"]
        hl_ratio = float(today_b["new_highs"]) / max(hl_total, 1)
        st.plotly_chart(
            _gauge_chart(hl_ratio, "Highs / (Highs+Lows)", max_val=1.0),
            use_container_width=True,
        )

    # ── A/D line chart ───────────────────────────────────────────────────────
    st.plotly_chart(_ad_line_chart(breadth_df, nifty), use_container_width=True)

    # ── Highs/Lows + McClellan ────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_highs_lows_chart(breadth_df), use_container_width=True)
    with c2:
        st.plotly_chart(_mclellan_chart(breadth_df), use_container_width=True)

    # ── Raw table ────────────────────────────────────────────────────────────
    with st.expander("📋 Raw breadth data (last 10 sessions)"):
        display = breadth_df[["advances", "declines", "unchanged",
                               "ad_ratio", "pct_above200", "new_highs", "new_lows",
                               "ad_line"]].tail(10).copy()
        display["ad_ratio"]     = display["ad_ratio"].map("{:.1%}".format)
        display["pct_above200"] = display["pct_above200"].map("{:.1f}%".format)
        st.dataframe(display[::-1], use_container_width=True)
