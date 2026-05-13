"""Options analytics UI — PCR, Max Pain, OI heatmap."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options.analytics import (
    compute_max_pain,
    compute_pcr,
    get_atm_iv,
    get_iv_percentile,
    get_oi_buildup,
    get_option_chain,
)

# ─────────────────────────────────────────────────────────────────────────────
# Supported indices
# ─────────────────────────────────────────────────────────────────────────────
_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

# Approximate spot proxies via yfinance for ATM calculation when live feed is
# unavailable — used only to position the ATM strike, not for trading.
_SPOT_YF = {
    "NIFTY":      "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
    "FINNIFTY":   "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MIDCAP_SELECT.NS",
}


@st.cache_data(ttl=120, show_spinner=False)
def _get_spot(symbol: str) -> float:
    """Fetch approximate spot price for ATM positioning."""
    try:
        import yfinance as yf

        ticker = _SPOT_YF.get(symbol, "^NSEI")
        hist = yf.Ticker(ticker).history(period="2d")
        if hist is not None and not hist.empty:
            if hasattr(hist.columns, "levels"):
                hist.columns = [c[0] for c in hist.columns]
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PCR interpretation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pcr_label(pcr: float) -> tuple[str, str]:
    """Return (interpretation text, hex color) for a PCR value."""
    if pcr > 1.3:
        return "Heavily oversold — likely bounce", "#00c896"
    if pcr >= 1.0:
        return "Mildly bullish", "#00bfa5"
    if pcr >= 0.8:
        return "Neutral", "#8892a4"
    return "Bearish — put unwinding or call buying", "#ff4466"


def _pcr_metric_color(pcr: float) -> str:
    if pcr > 1.2:
        return "#00c896"
    if pcr < 0.8:
        return "#ff4466"
    return "#8892a4"


# ─────────────────────────────────────────────────────────────────────────────
# Reusable metric card
# ─────────────────────────────────────────────────────────────────────────────

def _metric_card(col, label: str, value: str, sub: str = "", color: str = "#e8eaf0") -> None:
    col.markdown(
        f"<div class='devbloom-card' style='padding:.85rem 1rem;text-align:center'>"
        f"<div style='font-size:.62rem;color:#8892a4;text-transform:uppercase;"
        f"letter-spacing:.07em;margin-bottom:.3rem'>{label}</div>"
        f"<div style='font-size:1.25rem;font-family:JetBrains Mono,monospace;"
        f"font-weight:700;color:{color}'>{value}</div>"
        f"<div style='font-size:.72rem;color:#8892a4;margin-top:.2rem'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8892a4", size=11),
    margin=dict(l=0, r=0, t=36, b=0),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        color="#8892a4",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        color="#8892a4",
        tickfont=dict(size=10),
    ),
)


def _build_oi_chart(df: pd.DataFrame, spot: float, use_change: bool = False) -> go.Figure:
    """Horizontal bar chart: CE OI (red) and PE OI (green) for top 10 ATM strikes."""
    if df is None or df.empty:
        return go.Figure()

    # Pick 10 strikes closest to spot
    if spot > 0:
        df = df.copy()
        df["_dist"] = (df["strike"] - spot).abs()
        atm_df = df.nsmallest(10, "_dist").sort_values("strike")
    else:
        atm_df = df.sort_values("strike").tail(10)

    ce_col = "ce_coi" if use_change else "ce_oi"
    pe_col = "pe_coi" if use_change else "pe_oi"
    strikes = atm_df["strike"].astype(int).astype(str).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="CE OI" if not use_change else "CE ΔOI",
            y=strikes,
            x=atm_df[ce_col].tolist(),
            orientation="h",
            marker_color="rgba(255, 68, 102, 0.75)",
            hovertemplate="Strike %{y}<br>CE: %{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="PE OI" if not use_change else "PE ΔOI",
            y=strikes,
            x=[-v for v in atm_df[pe_col].tolist()],  # mirror to left
            orientation="h",
            marker_color="rgba(0, 200, 150, 0.75)",
            hovertemplate="Strike %{y}<br>PE: %{customdata:,.0f}<extra></extra>",
            customdata=atm_df[pe_col].tolist(),
        )
    )

    title = (
        "Open Interest by Strike — Support & Resistance"
        if not use_change
        else "Change in OI — Fresh Buildup / Unwinding"
    )
    layout = dict(_DARK_LAYOUT)
    layout.update(
        title=dict(text=title, font=dict(size=12, color="#c8cdd8"), x=0),
        barmode="overlay",
        height=340,
        bargap=0.25,
    )
    layout["xaxis"] = dict(
        _DARK_LAYOUT["xaxis"],
        title="Open Interest (contracts)",
        tickformat=",",
    )
    layout["yaxis"] = dict(
        _DARK_LAYOUT["yaxis"],
        title="Strike",
        autorange="reversed",
    )
    # Add a vertical line at 0
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_options_page() -> None:
    """Render the full Options Analytics page."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.4rem;margin:0 0 .1rem'>🎯 Options Analytics</h2>"
        "<p style='color:#8892a4;font-size:.78rem;margin:0 0 1rem'>"
        "PCR · Max Pain · OI Heatmap · IV Percentile — NSE live data</p>",
        unsafe_allow_html=True,
    )

    # ── Top bar: symbol selector + refresh ────────────────────────────────────
    tb1, tb2, tb3 = st.columns([2, 3, 1])
    with tb1:
        symbol = st.selectbox(
            "Index",
            _SYMBOLS,
            index=0,
            key="opt_symbol_sel",
            label_visibility="collapsed",
        )
    with tb3:
        refresh = st.button("↻ Refresh", key="opt_refresh", use_container_width=True)

    if refresh:
        st.cache_data.clear()

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner(f"Fetching {symbol} option chain…"):
        df, expiry = get_option_chain(symbol)
        spot = _get_spot(symbol)

    if df is None or df.empty:
        st.error(
            "Could not fetch option chain data. NSE API may be rate-limiting or "
            "the market is closed. Try again during market hours (9:15 AM – 3:30 PM IST)."
        )
        return

    # ── Expiry banner ─────────────────────────────────────────────────────────
    with tb2:
        st.markdown(
            f"<div style='padding:.45rem .8rem;background:rgba(0,212,255,0.08);"
            f"border:1px solid rgba(0,212,255,0.2);border-radius:6px;"
            f"font-family:JetBrains Mono,monospace;font-size:.8rem;color:#00d4ff'>"
            f"Expiry: {expiry}"
            f"{'&nbsp;&nbsp;|&nbsp;&nbsp;Spot: ₹' + f'{spot:,.2f}' if spot > 0 else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Compute analytics ─────────────────────────────────────────────────────
    pcr       = compute_pcr(df)
    max_pain  = compute_max_pain(df)
    atm_iv    = get_atm_iv(df, spot)
    iv_pct    = get_iv_percentile(df)
    total_oi  = int(df["ce_oi"].sum() + df["pe_oi"].sum())
    oi_levels = get_oi_buildup(df, spot)

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    pcr_color = _pcr_metric_color(pcr)
    _metric_card(m1, "PCR (OI)", f"{pcr:.2f}", "Put/Call OI ratio", pcr_color)
    _metric_card(m2, "Max Pain", f"₹{max_pain:,.0f}", "Price of max seller profit", "#e8eaf0")
    _metric_card(
        m3,
        "ATM IV",
        f"{atm_iv:.1f}%" if atm_iv else "—",
        f"IV Rank: {iv_pct:.0f}th pct",
        "#eab308" if iv_pct > 75 else ("#22c55e" if iv_pct < 30 else "#e8eaf0"),
    )
    _metric_card(
        m4,
        "Total OI",
        f"{total_oi / 1_000:.0f}K" if total_oi >= 1000 else str(total_oi),
        "CE + PE contracts",
        "#e8eaf0",
    )

    st.markdown("<div style='margin:.5rem 0'></div>", unsafe_allow_html=True)

    # ── PCR Interpretation ────────────────────────────────────────────────────
    interp, icolor = _pcr_label(pcr)
    if pcr > 1.3:
        icon = "🟢"
    elif pcr >= 1.0:
        icon = "🟢"
    elif pcr >= 0.8:
        icon = "⚪"
    else:
        icon = "🔴"

    st.markdown(
        f"<div style='padding:.65rem 1rem;background:rgba(255,255,255,0.03);"
        f"border-left:3px solid {icolor};border-radius:0 6px 6px 0;"
        f"margin-bottom:.75rem'>"
        f"<span style='font-size:.8rem;color:#8892a4'>PCR Signal: </span>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:.9rem;"
        f"font-weight:600;color:{icolor}'>{icon} {interp}</span>"
        f"<span style='font-size:.72rem;color:#8892a4;margin-left:.75rem'>"
        f"(PCR {pcr:.2f})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # PCR legend
    with st.expander("PCR Scale Reference", expanded=False):
        st.markdown(
            "<table style='width:100%;font-size:.78rem;border-collapse:collapse'>"
            "<tr><td style='padding:.3rem .6rem;color:#00c896'>PCR &gt; 1.3</td>"
            "<td style='color:#8892a4'>Heavily oversold — likely bounce</td></tr>"
            "<tr><td style='padding:.3rem .6rem;color:#00bfa5'>PCR 1.0 – 1.3</td>"
            "<td style='color:#8892a4'>Mildly bullish</td></tr>"
            "<tr><td style='padding:.3rem .6rem;color:#8892a4'>PCR 0.8 – 1.0</td>"
            "<td style='color:#8892a4'>Neutral</td></tr>"
            "<tr><td style='padding:.3rem .6rem;color:#ff4466'>PCR &lt; 0.8</td>"
            "<td style='color:#8892a4'>Bearish — put unwinding or call buying</td></tr>"
            "</table>",
            unsafe_allow_html=True,
        )

    # ── OI bar chart ──────────────────────────────────────────────────────────
    st.plotly_chart(
        _build_oi_chart(df, spot, use_change=False),
        width="stretch",
        config={"displayModeBar": False},
        key="opt_oi_chart",
    )

    # ── OI Change chart ───────────────────────────────────────────────────────
    has_coi = df["ce_coi"].abs().sum() + df["pe_coi"].abs().sum() > 0
    if has_coi:
        st.plotly_chart(
            _build_oi_chart(df, spot, use_change=True),
            width="stretch",
            config={"displayModeBar": False},
            key="opt_coi_chart",
        )
    else:
        st.info(
            "Change-in-OI data unavailable from this source "
            "(yfinance fallback doesn't provide intraday COI).",
            icon="ℹ️",
        )

    # ── Key levels callout ────────────────────────────────────────────────────
    resistance = oi_levels.get("resistance_levels", [])
    support    = oi_levels.get("support_levels", [])

    resist_str = (
        f"₹{resistance[0]['strike']:,.0f}" if resistance else "—"
    )
    support_str = (
        f"₹{support[0]['strike']:,.0f}" if support else "—"
    )

    st.markdown(
        f"<div style='padding:.75rem 1.1rem;background:rgba(255,255,255,0.03);"
        f"border:1px solid rgba(255,255,255,0.07);border-radius:8px;"
        f"font-size:.82rem;line-height:1.9'>"
        f"<span style='color:#8892a4'>Max Pain: </span>"
        f"<span style='color:#e8eaf0;font-family:JetBrains Mono,monospace;"
        f"font-weight:600'>₹{max_pain:,.0f}</span>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"<span style='color:#8892a4'>Strong Support (PE wall): </span>"
        f"<span style='color:#00c896;font-family:JetBrains Mono,monospace;"
        f"font-weight:600'>{support_str}</span>"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"<span style='color:#8892a4'>Strong Resistance (CE wall): </span>"
        f"<span style='color:#ff4466;font-family:JetBrains Mono,monospace;"
        f"font-weight:600'>{resist_str}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin:.75rem 0'></div>", unsafe_allow_html=True)

    # ── Full OI table (expandable) ────────────────────────────────────────────
    with st.expander("📋 Full Option Chain Table", expanded=False):
        display_cols = {
            "strike": "Strike",
            "ce_oi":   "CE OI",
            "ce_coi":  "CE ΔOI",
            "ce_iv":   "CE IV %",
            "ce_ltp":  "CE LTP",
            "pe_ltp":  "PE LTP",
            "pe_iv":   "PE IV %",
            "pe_coi":  "PE ΔOI",
            "pe_oi":   "PE OI",
        }
        available = [c for c in display_cols if c in df.columns]
        tdf = df[available].copy()
        tdf.columns = [display_cols[c] for c in available]

        # Highlight ATM row
        if spot > 0:
            atm_idx = (df["strike"] - spot).abs().idxmin()
            atm_strike = df.loc[atm_idx, "strike"]

            def _highlight_atm(row):
                if row["Strike"] == atm_strike:
                    return ["background-color: rgba(0,212,255,0.08)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                tdf.style.apply(_highlight_atm, axis=1).format(
                    {
                        "CE OI":   "{:,.0f}",
                        "CE ΔOI":  "{:+,.0f}",
                        "PE OI":   "{:,.0f}",
                        "PE ΔOI":  "{:+,.0f}",
                        "CE IV %": "{:.1f}",
                        "PE IV %": "{:.1f}",
                        "CE LTP":  "₹{:.2f}",
                        "PE LTP":  "₹{:.2f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.dataframe(tdf, hide_index=True, width="stretch")
