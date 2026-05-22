"""
Portfolio Heat Map — visual grid of all open positions.
Color = P&L direction. Size = position weight. Shows conviction + risk at a glance.
"""
from __future__ import annotations

import streamlit as st


def render_portfolio_heatmap(positions: list[dict]) -> None:
    """
    Render a Plotly treemap heatmap of all open positions.

    Each position dict should contain:
        symbol        : str   — ticker symbol
        entry_price   : float — average entry price
        current_price : float — current market price
        pnl_pct       : float — unrealised P&L percentage (e.g. 4.5 means +4.5%)
        position_pct  : float — weight of this position in the portfolio (0–100)
        archetype     : str   (optional) — setup name / pattern label
        conviction    : float (optional) — conviction score (0–100)
        stop_distance : float (optional) — % distance to stop loss
        days_held     : int   (optional) — number of days position has been held
    """
    if not positions:
        st.markdown(
            "<div style='background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);"
            "border-radius:10px;padding:1.5rem;text-align:center;color:#4a5568;font-size:.82rem'>"
            "No open positions to display.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        import pandas as pd
        import plotly.express as px
    except ImportError:
        st.warning("plotly / pandas not installed — cannot render heatmap.")
        return

    # ── Build dataframe ───────────────────────────────────────────────────────
    rows = []
    for p in positions:
        symbol        = str(p.get("symbol", "UNKNOWN"))
        entry_price   = float(p.get("entry_price", 0) or 0)
        current_price = float(p.get("current_price", 0) or 0)
        pnl_pct       = float(p.get("pnl_pct", 0) or 0)
        position_pct  = float(p.get("position_pct", 1) or 1)
        archetype     = str(p.get("archetype", "") or "")
        conviction    = p.get("conviction")
        stop_distance = float(p.get("stop_distance", 0) or 0)
        days_held     = int(p.get("days_held", 0) or 0)

        # Ensure positive size so treemap renders
        if position_pct <= 0:
            position_pct = 1.0

        rows.append({
            "symbol":        symbol,
            "position_pct":  position_pct,
            "pnl_pct":       pnl_pct,
            "entry_price":   entry_price,
            "current_price": current_price,
            "stop_distance": stop_distance,
            "days_held":     days_held,
            "archetype":     archetype,
            "conviction":    conviction if conviction is not None else "",
        })

    df = pd.DataFrame(rows)

    # ── Build label text inside each cell ────────────────────────────────────
    def _cell_label(row) -> str:
        sign = "+" if row["pnl_pct"] >= 0 else ""
        label = f"<b>{row['symbol']}</b><br>{sign}{row['pnl_pct']:.1f}%"
        if row["conviction"] != "":
            try:
                label += f"<br>Score: {float(row['conviction']):.0f}"
            except (ValueError, TypeError):
                pass
        return label

    df["label"] = df.apply(_cell_label, axis=1)

    # ── Treemap ───────────────────────────────────────────────────────────────
    fig = px.treemap(
        df,
        path=["symbol"],
        values="position_pct",
        color="pnl_pct",
        color_continuous_scale=["#ff4b4b", "#1e293b", "#00d4a0"],
        color_continuous_midpoint=0,
        custom_data=["entry_price", "current_price", "stop_distance", "archetype", "days_held"],
    )

    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{color:.1f}%",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "P&L: %{color:.2f}%<br>"
            "Size: %{value:.1f}%<br>"
            "Entry: ₹%{customdata[0]:,.2f}<br>"
            "Current: ₹%{customdata[1]:,.2f}<br>"
            "Stop Dist: %{customdata[2]:.1f}%<br>"
            "Archetype: %{customdata[3]}<br>"
            "Days held: %{customdata[4]}"
            "<extra></extra>"
        ),
        textfont=dict(size=13),
    )

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig, use_container_width=True, key="portfolio_heatmap")

    # ── Compact risk summary ──────────────────────────────────────────────────
    total_deployed = sum(p["position_pct"] for p in rows)
    largest        = max(rows, key=lambda p: p["position_pct"])
    highest_risk   = max(
        (p for p in rows if p["stop_distance"] > 0),
        key=lambda p: p["stop_distance"],
        default=None,
    )

    # Regime fit — count positions whose archetype is in the recommended playbooks
    try:
        recommended = st.session_state.get("regime_playbooks", [])
        aligned = sum(
            1 for p in rows
            if p["archetype"] and any(
                p["archetype"].upper().replace(" ", "_") in pb.upper()
                for pb in recommended
            )
        ) if recommended else 0
        regime_fit_str = f"{aligned}/{len(rows)}" if recommended else "N/A"
    except Exception:
        regime_fit_str = "N/A"

    summary_parts = [
        f"<span style='color:#8892a4'>Total deployed:</span> "
        f"<span style='color:#e8eaf0;font-weight:600'>{total_deployed:.1f}%</span>",

        f"<span style='color:#8892a4'>Largest position:</span> "
        f"<span style='color:#e8eaf0;font-weight:600'>"
        f"{largest['symbol']} ({largest['position_pct']:.1f}%)</span>",
    ]

    if highest_risk:
        summary_parts.append(
            f"<span style='color:#8892a4'>Highest risk:</span> "
            f"<span style='color:#f59e0b;font-weight:600'>"
            f"{highest_risk['symbol']} ({highest_risk['stop_distance']:.1f}% from stop)</span>"
        )

    if regime_fit_str != "N/A":
        summary_parts.append(
            f"<span style='color:#8892a4'>Regime fit:</span> "
            f"<span style='color:#00d4a0;font-weight:600'>"
            f"{regime_fit_str} positions aligned</span>"
        )

    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:1.5rem;padding:.4rem .2rem;"
        "font-size:.76rem;margin-top:.25rem'>"
        + "  ".join(f"<div>{p}</div>" for p in summary_parts)
        + "</div>",
        unsafe_allow_html=True,
    )
