"""
Kelly Criterion Position Sizer — interactive risk-of-ruin + Monte Carlo.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

_DB_PATH = Path("logs/journal.db")

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


# ── Journal stats loader ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_journal_stats() -> dict | None:
    """Return {win_rate, avg_win_pct, avg_loss_pct} from journal.db or None."""
    if not _DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(_DB_PATH)
        df   = pd.read_sql("SELECT pnl, pnl_pct FROM trades", conn)
        conn.close()
        if df.empty or len(df) < 5:
            return None
        df["pnl"]     = pd.to_numeric(df["pnl"],     errors="coerce")
        df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
        df = df.dropna()
        wins   = df[df["pnl"] > 0]["pnl_pct"]
        losses = df[df["pnl"] <= 0]["pnl_pct"]
        return {
            "win_rate":    round(len(wins) / len(df) * 100, 1),
            "avg_win_pct": round(wins.mean(), 2)   if len(wins)   > 0 else 2.0,
            "avg_loss_pct":round(abs(losses.mean()), 2) if len(losses) > 0 else 1.0,
        }
    except Exception:
        return None


# ── Kelly math ────────────────────────────────────────────────────────────────

def _compute_kelly(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float:
    """
    Full Kelly fraction = (W/L * p - q) / (W/L)
    where p=win_rate, q=1-win_rate, W=avg_win_pct, L=avg_loss_pct.
    Returns a fraction 0-1 (clipped at 0 minimum).
    """
    p = win_rate / 100.0
    q = 1.0 - p
    W = avg_win_pct
    L = avg_loss_pct
    if L == 0:
        return 0.0
    ratio = W / L
    kelly = (ratio * p - q) / ratio
    return max(0.0, kelly)


def _risk_of_ruin(kelly_fraction: float, win_rate: float,
                  avg_win_pct: float, avg_loss_pct: float,
                  drawdown_target: float = 0.50,
                  n_trades: int = 1000,
                  n_paths: int = 5000,
                  rng_seed: int = 42) -> float:
    """
    Estimate probability of hitting a drawdown_target (e.g. 50% loss) via simulation.
    """
    if kelly_fraction <= 0:
        return 0.0
    rng    = np.random.default_rng(rng_seed)
    p      = win_rate / 100.0
    ruined = 0
    for _ in range(n_paths):
        equity = 1.0
        peak   = 1.0
        for _ in range(n_trades):
            if rng.random() < p:
                equity *= (1 + kelly_fraction * avg_win_pct / 100)
            else:
                equity *= (1 - kelly_fraction * avg_loss_pct / 100)
            peak = max(peak, equity)
            if equity <= 0 or (peak - equity) / peak >= drawdown_target:
                ruined += 1
                break
    return ruined / n_paths


# ── Monte Carlo chart ─────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _monte_carlo(kelly_fraction: float, win_rate: float,
                 avg_win_pct: float, avg_loss_pct: float,
                 n_trades: int = 100, n_paths: int = 1000) -> np.ndarray:
    """
    Returns (n_paths, n_trades+1) equity-curve matrix starting at 1.0.
    """
    rng    = np.random.default_rng(99)
    p      = win_rate / 100.0
    paths  = np.ones((n_paths, n_trades + 1))
    for t in range(1, n_trades + 1):
        wins      = rng.random(n_paths) < p
        rets      = np.where(wins,
                             kelly_fraction * avg_win_pct   / 100,
                             -kelly_fraction * avg_loss_pct / 100)
        paths[:, t] = paths[:, t - 1] * (1 + rets)
        paths[:, t] = np.maximum(paths[:, t], 0)  # can't go below 0
    return paths


def _mc_fan_chart(paths: np.ndarray, capital: float) -> go.Figure:
    pcts   = [5, 25, 50, 75, 95]
    labels = ["5th", "25th", "50th", "75th", "95th"]
    colors_fill = [
        "rgba(255,68,102,0.10)",
        "rgba(255,184,0,0.10)",
        "rgba(0,212,255,0.12)",
        "rgba(0,255,136,0.12)",
        "rgba(0,255,136,0.06)",
    ]
    line_colors = [RED, AMBER, CYAN, GREEN, GREEN]

    n_trades = paths.shape[1] - 1
    x = list(range(n_trades + 1))
    percentiles = np.percentile(paths, pcts, axis=0) * capital  # convert to ₹

    fig = go.Figure()

    # Shaded bands between consecutive percentiles
    for i in range(len(pcts) - 1):
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=list(percentiles[i + 1]) + list(percentiles[i][::-1]),
            fill="toself",
            fillcolor=colors_fill[i],
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Percentile lines
    for i, (pct, label, col) in enumerate(zip(pcts, labels, line_colors)):
        fig.add_trace(go.Scatter(
            x=x, y=percentiles[i],
            mode="lines",
            name=f"{label} percentile",
            line=dict(color=col, width=1.5 if pct != 50 else 2.5,
                      dash="dot" if pct in [5, 95] else "solid"),
        ))

    fig.add_hline(y=capital, line_color="rgba(255,255,255,0.2)",
                  line_dash="dash", line_width=1)
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text="Monte Carlo Fan Chart (1,000 paths × 100 trades)",
                             font=dict(color=WHITE, size=13))
    layout["height"]  = 350
    layout["yaxis"]   = dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)",
                              tickprefix="₹", zeroline=False)
    fig.update_layout(**layout)
    return fig


def _kelly_comparison_chart(capital: float, full_k: float,
                             half_k: float, quarter_k: float) -> go.Figure:
    labels  = ["Quarter Kelly", "Half Kelly", "Full Kelly"]
    fracs   = [quarter_k, half_k, full_k]
    amounts = [f * capital for f in fracs]
    colors  = [GREEN, CYAN, AMBER if full_k <= 0.3 else RED]

    fig = go.Figure(go.Bar(
        x=labels, y=amounts,
        marker_color=colors, opacity=0.85,
        text=[f"₹{a:,.0f}<br>({f:.1%})" for a, f in zip(amounts, fracs)],
        textposition="outside",
        textfont=dict(color=WHITE, size=11),
    ))
    layout = dict(**_LAYOUT)
    layout["title"]  = dict(text="Suggested Position Sizes", font=dict(color=WHITE, size=13))
    layout["height"]  = 280
    layout["yaxis"]   = dict(color="#8892a4", gridcolor="rgba(255,255,255,0.04)",
                              tickprefix="₹", zeroline=False)
    fig.update_layout(**layout)
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render_kelly_sizer() -> None:
    st.markdown("## 🎯 Kelly Criterion Position Sizer")
    st.caption("Optimal bet sizing · risk of ruin · Monte Carlo simulation")

    # ── Load journal preset ─────────────────────────────────────────────────
    journal_stats = _load_journal_stats()

    # ── Sidebar / input panel ────────────────────────────────────────────────
    st.markdown(
        """<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
           border-radius:14px; padding:1.2rem 1.5rem; margin-bottom:1rem;">
           <span style="font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
           color:#8892a4;">INPUT PARAMETERS</span></div>""",
        unsafe_allow_html=True,
    )

    preset_used = False
    if journal_stats:
        if st.button("📂 Load from Journal Stats", use_container_width=False):
            st.session_state["ks_winrate"]  = journal_stats["win_rate"]
            st.session_state["ks_avg_win"]  = journal_stats["avg_win_pct"]
            st.session_state["ks_avg_loss"] = journal_stats["avg_loss_pct"]
            preset_used = True
        st.caption(
            f"Journal: {journal_stats['win_rate']:.1f}% WR · "
            f"+{journal_stats['avg_win_pct']:.2f}% avg win · "
            f"-{journal_stats['avg_loss_pct']:.2f}% avg loss"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        win_rate = st.number_input(
            "Win Rate (%)", min_value=1.0, max_value=99.0, step=0.5,
            value=st.session_state.get("ks_winrate", 55.0),
            key="ks_winrate",
        )
    with col2:
        avg_win = st.number_input(
            "Avg Win (%)", min_value=0.1, max_value=50.0, step=0.1,
            value=st.session_state.get("ks_avg_win", 2.5),
            key="ks_avg_win",
        )
    with col3:
        avg_loss = st.number_input(
            "Avg Loss (%)", min_value=0.1, max_value=50.0, step=0.1,
            value=st.session_state.get("ks_avg_loss", 1.5),
            key="ks_avg_loss",
        )

    col4, col5 = st.columns(2)
    with col4:
        capital = st.number_input(
            "Capital (₹)", min_value=10_000.0, max_value=1_00_00_000.0,
            step=10_000.0, value=5_00_000.0, format="%.0f",
        )
    with col5:
        max_kelly = st.slider(
            "Max Kelly Fraction (cap)", min_value=0.10, max_value=1.0,
            step=0.05, value=0.25,
        )

    # ── Compute Kelly variants ───────────────────────────────────────────────
    full_kelly    = min(_compute_kelly(win_rate, avg_win, avg_loss), max_kelly)
    half_kelly    = full_kelly / 2
    quarter_kelly = full_kelly / 4

    pos_full    = full_kelly    * capital
    pos_half    = half_kelly    * capital
    pos_quarter = quarter_kelly * capital

    st.markdown("---")

    # ── Danger zone warning ──────────────────────────────────────────────────
    raw_kelly = _compute_kelly(win_rate, avg_win, avg_loss)
    if raw_kelly > 0.30:
        st.error(
            f"🚨 **DANGER ZONE** — Full Kelly is **{raw_kelly:.1%}** which exceeds 30%. "
            "Betting full Kelly at this level leads to catastrophic drawdowns and eventual ruin. "
            "Always use Half Kelly or less in practice.",
            icon="⚠️",
        )
    elif raw_kelly <= 0:
        st.error(
            "🚫 **Negative Kelly** — your parameters imply a negative edge. "
            "Do NOT trade this setup until you improve your win rate or reward:risk.",
        )
        return

    # ── Kelly result cards ───────────────────────────────────────────────────
    st.markdown("### Kelly Variants")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Full Kelly",    f"{full_kelly:.1%}",    f"₹{pos_full:,.0f}")
    k2.metric("Half Kelly",    f"{half_kelly:.1%}",    f"₹{pos_half:,.0f}")
    k3.metric("Quarter Kelly", f"{quarter_kelly:.1%}", f"₹{pos_quarter:,.0f}")
    rr = raw_kelly
    k4.metric("Raw Kelly (uncapped)", f"{rr:.1%}",
              "⚠️ Capped" if rr > max_kelly else "✅ Under cap")

    # ── R:R and edge summary ─────────────────────────────────────────────────
    rr_ratio    = avg_win / avg_loss if avg_loss > 0 else 0
    expectancy  = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
    edge_per100 = expectancy * 100  # per 100 trades

    e1, e2, e3 = st.columns(3)
    e1.metric("Reward : Risk",     f"{rr_ratio:.2f} : 1")
    e2.metric("Expectancy / Trade", f"{expectancy:.2f}%")
    e3.metric("Edge per 100 trades", f"{edge_per100:.1f}%")

    # ── Comparison bar chart ─────────────────────────────────────────────────
    st.plotly_chart(
        _kelly_comparison_chart(capital, full_kelly, half_kelly, quarter_kelly),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Risk of ruin ─────────────────────────────────────────────────────────
    st.markdown("### Risk of Ruin (50% Drawdown)")
    st.caption("Estimated probability of hitting a 50% drawdown before completing 1,000 trades.")

    with st.spinner("Computing risk of ruin…"):
        ror_full    = _risk_of_ruin(full_kelly,    win_rate, avg_win, avg_loss, n_paths=2000)
        ror_half    = _risk_of_ruin(half_kelly,    win_rate, avg_win, avg_loss, n_paths=2000)
        ror_quarter = _risk_of_ruin(quarter_kelly, win_rate, avg_win, avg_loss, n_paths=2000)

    r1, r2, r3 = st.columns(3)

    def _ror_color(p: float) -> str:
        return RED if p > 0.20 else (AMBER if p > 0.05 else GREEN)

    r1.markdown(
        f"""<div style="background:rgba(255,255,255,0.04); border:1px solid {_ror_color(ror_full)}44;
         border-radius:12px; padding:.8rem 1rem; text-align:center;">
         <div style="font-size:.72rem; color:#8892a4; text-transform:uppercase;">Full Kelly RoR</div>
         <div style="font-size:1.8rem; font-weight:800; color:{_ror_color(ror_full)};
              font-family:'JetBrains Mono',monospace;">{ror_full:.1%}</div>
        </div>""", unsafe_allow_html=True,
    )
    r2.markdown(
        f"""<div style="background:rgba(255,255,255,0.04); border:1px solid {_ror_color(ror_half)}44;
         border-radius:12px; padding:.8rem 1rem; text-align:center;">
         <div style="font-size:.72rem; color:#8892a4; text-transform:uppercase;">Half Kelly RoR</div>
         <div style="font-size:1.8rem; font-weight:800; color:{_ror_color(ror_half)};
              font-family:'JetBrains Mono',monospace;">{ror_half:.1%}</div>
        </div>""", unsafe_allow_html=True,
    )
    r3.markdown(
        f"""<div style="background:rgba(255,255,255,0.04); border:1px solid {_ror_color(ror_quarter)}44;
         border-radius:12px; padding:.8rem 1rem; text-align:center;">
         <div style="font-size:.72rem; color:#8892a4; text-transform:uppercase;">Quarter Kelly RoR</div>
         <div style="font-size:1.8rem; font-weight:800; color:{_ror_color(ror_quarter)};
              font-family:'JetBrains Mono',monospace;">{ror_quarter:.1%}</div>
        </div>""", unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Monte Carlo simulation ────────────────────────────────────────────────
    st.markdown("### Monte Carlo Simulation — 1,000 Paths × 100 Trades")

    mc_kelly = st.radio(
        "Kelly variant to simulate",
        options=["Quarter Kelly", "Half Kelly", "Full Kelly"],
        horizontal=True,
        index=1,
    )
    sim_fraction = {"Quarter Kelly": quarter_kelly, "Half Kelly": half_kelly,
                    "Full Kelly": full_kelly}[mc_kelly]

    with st.spinner("Running Monte Carlo…"):
        paths = _monte_carlo(sim_fraction, win_rate, avg_win, avg_loss,
                             n_trades=100, n_paths=1000)

    fig_mc = _mc_fan_chart(paths, capital)
    st.plotly_chart(fig_mc, use_container_width=True)

    # Outcome stats
    final_equities = paths[:, -1] * capital
    median_final   = float(np.median(final_equities))
    p5_final       = float(np.percentile(final_equities, 5))
    p95_final      = float(np.percentile(final_equities, 95))
    pct_profitable = float(np.mean(final_equities > capital))
    pct_ruin       = float(np.mean(final_equities < capital * 0.5))

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Median (100 trades)", f"₹{median_final:,.0f}",
              f"{(median_final/capital - 1)*100:+.1f}%")
    s2.metric("5th Percentile",      f"₹{p5_final:,.0f}")
    s3.metric("95th Percentile",     f"₹{p95_final:,.0f}")
    s4.metric("% Paths Profitable",  f"{pct_profitable:.0%}")
    s5.metric("% Paths Ruined (<50%)", f"{pct_ruin:.0%}")

    # ── Methodology note ─────────────────────────────────────────────────────
    with st.expander("ℹ️ Methodology"):
        st.markdown(
            f"""
**Kelly Formula:**
```
Full Kelly = (W/L × p − q) / (W/L)
```
where `p` = win rate, `q` = 1 − p, `W` = avg win %, `L` = avg loss %.

**Current inputs:** p={win_rate:.1f}%, W={avg_win:.2f}%, L={avg_loss:.2f}%
→ Full Kelly = **{raw_kelly:.2%}** (capped at {max_kelly:.0%} → **{full_kelly:.2%}**)

**Risk of Ruin:** Simulated over 1,000 trades across 2,000 paths.
Ruin = drawdown exceeds 50% from peak.

**Monte Carlo:** Each trade randomly wins (probability = win rate) or loses,
sizing each bet as `kelly_fraction × current_equity × avg_win/loss %`.
Fan bands show 5th / 25th / 50th / 75th / 95th percentile equity curves.

> **Rule of thumb:** Never bet more than Half Kelly. Quarter Kelly gives ~75% of
> the expected geometric growth rate with dramatically lower variance.
            """
        )
