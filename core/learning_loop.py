"""
Learning Loop — adapts to YOUR trading edge over time.

Tracks performance by: archetype × regime × quality_tier
Computes your personal win rate vs system baseline.
Weights future recommendations toward your proven edges.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── DB path ───────────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "logs" / "learning_loop.db"

# Universal baseline win rate used when quality_engine rates are unavailable
_UNIVERSAL_BASELINE = 0.55


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_outcomes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol    TEXT,
            archetype TEXT,
            regime    TEXT,
            tier      TEXT,
            entry     REAL,
            exit      REAL,
            pnl_r     REAL,
            won       INTEGER
        )
        """
    )
    conn.commit()
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def record_trade_outcome(
    symbol: str,
    archetype: str,
    regime: str,
    tier: str,
    entry: float,
    exit: float,
    pnl_r: float,
) -> None:
    """Store a completed trade outcome to the learning-loop database."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO trade_outcomes
                (timestamp, symbol, archetype, regime, tier, entry, exit, pnl_r, won)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                symbol,
                archetype,
                regime,
                tier,
                entry,
                exit,
                pnl_r,
                1 if pnl_r > 0 else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _get_baseline(archetype: str) -> float:
    """
    Try to read historical baseline win rate from QualityEngine playbooks.
    Falls back to _UNIVERSAL_BASELINE if unavailable.
    """
    try:
        from playbooks import get_all_playbooks  # type: ignore
        for pb in get_all_playbooks():
            if pb.archetype == archetype or getattr(pb, "id", "") == archetype:
                return float(pb.baseline_win_rate)
    except Exception:
        pass
    return _UNIVERSAL_BASELINE


def get_personal_edge(min_trades: int = 5) -> dict[str, dict]:
    """
    Returns your personal edge per archetype.

    Shape::

        {
          "VCP_BREAKOUT": {
              "win_rate":    0.73,
              "avg_r":       1.4,
              "trades":      11,
              "vs_baseline": +0.18,   # positive = you outperform
          },
          ...
        }

    Only archetypes with >= min_trades are included.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT archetype,
                   COUNT(*)        AS trades,
                   SUM(won)        AS wins,
                   AVG(pnl_r)      AS avg_r
            FROM   trade_outcomes
            GROUP  BY archetype
            HAVING trades >= ?
            """,
            (min_trades,),
        ).fetchall()
    finally:
        conn.close()

    result: dict[str, dict] = {}
    for archetype, trades, wins, avg_r in rows:
        wr = wins / trades if trades else 0.0
        baseline = _get_baseline(archetype)
        result[archetype] = {
            "win_rate":    round(wr, 4),
            "avg_r":       round(avg_r or 0.0, 4),
            "trades":      trades,
            "vs_baseline": round(wr - baseline, 4),
        }
    return result


def get_top_edges(n: int = 3) -> list[dict]:
    """
    Returns your top N archetypes by personal win rate (min 5 trades).
    Each dict has keys: archetype, win_rate, avg_r, trades, vs_baseline.
    """
    edge = get_personal_edge(min_trades=5)
    ranked = sorted(edge.items(), key=lambda kv: kv[1]["win_rate"], reverse=True)
    return [{"archetype": k, **v} for k, v in ranked[:n]]


def get_regime_edge(regime: str) -> dict[str, float | int]:
    """
    Returns your aggregate win stats for a specific market regime.

    Returns::

        {"win_rate": float, "avg_r": float, "trades": int}
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS trades,
                   SUM(won)  AS wins,
                   AVG(pnl_r) AS avg_r
            FROM   trade_outcomes
            WHERE  regime = ?
            """,
            (regime,),
        ).fetchone()
    finally:
        conn.close()

    if row is None or row[0] == 0:
        return {"win_rate": 0.0, "avg_r": 0.0, "trades": 0}

    trades, wins, avg_r = row
    wr = (wins or 0) / trades if trades else 0.0
    return {
        "win_rate": round(wr, 4),
        "avg_r":    round(avg_r or 0.0, 4),
        "trades":   trades,
    }


# ── Streamlit UI Component ────────────────────────────────────────────────────

def render_personal_edge_ui() -> None:
    """
    Streamlit component showing your personal trading edge analytics.
    Displays:
      1. Header
      2. Bar chart — personal WR vs baseline per archetype
      3. Text summary of strongest / weakest archetypes
      4. Regime performance table
    """
    import streamlit as st

    st.markdown(
        "<h3 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px'>"
        "🧠 Your Personal Trading Edge</h3>"
        "<span style='color:#4a5568;font-size:.65rem'>"
        "Adapts as you trade — data from logs/learning_loop.db</span>",
        unsafe_allow_html=True,
    )

    edge = get_personal_edge(min_trades=5)

    if not edge:
        st.markdown(
            "<div style='background:#111827;border:1px solid #1e293b;border-radius:10px;"
            "padding:24px;text-align:center;color:#4a5568;font-size:.8rem'>"
            "No data yet — record at least 5 completed trades per archetype to see your edge."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Bar chart ─────────────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go

        archetypes = list(edge.keys())
        personal_wrs = [edge[a]["win_rate"] * 100 for a in archetypes]
        baselines    = [(_get_baseline(a)) * 100 for a in archetypes]
        arch_labels  = [a.replace("_", " ") for a in archetypes]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Your Win Rate",
            x=arch_labels,
            y=personal_wrs,
            marker_color="#00d4a0",
            opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="System Baseline",
            x=arch_labels,
            y=baselines,
            marker_color="#4a5568",
            opacity=0.65,
        ))
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=280,
            margin=dict(l=10, r=10, t=20, b=60),
            legend=dict(
                orientation="h", y=1.08, x=0,
                font=dict(size=9, color="#8892a4"),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                tickfont=dict(size=9, color="#8892a4"),
                showgrid=False,
                tickangle=-25,
            ),
            yaxis=dict(
                title=dict(text="Win Rate %", font=dict(size=9, color="#4a5568")),
                showgrid=True,
                gridcolor="#1e293b",
                tickfont=dict(size=9, color="#4a5568"),
            ),
        )
        st.plotly_chart(fig, width="stretch", key="learning_loop_bar_chart")
    except Exception as exc:
        st.caption(f"Chart unavailable: {exc}")

    # ── Text summary ──────────────────────────────────────────────────────────
    ranked = sorted(edge.items(), key=lambda kv: kv[1]["win_rate"], reverse=True)
    if ranked:
        best_arch, best_data  = ranked[0]
        worst_arch, worst_data = ranked[-1]

        best_wr   = best_data["win_rate"] * 100
        best_vs   = best_data["vs_baseline"] * 100
        worst_wr  = worst_data["win_rate"] * 100
        worst_vs  = worst_data["vs_baseline"] * 100

        best_label  = best_arch.replace("_", " ").title()
        worst_label = worst_arch.replace("_", " ").title()

        best_sign  = "+" if best_vs >= 0 else ""
        worst_sign = "+" if worst_vs >= 0 else ""

        st.markdown(
            f"<div style='background:#111827;border:1px solid #1e293b;border-radius:8px;"
            f"padding:10px 14px;margin:8px 0;font-size:.78rem;color:#c9d1e0;line-height:1.6'>"
            f"You're strongest in <span style='color:#00d4a0;font-weight:700'>{best_label}</span> "
            f"(<span style='color:#00d4a0'>{best_wr:.0f}% WR, {best_sign}{best_vs:.0f}% vs baseline</span>). "
            f"Weakest in <span style='color:#ff4b4b;font-weight:700'>{worst_label}</span> "
            f"(<span style='color:#ff4b4b'>{worst_wr:.0f}% WR, {worst_sign}{worst_vs:.0f}% vs baseline</span>)."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Regime performance table ───────────────────────────────────────────────
    st.markdown(
        "<div style='color:#8892a4;font-size:.65rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:.1em;margin:12px 0 6px'>Regime Performance</div>",
        unsafe_allow_html=True,
    )

    conn = _get_conn()
    try:
        regime_rows = conn.execute(
            """
            SELECT regime,
                   COUNT(*)   AS trades,
                   SUM(won)   AS wins,
                   AVG(pnl_r) AS avg_r
            FROM   trade_outcomes
            GROUP  BY regime
            ORDER  BY trades DESC
            """
        ).fetchall()
    finally:
        conn.close()

    if not regime_rows:
        st.caption("No regime data yet.")
        return

    header = (
        "<div style='display:grid;grid-template-columns:1fr 80px 80px 80px;"
        "gap:6px;padding:5px 10px;font-size:.58rem;color:#4a5568;"
        "text-transform:uppercase;letter-spacing:.08em;font-weight:700;"
        "border-bottom:1px solid #1e293b;margin-bottom:4px'>"
        "<span>Regime</span>"
        "<span style='text-align:center'>Trades</span>"
        "<span style='text-align:center'>Win Rate</span>"
        "<span style='text-align:center'>Avg R</span>"
        "</div>"
    )
    st.markdown(header, unsafe_allow_html=True)

    for regime_name, trades, wins, avg_r in regime_rows:
        wr = (wins or 0) / trades if trades else 0.0
        wr_pct = wr * 100
        wr_col = "#00d4a0" if wr_pct >= 55 else "#f59e0b" if wr_pct >= 45 else "#ff4b4b"
        avg_r_val = avg_r or 0.0
        avg_r_col = "#00d4a0" if avg_r_val > 0 else "#ff4b4b"
        regime_plain = regime_name.replace("_", " ")

        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 80px 80px 80px;"
            f"gap:6px;padding:6px 10px;background:#111827;border:1px solid #1e293b;"
            f"border-radius:6px;margin-bottom:3px;align-items:center'>"
            f"<span style='font-size:.72rem;color:#c9d1e0'>{regime_plain}</span>"
            f"<span style='font-size:.72rem;color:#8892a4;text-align:center'>{trades}</span>"
            f"<span style='font-size:.72rem;color:{wr_col};font-weight:700;text-align:center'>"
            f"{wr_pct:.0f}%</span>"
            f"<span style='font-size:.72rem;color:{avg_r_col};font-weight:700;text-align:center'>"
            f"{avg_r_val:+.2f}R</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
