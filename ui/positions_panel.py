"""
Positions Panel — shared component: the user's book, one source of truth.

Used by the Smart Scanner AND the My Portfolio page so both surfaces
show identical reality: portfolio risk meter, open positions with
R-progress + exit advice, recent trade journal.
"""
from __future__ import annotations

import streamlit as st


def render_risk_meter() -> None:
    """Account-level risk strip (OK/CAUTION/DANGER)."""
    try:
        from risk.portfolio_risk import portfolio_risk_report
        prr = portfolio_risk_report()
        if not prr["n_positions"]:
            return
        vc = {"OK": "#00d4a0", "CAUTION": "#f59e0b",
              "DANGER": "#ff4b4b"}[prr["verdict"]]
        st.markdown(
            f"<div style='background:{vc}12;border:1px solid {vc}44;"
            f"border-radius:8px;padding:8px 14px;margin-bottom:8px;"
            f"font-size:.8rem;color:#c9d1d9'>"
            f"<b style='color:{vc}'>Portfolio: {prr['verdict']}</b> · "
            f"deployed ₹{prr['deployed']:,.0f} ({prr['deployed_pct']:.0f}%) · "
            f"agar saare stops hit hue: <b style='color:#ff4b4b'>"
            f"−₹{prr['open_risk']:,.0f}</b> ({prr['open_risk_pct']:.1f}% "
            f"of capital) · {prr['n_positions']}/{prr['max_positions']} positions"
            + "".join(f"<div style='color:#f59e0b;margin-top:3px'>{w}</div>"
                      for w in prr["warnings"])
            + "</div>", unsafe_allow_html=True)
    except Exception:
        pass


def render_open_positions() -> int:
    """Open trades with live R-progress + advice. Returns count rendered."""
    try:
        from risk.position_manager import review_positions
        pos = review_positions()
    except Exception:
        pos = []
    if not pos:
        return 0
    render_risk_meter()
    for p in pos:
        r = p.get("r_progress")
        pnl = p.get("pnl")
        pnl_col = "#00d4a0" if (pnl or 0) >= 0 else "#ff4b4b"
        live_txt = f"₹{p['live']:,.1f}" if p["live"] else "—"
        st.markdown(
            f"<div style='background:#0d1421;border:1px solid #1e293b;"
            f"border-radius:10px;padding:10px 14px;margin-bottom:6px'>"
            f"<span style='color:#e2e8f0;font-weight:700;"
            f"font-family:JetBrains Mono,monospace'>{p['symbol']}</span>"
            f"<span style='font-size:.72rem;color:#8892a4'> "
            f"({p['mode']}{' · 🤖' if p.get('autopilot') else ''}) · "
            f"{p['qty']} sh @ ₹{p['entry']:,.1f}</span>"
            f" &nbsp; live <b style='color:#e2e8f0'>{live_txt}</b>"
            + (f" &nbsp; <b style='color:{pnl_col}'>₹{pnl:+,.0f}</b>"
               if pnl is not None else "")
            + (f" &nbsp; <span style='font-size:.75rem;color:#7dd3fc'>"
               f"{r:+.1f}R</span>" if r is not None else "")
            + f"<div style='font-size:.8rem;color:#c9d1d9;margin-top:5px'>"
              f"{p['advice']}</div>"
            + "</div>",
            unsafe_allow_html=True)
    return len(pos)


def render_trade_journal(limit: int = 10) -> None:
    """Recent trade journal table (paper + live)."""
    try:
        from execution.trade_executor import recent_trades
        rt = recent_trades(limit)
    except Exception:
        rt = []
    if not rt:
        return
    with st.expander(f"📒 Mere trades ({len(rt)} recent)"):
        st.dataframe(
            [{"Time": t["placed_at"][5:16], "Mode": t["mode"],
              "Stock": t["symbol"], "Qty": t["qty"],
              "Entry": f"₹{t['entry_price']:,.1f}" if t["entry_price"] else "-",
              "Stop": f"₹{t['stop_price']:,.1f}" if t["stop_price"] else "-",
              "Target": f"₹{t['target_price']:,.1f}" if t["target_price"] else "-",
              "Status": t["status"]} for t in rt],
            width="stretch", hide_index=True)
