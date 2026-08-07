"""
IronLock Status Widget — shows the 6 deterministic pre-trade gates as a compact badge row.

Usage:
    from ui.ironlock_widget import render_ironlock_status, is_all_clear
    render_ironlock_status()
    if is_all_clear(): ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import streamlit as st


def _get_gate_statuses() -> List[Tuple[str, bool, str]]:
    """
    Evaluate all 6 IronLock gates from session state + portfolio log.

    Returns:
        List of (gate_name, is_open: bool, reason_if_blocked: str)
    """
    kill_switch = st.session_state.get("kill_switch_active", False)
    # Also check the key used by the Terminal kill button
    if not kill_switch:
        kill_switch = st.session_state.get("kill_switch", False)

    # Try to read portfolio state from disk
    pf_path = Path("logs/portfolio_state.json")
    pf: dict = {}
    if pf_path.exists():
        try:
            pf = json.loads(pf_path.read_text())
        except Exception:
            pass

    daily_drawdown = float(pf.get("daily_drawdown_pct", 0.0))
    consecutive_losses = int(pf.get("consecutive_losses", 0))
    open_positions = len(pf.get("positions", {}))

    # Get regime confidence from session cache or regime engine
    regime = st.session_state.get("cached_regime", {})
    if not regime:
        try:
            from ui.regime_bar import get_regime
            regime = get_regime()
        except Exception:
            regime = {}
    regime_confidence = float(regime.get("confidence", 100.0))
    # regime_score is a 0-100 float; treat it as percent
    if regime_confidence == 100.0 and "regime_score" in regime:
        regime_confidence = float(regime["regime_score"])

    gates: List[Tuple[str, bool, str]] = [
        (
            "Kill Switch",
            not kill_switch,
            "Active — all trades blocked" if kill_switch else "",
        ),
        (
            "Regime",
            regime_confidence >= 40,
            f"Confidence {regime_confidence:.0f}% < 40%" if regime_confidence < 40 else "",
        ),
        (
            "Daily Loss",
            daily_drawdown < 5.0,
            f"Down {daily_drawdown:.1f}% today" if daily_drawdown >= 5.0 else "",
        ),
        (
            "F&O Ban",
            True,  # Cannot reliably check without Kite live session — default open
            "",
        ),
        (
            "Loss Streak",
            consecutive_losses < 3,
            f"{consecutive_losses} consecutive losses" if consecutive_losses >= 3 else "",
        ),
        (
            "Max Positions",
            open_positions < 5,
            f"{open_positions}/5 positions open" if open_positions >= 5 else "",
        ),
    ]
    return gates


def is_all_clear() -> bool:
    """Return True if every IronLock gate is open (no gate is blocked)."""
    return all(is_open for _, is_open, _ in _get_gate_statuses())


def render_ironlock_status() -> None:
    """
    Renders a compact one-line row of 6 IronLock gate badges.

    - 🟢 Gate Name      — gate is open (trade allowed)
    - 🔴 Gate Name: reason — gate is blocked

    Call this after render_regime_bar() for a persistent status strip.
    """
    gates = _get_gate_statuses()
    all_ok = all(is_open for _, is_open, _ in gates)

    badges_html = []
    for name, is_open, reason in gates:
        if is_open:
            badge = (
                f"<span style='display:inline-block;background:#052e16;color:#4ade80;"
                f"border:1px solid #166534;border-radius:4px;padding:2px 8px;"
                f"font-size:.70rem;font-weight:600;font-family:JetBrains Mono,monospace;"
                f"white-space:nowrap;margin-right:3px'>🟢 {name}</span>"
            )
        else:
            detail = f": {reason}" if reason else ""
            badge = (
                f"<span style='display:inline-block;background:#2d0a0a;color:#f87171;"
                f"border:1px solid #7f1d1d;border-radius:4px;padding:2px 8px;"
                f"font-size:.70rem;font-weight:600;font-family:JetBrains Mono,monospace;"
                f"white-space:nowrap;margin-right:3px' title='{reason}'>🔴 {name}{detail}</span>"
            )
        badges_html.append(badge)

    summary_color = "#4ade80" if all_ok else "#f87171"
    summary_label = "ALL CLEAR" if all_ok else "BLOCKED"
    summary_badge = (
        f"<span style='display:inline-block;background:{'#052e16' if all_ok else '#2d0a0a'};"
        f"color:{summary_color};border:1px solid {'#166534' if all_ok else '#7f1d1d'};"
        f"border-radius:4px;padding:2px 8px;font-size:.70rem;font-weight:700;"
        f"font-family:JetBrains Mono,monospace;white-space:nowrap;margin-right:6px;"
        f"letter-spacing:.05em'>⚡ {summary_label}</span>"
    )

    row_html = (
        "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:1px;"
        "padding:3px 0;border-top:1px solid rgba(255,255,255,0.05);margin-top:2px'>"
        "<span style='font-size:.65rem;color:#6b7280;font-family:JetBrains Mono,monospace;"
        "margin-right:6px;white-space:nowrap'>🔒 IronLock</span>"
        + summary_badge
        + "".join(badges_html)
        + "</div>"
    )
    st.markdown(row_html, unsafe_allow_html=True)
