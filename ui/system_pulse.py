"""
🫀 System Pulse — one glance: is every daemon alive, how fast is the
machinery. Renders inside the sidebar Diagnostics expander.
"""
from __future__ import annotations

import streamlit as st

_DOT = {"OK": "🟢", "SLOW": "🟡", "DEAD": "🔴", "NEVER": "⚪"}
_LABEL = {
    "auto_scan":         "Market scan",
    "autopilot":         "Autopilot",
    "sniper":            "Breakout sniper",
    "live_ticker":       "Live Watch stream",
    "telegram_listener": "Telegram buttons",
    "quotes":            "Quote source",
}


def _age_str(age_s: float) -> str:
    if age_s < 0:
        return "kabhi nahi chala"
    if age_s < 90:
        return f"{age_s:.0f}s ago"
    if age_s < 5400:
        return f"{age_s / 60:.0f}m ago"
    return f"{age_s / 3600:.1f}h ago"


def render_system_pulse() -> None:
    try:
        from core.health import pulse
        p = pulse()
    except Exception:
        st.caption("pulse unavailable")
        return

    daemons = p.get("daemons", {})
    if not daemons:
        st.caption("🫀 Pulse: daemons abhi report nahi kar rahe "
                   "(pehla cycle chalne do)")
        return

    for name in ("auto_scan", "autopilot", "sniper", "live_ticker",
                 "telegram_listener", "quotes"):
        d = daemons.get(name)
        if not d:
            continue
        dot = _DOT.get(d["status"], "⚪")
        note = f" · {d['note']}" if d.get("note") else ""
        st.caption(f"{dot} {_LABEL.get(name, name)} — "
                   f"{_age_str(d['age_s'])}{note}")

    lat = p.get("latency", {})
    bits = []
    if "scan_cycle" in lat:
        bits.append(f"scan p50 {lat['scan_cycle']['p50_ms'] / 1000:.1f}s")
    if "quote_fetch" in lat:
        q = lat["quote_fetch"]
        bits.append(f"quotes p50 {q['p50_ms']:.0f}ms / p95 {q['p95_ms']:.0f}ms")
    qc = p.get("counters", {}).get("quote_cache", {})
    if qc:
        hits, misses = qc.get("hits", 0), qc.get("misses", 0)
        total = hits + misses
        if total:
            bits.append(f"quote-cache {hits / total * 100:.0f}% hit")
    if bits:
        st.caption("⚡ " + " · ".join(bits))
