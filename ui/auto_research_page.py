"""
🧠 Autonomous Research Brain — the "watch it think" page.

A thin, read-only window into the brain. It shows the live chain-of-thought (the append-only
research thread), the proposals parked at the human gate, and what the brain has learned is
improving or decaying. You can let it think one cycle now, or start/stop the headless loop.

It has NO order actions and cannot approve anything. Approving a proposal for PAPER happens
in Strategy Studio, by a person — never here, and never automatically.
"""
from __future__ import annotations

import streamlit as st

from research.auto_research.scheduler import get_brain
from research.auto_research import loop as L

_KIND_ICON = {"OBSERVE": "👁️", "REASON": "🧩", "DECIDE": "⚖️", "PROPOSE": "📌",
              "CONCLUDE": "✅"}


def _readiness_line() -> dict:
    try:
        return L.canonical_readiness()
    except Exception as e:
        return {"color": "red", "can_run": False, "reasons": [str(e)]}


def render_auto_research() -> None:
    st.markdown("## 🧠 Autonomous Research Brain")
    st.caption("The system studies the market on its own — generating ideas, reasoning "
               "through them out loud, rejecting the weak ones, and proposing improvements. "
               "It stops at one gate: a **human** must approve anything. It never trades.")
    st.info("Nothing here places an order or approves a strategy. The brain parks its best "
            "ideas for you to review in **Strategy Studio**. Live stays locked.")

    brain = get_brain()

    r = _readiness_line()
    colour = {"green": "#00d4a0", "amber": "#f59e0b", "red": "#ff4b4b"}.get(r["color"], "#888")
    st.markdown(
        f"<div style='background:{colour}18;border-left:5px solid {colour};"
        f"border-radius:10px;padding:.6rem .9rem;margin:.4rem 0'>"
        f"<b style='color:{colour}'>Research data: {r['color'].upper()}</b> — "
        f"{'ready to think on real history' if r.get('can_run') else 'not ready; the brain will say so honestly'}"
        f"</div>", unsafe_allow_html=True)
    if not r.get("can_run"):
        for reason in (r.get("reasons") or [])[:3]:
            st.markdown(f"- {reason}")
        st.caption("Load real NSE history in 🗂️ Historical Data Setup, then the brain has "
                   "something honest to study.")

    # ── controls ─────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💭 Think one cycle now"):
            with st.spinner("Reasoning…"):
                try:
                    brain.run_once()
                except Exception as e:
                    st.error(f"Cycle failed safely (nothing traded): {e}")
    with c2:
        if not brain.state.running and st.button("▶️ Start thinking loop"):
            brain.start(); st.success("Brain is now thinking on a schedule.")
    with c3:
        if brain.state.running and st.button("⏸️ Pause loop"):
            brain.stop(); st.info("Paused. Nothing was pending.")

    st.markdown(f"**Cycles run:** {brain.state.cycles_run} · "
                f"**Proposals parked:** {brain.state.total_proposals} · "
                f"**Loop:** {'running' if brain.state.running else 'idle'}")
    if brain.state.last_error:
        st.warning(f"Last cycle note: {brain.state.last_error}")

    # ── proposals parked at the human gate ───────────────────────────────────────
    rep = brain.state.last_report
    st.markdown("### 📌 Proposals waiting for you")
    props = (rep or {}).get("proposals", [])
    if not props:
        st.caption("None yet. When the brain finds something worth a human look, it appears "
                   "here — and you decide in Strategy Studio.")
    for p in props:
        st.markdown(
            f"- **{p['name']}** ({p['family']}) — {p['net_expectancy_R']:+.2f}R over "
            f"{p['n_trades']} trades · _{p['recommendation']}_ · at gate "
            f"`{p['lifecycle_state']}`")
    if props:
        st.caption("These are parked, not approved. Open **🧪 Strategy Studio** to review the "
                   "full case and decide. The brain will not approve them for you.")

    # ── the chain-of-thought thread ──────────────────────────────────────────────
    st.markdown("### 🧵 What the brain is thinking (newest last)")
    entries = brain.thread.all()
    if not entries:
        st.caption("Empty — press **Think one cycle now** to watch it reason.")
    else:
        last = brain.thread.last_cycle()
        show = [e for e in entries if e.cycle >= max(1, last - 1)]   # last ~2 cycles
        for e in show:
            icon = _KIND_ICON.get(e.kind, "•")
            st.markdown(f"{icon} _cycle {e.cycle}_ · **{e.kind}** — {e.text}")
        with st.expander("Full thread (technical)"):
            st.json([e.as_dict() for e in entries])

    # ── learning: what's improving / decaying ────────────────────────────────────
    st.markdown("### 📈 What it has learned")
    tracks = brain.ledger.tracks()
    if not tracks:
        st.caption("Nothing tracked yet — the brain learns across cycles once it has "
                   "market-evidence proposals.")
    for fam, t in sorted(tracks.items()):
        trend = "▲ improving" if t.last_R >= t.best_R else "▼ fading"
        st.markdown(f"- **{fam}**: best {t.best_R:+.2f}R, now {t.last_R:+.2f}R "
                    f"({trend}, {t.observations} looks)")

    st.divider()
    st.caption("Autonomy with a seatbelt: the brain does the whole research loop by itself "
               "and stops before risking money. By design, only a person can approve a "
               "strategy for paper — and live remains locked.")
