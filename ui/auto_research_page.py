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
        if st.button("💭 Request one research cycle"):
            from research.autonomy.controls import request_control, RUN_RESEARCH_NOW
            request_control(RUN_RESEARCH_NOW, reason="owner requested bounded research cycle")
            st.success("Research cycle queued for the autonomy supervisor.")
    with c2:
        st.caption("The dedicated autonomy service owns the schedule.")
    with c3:
        st.caption("Pause or resume new entries from Automatic Paper Trading.")

    st.markdown(f"**Cycles run:** {brain.state.cycles_run} · "
                f"**Proposals parked:** {brain.state.total_proposals} · "
                f"**Loop:** {'running' if brain.state.running else 'idle'}")
    if brain.state.last_error:
        st.warning(f"Last cycle note: {brain.state.last_error}")

    _paper_autonomy_panel(brain)

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
    st.caption("Full PAPER autonomy trades only SIMULATED money and can never reach live — "
               "the paper autopilot is structurally barred from the live-review step. "
               "Moving a strategy toward LIVE still requires a person, and live remains "
               "migration-locked.")


def _paper_autonomy_panel(brain) -> None:
    st.markdown("### 🤖 Full paper autonomy")
    engaged = brain.state.paper_autonomy
    if engaged:
        st.success("ENGAGED — the brain deploys survivors to paper, trades them, and retires "
                   "proven losers on its own. Only simulated money is at risk.")
        gc1, gc2 = st.columns(2)
        with gc1:
            if st.button("🌱 Request one learning/research cycle", type="primary"):
                from research.autonomy.controls import request_control, RUN_RESEARCH_NOW
                request_control(RUN_RESEARCH_NOW, reason="owner requested bounded learning/research cycle")
                st.success("Research cycle queued for the autonomy supervisor.")
        with gc2:
            if st.button("⏹️ Pause new paper entries"):
                from research.autonomy.controls import request_control, PAUSE_NEW_PAPER_ENTRIES
                request_control(PAUSE_NEW_PAPER_ENTRIES, reason="owner paused paper entries from research page")
                st.success("Pause request queued. Existing positions remain manageable.")
        st.caption(f"Days grown: **{getattr(brain.state,'days_grown',0)}** · last: "
                   f"`{getattr(brain.state,'last_grow_date','—') or '—'}`. Each day it "
                   "backtests fresh ideas, forward-tests survivors in paper, and keeps only "
                   "what holds up out-of-sample.")
    else:
        st.caption("Off. When on, the brain auto-approves real-data survivors for PAPER, "
                   "places simulated trades, learns from the outcomes, and retires losers — "
                   "no human in the loop. It can 'blow up' paper money; it can never touch "
                   "live (that step stays user-only).")
        if st.button("▶️ Engage full paper autonomy", type="primary"):
            from research.autonomy.controls import request_control, ENABLE_PAPER_AUTO, RESUME_NEW_PAPER_ENTRIES
            request_control(ENABLE_PAPER_AUTO, reason="owner enabled PAPER_AUTO from research page")
            request_control(RESUME_NEW_PAPER_ENTRIES, reason="owner resumed paper entries from research page")
            st.success("PAPER_AUTO controls queued for the autonomy supervisor.")

    if not (engaged or brain.paper.strategies):
        return
    rep = brain.paper.performance_report()
    book = rep["book"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paper equity", f"₹{book['equity']:,.0f}",
              f"{book['equity'] - book['capital']:+,.0f}")
    c2.metric("Deployed", rep["deployed"])
    c3.metric("Active", rep["active"])
    c4.metric("Retired (losers)", len(rep["retired"]))
    if rep["per_strategy"]:
        st.markdown("**Per-strategy paper performance**")
        for sid, s in rep["per_strategy"].items():
            st.markdown(f"- `{sid}` — {s['n_trades']} trades · {s['expectancy_R']:+.2f}R · "
                        f"win {s['win_rate']:.0%} · net ₹{s['net_pnl']:,.0f}")
    with st.expander("Technical — paper book"):
        st.json(rep)
