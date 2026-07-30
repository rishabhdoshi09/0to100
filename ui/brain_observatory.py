"""
🔭 Brain Observatory — see the two brains think, separately, and the human live-review queue.

Three tabs:
  • Automatic Strategies — what's running in paper, allocations, forward results, evidence state.
  • The Two Brains — Brain 1 (what it believes) vs Brain 2 (experiments & allocation moves).
  • Live Review Candidates — a HUMAN review queue (never an auto-buy), with honest empty states.

Read-only. It renders whatever the intelligence layer has produced; with no real data it shows
honest empty states and never fabricates candidates or evidence.
"""
from __future__ import annotations

import streamlit as st

from ui import theme as T


def _brain():
    try:
        from research.auto_research import get_brain
        return get_brain()
    except Exception:
        return None


def _cards():
    """Latest evidence cards from the canonical event store, if one has been populated."""
    try:
        from pathlib import Path
        from research.intelligence.event_store import EventStore
        p = Path(__file__).resolve().parent.parent / "logs" / "intelligence" / "events.jsonl"
        if not p.exists():
            return []
        return list(EventStore(p).latest_cards().values())
    except Exception:
        return []


def _empty(msg: str) -> None:
    st.markdown(f"<div class='qt-card' style='color:var(--qt-muted);text-align:center;"
                f"padding:1.6rem'>{msg}</div>", unsafe_allow_html=True)


def render_brain_observatory() -> None:
    st.markdown("## 🔭 Brain Observatory")
    st.caption("Two brains, separated: Brain 1 interprets evidence, Brain 2 allocates paper "
               "risk. They talk only through immutable evidence records. **Paper-only — the "
               "live door is user-owned.**")
    t1, t2, t3 = st.tabs(["🤖 Automatic Strategies", "🧠 The Two Brains",
                          "🎓 Live Review Candidates"])
    with t1:
        _automatic_strategies()
    with t2:
        _two_brains()
    with t3:
        _live_review()


def _automatic_strategies() -> None:
    brain = _brain()
    T.section("Running in paper", eyebrow="Hands-off",
              sub="Strategies the system is forward-testing with simulated money.")
    if brain is None:
        return _empty("The research brain isn't available in this session.")
    try:
        rep = brain.paper.performance_report()
    except Exception:
        rep = None
    if not rep or not rep.get("deployed"):
        return _empty("No strategies are running in paper yet. With no research-grade NSE "
                      "data loaded, the system deploys nothing — by design.")
    book = rep["book"]
    T.stat_grid([
        {"k": "Paper equity", "v": f"₹{book['equity']:,.0f}",
         "d": f"{book['equity'] - book['capital']:+,.0f}"},
        {"k": "Deployed", "v": rep["deployed"]},
        {"k": "Active", "v": rep["active"]},
        {"k": "Retired", "v": len(rep["retired"])},
    ], cols=4)
    for sid, s in rep.get("per_strategy", {}).items():
        col = T.GREEN if s["expectancy_R"] > 0 else T.RED
        st.markdown(f"- `{sid}` — {s['n_trades']} trades · "
                    f"<b style='color:{col}'>{s['expectancy_R']:+.2f}R</b> · "
                    f"net ₹{s['net_pnl']:,.0f} · {s['open_positions']} open",
                    unsafe_allow_html=True)
    st.caption("Paper-only. Pause/retire controls live on the 🧠 Research Brain page.")


def _two_brains() -> None:
    c1, c2 = st.columns(2)
    cards = _cards()
    with c1:
        T.section("Brain 1 — what it believes", eyebrow="Evidence")
        if not cards:
            _empty("No evidence cards yet. Brain 1 issues INSUFFICIENT_EVIDENCE until real "
                   "forward results exist.")
        else:
            for c in cards[:8]:
                col = {"CONFIRMED": T.GREEN, "OVERFIT": T.RED, "DECAYING": T.RED}.get(
                    c.evidence_state, T.AMBER)
                st.markdown(f"- `{c.strategy_id}` — "
                            f"<b style='color:{col}'>{c.evidence_state}</b> · "
                            f"lb {c.lower_bound_R:+.2f}R · conf {c.confidence:.0%}",
                            unsafe_allow_html=True)
    with c2:
        T.section("Brain 2 — experiments & allocation", eyebrow="Strategy")
        brain = _brain()
        jrnl = []
        try:
            jrnl = brain.paper.performance_report().get("journal", []) if brain else []
        except Exception:
            jrnl = []
        if not jrnl:
            _empty("No allocation moves yet. Brain 2 acts only on qualified evidence cards.")
        else:
            for j in reversed(jrnl[-8:]):
                col = T.GREEN if j.get("action") == "DEPLOY" else T.RED
                st.markdown(f"- <b style='color:{col}'>{j.get('action')}</b> "
                            f"`{j.get('strategy_id','')}` ({j.get('family','')})",
                            unsafe_allow_html=True)


def _live_review() -> None:
    T.section("Live review candidates", eyebrow="Human decision required",
              sub="A review queue — NOT an auto-buy list. Only you can approve anything for live.")
    st.info("Nothing is bought here. Each candidate is a strategy whose forward evidence "
            "qualified it for **your** review. Live remains locked until you approve.")
    cands = [c for c in _cards()
             if getattr(c, "evidence_state", "") == "CONFIRMED"]
    if not cands:
        return _empty("No candidates. A strategy reaches this queue only after enough "
                      "confirmed forward evidence — with no data loaded, this stays empty.")
    for c in cands:
        with st.expander(f"{c.strategy_id} · {c.family} · {c.evidence_state}"):
            st.markdown(f"- Forward trades: **{c.forward_trades}** · "
                        f"lower-bound edge **{c.lower_bound_R:+.2f}R**")
            st.markdown(f"- Forward/backtest: **{c.forward_to_backtest:.2f}** · "
                        f"deflated Sharpe **{c.deflated_sharpe:.2f}**")
            if c.conflicting_reasons:
                st.markdown("**Reasons not to take it:**")
                for r in c.conflicting_reasons:
                    st.markdown(f"- ⚠️ {r}")
            st.caption("Approving for live is a manual, user-only action — not available here.")
