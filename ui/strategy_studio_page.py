"""
🧪 Strategy Studio — the user-facing page (Discover · Review · Tweak · Compare · Approve).

Thin renderer over the pure `research.strategy_studio` modules (all logic + safety live
there, tested). Plain language by default; technical detail behind expanders. It contains
NO order actions of any kind and cannot connect a strategy to execution. Only a USER can
approve a strategy for PAPER; LIVE is out of scope and stays locked.
"""
from __future__ import annotations

import streamlit as st

from research.strategy_studio import spec as S
from research.strategy_studio import discovery as D
from research.strategy_studio import review as R
from research.strategy_studio import tweak as T
from research.strategy_studio import approval as A
from research.strategy_studio import wizard as W


def _data_status() -> dict | None:
    """Read the current research-data readiness (read-only) from the data-setup snapshot,
    if present. Never fabricates readiness."""
    snap = st.session_state.get("_hds_snapshot")
    if not snap:
        return None
    status = snap.get("status", "")
    color = "green" if status == "READY" else "amber" if "LIMIT" in status else "red"
    return {"color": color, "can_run": color in ("green", "amber"), "reasons": []}


def render_strategy_studio() -> None:
    st.markdown("## 🧪 Strategy Studio")
    st.caption("QuantTerm can suggest its own trading ideas, test them honestly, and show "
               "you the case for and against each one. **Nothing here can place an order.** "
               "Only you can approve an idea for paper practice; real-money trading stays "
               "locked.")
    with st.expander("❔ What is this? (start here)", expanded=False):
        st.markdown("- **Discover** — the system studies data and proposes ideas.\n"
                    "- **Review** — read *why it might work* and *why it might fail*.\n"
                    "- **Tweak** — change one rule at a time (each change is re-tested).\n"
                    "- **Compare** — put versions side by side.\n"
                    "- **Approve** — you decide if an idea is good enough for paper practice.")

    tabs = st.tabs(["1 · Discover", "2 · Review", "3 · Tweak", "4 · Compare", "5 · Approve"])
    data = _data_status()

    # ── 1. Discover ──
    with tabs[0]:
        st.markdown("### Discover ideas")
        if not D.data_ready(data):
            st.warning(D.DISCOVERY_UNAVAILABLE_MSG)
            st.caption("Load real historical data in **Historical Data Setup** first. "
                       "You can still explore the workflow below with a demonstration set, "
                       "clearly labelled — a demo is **not** market evidence.")
        seed = st.number_input("Idea batch (changes which ideas appear)", 1, 9999, 1)
        if st.button("💡 Suggest strategies"):
            cands = D.generate(D.DiscoveryBudget(seed=int(seed), max_search_attempts=40))
            st.session_state["_ss_cands"] = [c.as_dict() for c in cands]
            st.session_state["_ss_specs"] = cands
            st.success(f"Proposed {len(cands)} ideas across "
                       f"{len({c.family for c in cands})} families. None are trades — they "
                       "are ideas to review.")
        specs = st.session_state.get("_ss_specs", [])
        for c in specs[:10]:
            st.markdown(f"- **{c.name}** — {c.hypothesis}")

    # ── 2. Review (Convince Me) ──
    with tabs[1]:
        st.markdown("### Convince Me")
        specs = st.session_state.get("_ss_specs", [])
        if not specs:
            st.info("Discover some ideas first.")
        else:
            names = [c.name for c in specs]
            pick = st.selectbox("Which idea?", names)
            spec = next(c for c in specs if c.name == pick)
            # DEMO evidence — explicitly synthetic (never market evidence)
            ev = D.EvidenceReport(n_trades=40, n_symbols=20, net_expectancy_R=0.25,
                                  gross_expectancy_R=0.35, cost_drag_R=0.1, verdict="INCONCLUSIVE",
                                  regime_consistency=0.6, sector_consistency=0.55,
                                  is_synthetic=True)
            cm = R.convince_me(spec, ev, data_status=data, n_attempts=40, n_rejected=12,
                               trades=[{"net_R": 1.1}, {"net_R": -0.9, "exit_reason": "gap_stop"}],
                               dataset_period="demonstration set", limitations=["demonstration only"])
            st.info(f"System says: **{cm['system_recommendation']}**")
            if cm["synthetic_labelled_non_evidence"]:
                st.warning("These numbers are from a DEMONSTRATION set — **not market "
                           "evidence.** Load real data to get a real verdict.")
            st.markdown(f"**Idea:** {cm['one_line_idea']}")
            st.markdown("**How it works:**")
            for step in cm["how_it_works"]:
                st.markdown(f"1. {step}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Why it might work")
                for k, items in cm["why_it_may_work"].items():
                    for it in items:
                        st.markdown(f"- *({k.replace('_',' ')})* {it}")
            with c2:
                st.markdown("#### Why it might fail")
                w = cm["what_could_go_wrong"]
                st.markdown(f"- Worst period: {w['worst_period']}")
                st.markdown(f"- {w['gap_risk']}")
                st.markdown(f"- {w['liquidity_risk']}")
                for inv in w["invalidation_conditions"]:
                    st.markdown(f"- Could break if: {inv}")
            st.markdown("#### How reliable is the evidence?")
            st.json(cm["confidences"])
            with st.expander("Technical details"):
                st.json(cm)

    # ── 3. Tweak ──
    with tabs[2]:
        st.markdown("### Change the rules")
        specs = st.session_state.get("_ss_specs", [])
        if not specs:
            st.info("Discover some ideas first.")
        else:
            spec = specs[0]
            st.caption("Ask in plain words (e.g. \"reduce the maximum stop to 5%\"). Every "
                       "real change makes a NEW version and needs a fresh test.")
            req = st.text_input("What would you like to change?")
            if req:
                diff = T.parse_nl(req, spec)
                if diff["status"] == "ready":
                    st.json(T.tweak_impact_preview(spec, diff))
                    if st.button("Create new version & (re)test"):
                        new = T.apply_diff(spec, diff)
                        st.success(f"Created {new.strategy_id} v{new.version}. Its evidence "
                                   "starts fresh — the old evidence does not carry over.")
                else:
                    st.warning(diff.get("why", "Please rephrase."))

    # ── 4. Compare ──
    with tabs[3]:
        st.markdown("### Compare versions")
        specs = st.session_state.get("_ss_specs", [])
        if len(specs) >= 2:
            rows = [{"name": c.name, "spec": c,
                     "ev": D.EvidenceReport(net_expectancy_R=0.2, is_synthetic=True)}
                    for c in specs[:3]]
            cmp = R.compare(rows)
            st.table(cmp["rows"])
            st.caption(cmp["note"] + " The system never auto-picks the highest number.")
        else:
            st.info("Discover at least two ideas to compare.")

    # ── 5. Approve (user-only; PAPER only; no live) ──
    with tabs[4]:
        st.markdown("### Approve for paper practice")
        st.info("This only ever approves **paper** (pretend-money) practice. There is no "
                "live-trading button here, and approval alone does not start anything — "
                "you confirm paper activation separately.")
        st.caption("A strategy can only be approved when its evidence is real (not a demo) "
                   "and the data-readiness gate is green. Until real data is loaded, this "
                   "stays disabled — on purpose.")
        st.button("Approve for Paper Testing", disabled=True,
                  help="Disabled: no real research-grade evidence yet (demo data is not "
                       "market evidence).")
