"""
🛰️ Research OS — internal mission control (Layer 1).

Dense over pretty (deliberately): this page answers "is my research organisation
healthy?" at a glance — beliefs, edge health, the gate scorecard, data quality,
research debt, a Time Machine, and an Evidence Explorer. It RENDERS the
`research_overview` aggregation layer; it computes nothing itself (the same layer
JARVIS queries). Fully fail-open — a down feed shows an empty section, never a
stack trace.
"""
from __future__ import annotations


def _kpi_row(st, items):
    """A compact row of metric tiles: items = [(label, value, help?)]."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        col.metric(label, value)


def render_research_dashboard() -> None:
    import streamlit as st

    st.subheader("🛰️ Research OS — mission control")
    st.caption("Is the *research* healthy? (not the market) · renders the "
               "Research Overview layer · dense by design")

    try:
        from research import research_overview as RO
    except Exception as exc:                                # pragma: no cover
        st.error(f"Research Overview unavailable: {exc}")
        return

    o = RO.overview()
    rh = o.get("research_health", {})
    kg = o.get("knowledge_growth", {})
    eh = o.get("edge_health", {})
    dh = o.get("data_health", {})
    debt = o.get("research_debt", {})

    # ── 🛡️ Governance Sentinel — the safety authority + human kill-switch ──────
    try:
        from core import governance as _G
        g = _G.assess(force=True)
        _c = {"NORMAL": "#00d4a0", "DE_RISK": "#f59e0b", "HALT": "#ff4b4b"}[g["state"]]
        gc1, gc2 = st.columns([3, 1])
        with gc1:
            st.markdown(
                f"#### 🛡️ Governance: <span style='color:{_c}'>{g['state']}</span>",
                unsafe_allow_html=True)
            if g["reasons"]:
                for r in g["reasons"]:
                    st.caption(f"• {r}")
            else:
                st.caption("All clear — no kill conditions or rollback triggers active.")
        with gc2:
            _halted = bool(_G._read_state().get("manual_halt"))
            if _halted:
                if st.button("▶️ Resume trading", key="gov_resume", width="stretch"):
                    _G.set_manual_halt(False); _G._cache.update(ts=0.0, data=None)
                    st.rerun()
            else:
                if st.button("🛑 KILL SWITCH", key="gov_kill", type="primary",
                             width="stretch", help="Halt all new LIVE orders now."):
                    _G.set_manual_halt(True); _G._cache.update(ts=0.0, data=None)
                    st.rerun()
        with st.expander("🎚️ Evidence levels — what the market has actually PROVEN"):
            from core.evidence_levels import report as _elr, headline as _elh
            st.caption(_elh())
            st.dataframe([{"Capability": r["capability"], "Level": r["label"],
                           "Basis": r["basis"]} for r in _elr()],
                        width="stretch", hide_index=True)
        st.divider()
    except Exception as _exc:
        st.caption(f"Governance panel unavailable: {_exc}")

    # ── 📈 Knowledge Growth (the metric that matters most: is it LEARNING?) ────
    st.markdown("#### 📈 Knowledge Growth — *is the Research OS learning?*")
    _kpi_row(st, [
        ("Net knowledge / mo", f"{kg.get('net_per_month', 0):+.1f}"),
        ("Validated / mo", kg.get("validated_per_month", 0)),
        ("Retired / mo", kg.get("retired_per_month", 0)),
        ("Avg evidence / belief", kg.get("avg_evidence_per_belief", 0)),
    ])
    st.caption("📈 Learning — net validated knowledge is growing."
               if kg.get("learning") else
               "⚠️ Not yet compounding — validated ≤ retired this window (early, or "
               "the flywheel needs more observation history).")

    # ── 🧪 Research Health ────────────────────────────────────────────────────
    st.markdown("#### 🧪 Research Health")
    _kpi_row(st, [
        ("Active beliefs", rh.get("beliefs_active", 0)),
        ("On watch", rh.get("beliefs_watch", 0)),
        ("Retired", rh.get("beliefs_retired", 0)),
        ("Rejected (neg. knowledge)", rh.get("beliefs_rejected", 0)),
    ])
    _kpi_row(st, [
        ("Promoted this week", rh.get("promoted_this_week", 0)),
        ("→ Watch this week", rh.get("to_watch_this_week", 0)),
        ("Retired this week", rh.get("retired_this_week", 0)),
        ("Overdue for review", rh.get("beliefs_overdue_review", 0)),
    ])
    calib = rh.get("calibration", {})
    if calib.get("n"):
        st.caption(f"🎯 Calibration: {calib.get('insight', '')} "
                   f"(ECE {calib.get('ece')}, n={calib.get('n')})")
    rej = rh.get("recently_rejected_hypotheses") or []
    if rej:
        st.caption("🗑️ Recently rejected hypotheses: " + ", ".join(rej))

    # ── 📉 Edge Health ────────────────────────────────────────────────────────
    st.markdown("#### 📉 Edge Health")
    _kpi_row(st, [
        ("Durable", eh.get("durable", 0)),
        ("Cyclical", eh.get("cyclical", 0)),
        ("Recovering", eh.get("recovering", 0)),
        ("Decaying", eh.get("decaying", 0)),
        ("Dead", eh.get("dead", 0)),
    ])
    mrt = eh.get("median_recovery_trades")
    st.caption(f"Tracked signals: {eh.get('tracked_signals', 0)} · "
               f"median recovery: {mrt if mrt is not None else '—'} trades")

    # ── ⚖️ Gate Scorecard ─────────────────────────────────────────────────────
    st.markdown("#### ⚖️ Gate Scorecard")
    card = o.get("gate_scorecard", [])
    if card:
        st.dataframe(
            [{"Gate": r["gate"], "Verdict": r["verdict"],
              "Saved": r["saved"], "Cost": r["cost"],
              "Net fwd %": r["net_fwd_pct"],
              "Modeled R*": r.get("modeled_avg_r"),
              "Confidence": r["confidence"], "Trend": r["trend"], "n": r["n"]}
             for r in card],
            width="stretch", hide_index=True)
        st.caption("*Modeled R is counterfactual (hypothetical ATR-stop) — the "
                   "observed Net fwd % is the canonical metric.")
    else:
        st.caption("No settled rejection evidence yet — control group still filling.")

    # ── 📊 Data Health ────────────────────────────────────────────────────────
    st.markdown("#### 📊 Data Health")
    bk = dh.get("by_kind", {})
    _kpi_row(st, [
        ("Total observations", dh.get("total_observations", 0)),
        ("Trades", (bk.get("TRADE", {}) or {}).get("total", 0)),
        ("Rejections", (bk.get("REJECTION", {}) or {}).get("total", 0)),
        ("Near-misses", (bk.get("NEAR_MISS", {}) or {}).get("total", 0)),
    ])
    _on = dh.get("on_current_schema", True)
    st.caption(f"Schema: {dh.get('current_schema')} · versions in store: "
               f"{', '.join(dh.get('schema_versions', []) or ['—'])} "
               f"{'✅' if _on else '⚠️ mixed — some rows on an older schema'}")
    if dh.get("impossible_values") or dh.get("stale_values"):
        st.caption(f"⚠️ Data problems: {dh.get('impossible_values', 0)} impossible, "
                   f"{dh.get('stale_values', 0)} stale values on record.")
    thin = dh.get("thin_features") or []
    if thin:
        st.dataframe([{"Feature": t["feature"], "Fill rate": t["fill_rate"]}
                      for t in thin], width="stretch", hide_index=True)

    # ── 🧾 Research Debt ──────────────────────────────────────────────────────
    st.markdown("#### 🧾 Research Debt")
    _kpi_row(st, [
        ("Experiments awaiting validation", debt.get("experiments_awaiting_validation", 0)),
        ("Beliefs overdue for review", debt.get("beliefs_overdue_review", 0)),
        ("Drift alerts unresolved", debt.get("drift_alerts_unresolved", 0)),
        ("Schemas awaiting migration", debt.get("schemas_awaiting_migration", 0)),
    ])

    st.divider()

    # ── 🕰️ Time Machine ───────────────────────────────────────────────────────
    with st.expander("🕰️ Time Machine — what did the system believe on a past date?"):
        import datetime
        pick = st.date_input("As-of date", value=datetime.date.today(),
                             key="research_os_timemachine")
        tm = RO.time_machine(datetime.datetime.combine(
            pick, datetime.time(23, 59)).isoformat())
        st.caption(f"On {pick}: {tm.get('active', 0)} active · {tm.get('watch', 0)} "
                   f"on watch · {tm.get('total', 0)} beliefs total.")
        if tm.get("beliefs"):
            st.dataframe([{"Belief": b["statement"], "Status": b["status"],
                           "Evidence": b.get("evidence_n"), "EV(R)": b.get("ev_r")}
                          for b in tm["beliefs"]],
                         width="stretch", hide_index=True)

    # ── 🔬 Evidence Explorer ──────────────────────────────────────────────────
    with st.expander("🔬 Evidence Explorer — trace any object's provenance"):
        nid = st.text_input("Node id (e.g. GATE:extension_guard, BELIEF:<id>)",
                            key="research_os_evidence")
        if nid:
            try:
                from research.evidence_graph import explain
                st.code(explain(nid.strip()))
            except Exception as exc:
                st.caption(f"Nothing to show: {exc}")
