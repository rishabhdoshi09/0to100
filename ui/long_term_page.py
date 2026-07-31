"""Read-only retail long-term page backed by the supervisor-owned service."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.long_term_store import load_long_term_scan
from research.autonomy.controls import (
    request_control, RUN_LONG_TERM_SCAN_NOW, REFRESH_LONG_TERM_NOW, TRACK_LONG_TERM_IDEA,
)


def _display(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Stock": r.get("symbol"),
        "Class": str(r.get("classification", "")).replace("_", " ").title(),
        "Combined": r.get("combined_score"),
        "Technical": r.get("technical_score"),
        "Fundamental": r.get("fundamental_score"),
        "Coverage": f"{float(r.get('fundamental_coverage', 0) or 0)*100:.0f}%",
        "Sector": r.get("sector"),
        "Price": r.get("price"),
        "12m %": r.get("mom_12m_pct"),
        "Below high %": r.get("from_high_pct"),
        "Timing": str(r.get("timing", "")).replace("_", " ").title(),
        "Quality": " · ".join(r.get("quality_factors", [])[:3]),
        "Risks": " · ".join(r.get("risk_flags", [])[:3]),
    } for r in rows])


def _current_price(symbol: str) -> float:
    try:
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(symbol)
        return float(df["close"].iloc[-1]) if df is not None and len(df) else 0.0
    except Exception:
        return 0.0


def render_long_term() -> None:
    st.title("Long-Term Picks")
    st.caption("A separate current-investing lens: official price history plus current fundamental "
               "quality, valuation and governance coverage. It never feeds current fundamentals into historical backtests.")

    a, b = st.columns(2)
    if a.button("Run long-term scan", type="primary", width="stretch"):
        request_control(RUN_LONG_TERM_SCAN_NOW, reason="owner requested current long-term scan")
        st.success("Long-term scan queued. Refresh this page after the supervisor completes it.")
    if b.button("Refresh shortlist fundamentals", width="stretch"):
        request_control(REFRESH_LONG_TERM_NOW,
                        reason="owner requested current fundamental refresh for long-term shortlist")
        st.success("Fundamental refresh and long-term rescan queued.")

    payload = load_long_term_scan()
    if not payload:
        st.info("No saved long-term scan yet. Run the scan first; missing fundamentals will be shown honestly.")
        return

    summary = dict(payload.get("summary", {}) or {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality compounders", summary.get("quality_compounder", 0))
    c2.metric("GARP candidates", summary.get("garp_candidate", 0))
    c3.metric("Long-term watch", summary.get("long_term_watch", 0))
    c4.metric("Fundamental coverage", f"{float(summary.get('coverage_pct', 0) or 0):.0f}%")

    st.caption(f"Last scan: {str(payload.get('scanned_at', ''))[:19]} · "
               f"Source: {payload.get('fundamentals_source', 'current fundamentals')} · "
               "Fundamentals are current snapshots, not point-in-time history.")

    records = list(payload.get("records", []) or [])
    groups = {
        "Quality Compounders": [r for r in records if r.get("classification") == "QUALITY_COMPOUNDER"],
        "GARP": [r for r in records if r.get("classification") == "GARP_CANDIDATE"],
        "Quality but Expensive": [r for r in records if r.get("classification") == "QUALITY_BUT_EXPENSIVE"],
        "Watch": [r for r in records if r.get("classification") == "LONG_TERM_WATCH"],
        "Needs Fundamentals": [r for r in records if r.get("classification") == "NEEDS_FUNDAMENTALS"],
        "Avoid / Review": [r for r in records if r.get("classification") == "AVOID_REVIEW"],
    }
    tabs = st.tabs(list(groups))
    for tab, (title, rows) in zip(tabs, groups.items()):
        with tab:
            if rows:
                st.dataframe(_display(rows), hide_index=True, width="stretch")
            else:
                st.info(f"No stocks in {title}.")

    try:
        from core.long_term_tracker import active_picks, exited_picks
        active = active_picks()
    except Exception:
        active, exited_picks = [], lambda *_: []

    st.divider()
    st.subheader("Tracked long-term ideas")
    if active:
        table = []
        for item in active:
            entry = float(item.get("entry_price") or 0)
            current = _current_price(str(item.get("symbol", "")))
            ret = ((current / entry - 1) * 100) if entry > 0 and current > 0 else None
            table.append({"Stock": item.get("symbol"), "Added": entry, "Current": current,
                          "Return %": round(ret, 1) if ret is not None else None,
                          "Score": item.get("score"), "Thesis": item.get("thesis")})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    else:
        st.caption("No tracked long-term ideas yet.")

    eligible = [r for r in records if r.get("classification") in
                ("QUALITY_COMPOUNDER", "GARP_CANDIDATE")]
    tracked = {str(p.get("symbol", "")).upper() for p in active}
    if eligible:
        with st.expander("Track a shortlisted idea"):
            options = [r["symbol"] for r in eligible if r["symbol"] not in tracked]
            if options:
                selected = st.selectbox("Stock", options)
                if st.button("Track this idea", width="stretch"):
                    request_control(TRACK_LONG_TERM_IDEA, value={"symbol": selected},
                                    reason="owner tracked an eligible long-term research idea")
                    st.success(f"{selected} tracking request queued for the autonomy supervisor.")
            else:
                st.caption("All eligible shortlist names are already tracked.")

    revised = exited_picks(limit=20) if callable(exited_picks) else []
    if revised:
        with st.expander("Revised / exited ideas"):
            st.dataframe(pd.DataFrame(revised), hide_index=True, width="stretch")

    st.warning(payload.get("disclaimer", "Current shortlist only; perform independent due diligence."))
