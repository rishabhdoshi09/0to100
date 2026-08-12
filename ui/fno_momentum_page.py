"""Retail F&O Momentum page.

Every current individual-stock F&O underlying is considered. Filters only
change what is displayed; they never shrink the evaluated universe silently.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date

import pandas as pd
import streamlit as st

from data.fno_universe import current_fno_universe, evaluate_all_underlyings


def _get_data_client():
    """Use the existing authenticated Zerodha session through its data-only view."""
    try:
        from research.intelligence.data.kite_activation import KiteDataClient

        return KiteDataClient.from_config()
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def _load_universe(_as_of: date):
    return current_fno_universe(_get_data_client(), as_of=_as_of)


@st.cache_data(ttl=900, show_spinner=False)
def _run_funnel(_as_of: date):
    universe = _load_universe(_as_of)

    # One bulk prefetch for the complete F&O stock universe, then the existing
    # compute-only scanner evaluates every mapped underlying.
    from scan.bulk_fetcher import get_cached, prefetch
    from scan.unified_scanner import UnifiedScanner

    prefetch(universe.symbols)
    scanner = UnifiedScanner()
    funnel = evaluate_all_underlyings(
        universe,
        history_getter=get_cached,
        analyzer=scanner._analyze,
        minimum_sessions=60,
    )
    return universe, funnel


def _status_label(row) -> str:
    if row.qualified:
        return "Momentum qualified"
    return {
        "history": "Data not ready",
        "analysis": "Could not evaluate",
        "safety_checks": "Failed safety checks",
        "momentum": "Momentum not strong enough",
    }.get(row.stage, "Not qualified")


def _rows_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Stock": row.symbol,
                "Company": row.company_name,
                "Status": _status_label(row),
                "Price": row.price or None,
                "Momentum 5d %": row.momentum_5d or None,
                "Score": row.score or None,
                "RSI": row.rsi or None,
                "Volume vs normal": f"{row.volume_ratio:.2f}x" if row.volume_ratio else "—",
                "Current future": row.future_symbol,
                "Expiry": row.expiry,
                "Lot size": row.lot_size,
                "Why": row.reason,
            }
            for row in rows
        ]
    )


def render_fno_momentum_page() -> None:
    st.title("F&O Momentum")
    st.caption(
        "QuantTerm checks every currently listed individual-stock F&O underlying. "
        "A stock disappears only when a visible data, momentum, liquidity or safety rule rejects it."
    )

    top1, top2 = st.columns([4, 1])
    with top2:
        if st.button("Refresh", type="primary", width="stretch"):
            _load_universe.clear()
            _run_funnel.clear()
            st.rerun()

    with st.spinner("Reading current F&O universe and evaluating momentum…"):
        universe, funnel = _run_funnel(date.today())

    if universe.source == "unavailable":
        st.error(
            "F&O instrument data is not available. Complete the normal Zerodha login or refresh "
            "the instrument master. QuantTerm will not show a made-up shortlist."
        )
        return

    if not universe.underlyings:
        st.warning(
            "No individual-stock F&O underlyings could be mapped. Open Data and Zerodha, refresh "
            "the instrument master, then retry."
        )
        if universe.exclusions:
            st.dataframe(
                pd.DataFrame([asdict(row) for row in universe.exclusions]),
                hide_index=True,
                width="stretch",
            )
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current F&O stocks", universe.unique_stock_underlyings)
    c2.metric("Mapped correctly", universe.mapped_underlyings)
    c3.metric("Data ready", funnel.data_ready)
    c4.metric("Evaluated", funnel.evaluated)
    c5.metric("Momentum qualified", funnel.momentum_qualified)

    st.caption(
        f"Source: {universe.source} · Stock-future contracts read: "
        f"{universe.total_future_contracts - universe.index_future_contracts:,} · "
        f"Index futures excluded: {universe.index_future_contracts:,}"
    )

    if universe.exclusions:
        with st.expander(f"Mapping problems ({len(universe.exclusions)})"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Underlying": row.underlying,
                            "Stage": row.stage,
                            "Reason": row.reason,
                        }
                        for row in universe.exclusions
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    tabs = st.tabs(["Qualified", "All considered", "Excluded with reasons"])

    with tabs[0]:
        # deterministic: score descending, symbol as the stable secondary key for ties
        qualified = sorted(funnel.qualified, key=lambda row: (-row.score, row.symbol))
        if not qualified:
            st.info(
                "No F&O stock currently passes the momentum rules. This is a valid result; "
                "QuantTerm does not force weak trades."
            )
        else:
            st.dataframe(_rows_frame(qualified), hide_index=True, width="stretch")

    with tabs[1]:
        status_filter = st.multiselect(
            "Show status",
            options=sorted({_status_label(row) for row in funnel.rows}),
            default=sorted({_status_label(row) for row in funnel.rows}),
        )
        shown = [row for row in funnel.rows if _status_label(row) in status_filter]
        st.caption(
            f"Displayed: {len(shown)} · Evaluated universe remains: {funnel.total_underlyings}. "
            "Changing this filter never changes which stocks were considered."
        )
        st.dataframe(_rows_frame(shown), hide_index=True, width="stretch")

    with tabs[2]:
        excluded = funnel.excluded
        if not excluded:
            st.success("Every mapped F&O stock qualified.")
        else:
            reason_counts = (
                pd.Series([_status_label(row) for row in excluded])
                .value_counts()
                .rename_axis("Reason group")
                .reset_index(name="Stocks")
            )
            st.dataframe(reason_counts, hide_index=True, width="stretch")
            st.dataframe(_rows_frame(excluded), hide_index=True, width="stretch")

    st.info(
        "Momentum is calculated from the underlying cash stock. F&O availability is an overlay, "
        "not a reason to ignore the rest of the current F&O universe. Futures orders are not placed "
        "from this page."
    )
