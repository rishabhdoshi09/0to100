"""Plain-language retail help for QuantTerm."""
from __future__ import annotations

import streamlit as st


def render_retail_help() -> None:
    st.title("Help")
    st.caption("How to use QuantTerm without needing to understand its internal research machinery.")

    st.subheader("What QuantTerm does")
    st.write(
        "QuantTerm searches for trading ideas, tests them against historical evidence, and—only when "
        "the existing gates allow it—can operate them with simulated money. Its job is also to say "
        "no when the data or evidence is not strong enough."
    )

    st.subheader("Paper trading")
    st.write(
        "Paper trading uses real market observations with simulated capital. It can reveal operating "
        "mistakes and weak strategies, but it does not prove that live fills, taxes, liquidity and "
        "future market conditions will match the simulation."
    )

    st.subheader("What a backtest can and cannot prove")
    st.write(
        "A backtest shows how fixed rules would have behaved on the available historical data under "
        "stated cost and execution assumptions. A profitable backtest can still be chance, overfit, "
        "or dependent on data that was not actually available at the time. QuantTerm therefore keeps "
        "the scientific checks behind the plain result instead of replacing them."
    )

    st.subheader("Why no trade can be correct")
    st.write(
        "When every candidate fails the trend, liquidity, data, risk or evidence gates, the correct "
        "output is zero qualified opportunities. The product must not force a weak trade merely to "
        "look active."
    )

    st.subheader("Readiness")
    st.write(
        "Historical research readiness, live-data readiness and Zerodha connectivity are separate. "
        "For example, an expired Zerodha session can pause today's live scan while a verified immutable "
        "snapshot remains valid for reproducible research."
    )

    with st.expander("Glossary", expanded=False):
        st.markdown(
            """
- **Immutable snapshot:** a verified historical dataset that is never edited in place.
- **Qualified opportunity:** a stock that passed the currently applicable data, trend, liquidity and safety rules. It is not automatically a live BUY recommendation.
- **PAPER_AUTO:** the canonical automatic simulated-trading mode. It cannot authorize live trading.
- **Reconciled:** the saved runtime state agrees with the current simulated position book.
- **Expectancy:** the average profit or loss per trade, usually measured in units of risk.
- **Inconclusive:** the available evidence is not strong enough to call the strategy reliably good or reliably bad.
"""
        )
