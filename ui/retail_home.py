"""Default retail Home view for QuantTerm."""
from __future__ import annotations

import streamlit as st

from product.gather import gather_product_inputs
from product.projection import ProductStatus, ReadinessCard, build_home_projection


def _show_card(card: ReadinessCard) -> None:
    label = f"**{card.title} — {card.status.value}**\n\n{card.summary}"
    if card.detail:
        label += f"\n\n{card.detail}"
    if card.status == ProductStatus.READY:
        st.success(label)
    elif card.status in {ProductStatus.MISSING, ProductStatus.STALE}:
        st.error(label)
    elif card.status in {ProductStatus.ATTENTION, ProductStatus.UNKNOWN}:
        st.warning(label)
    else:
        st.info(label)


def render_retail_home() -> None:
    st.title("QuantTerm Home")
    st.caption(
        "A plain-language view of the real research, market-data and automatic paper-trading state. "
        "This page does not calculate signals or create a second source of truth."
    )

    inputs = gather_product_inputs()
    view = build_home_projection(inputs)

    _show_card(view.overall)
    st.subheader("What should I do next?")
    st.info(view.primary_action)

    a, b, c = st.columns(3)
    with a:
        _show_card(view.research)
    with b:
        _show_card(view.paper)
    with c:
        _show_card(view.opportunities)

    d, e, f = st.columns(3)
    with d:
        _show_card(view.live)
    with e:
        _show_card(view.broker)
    with f:
        _show_card(view.market)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "Paper equity",
        f"₹{inputs.paper_equity:,.0f}" if inputs.paper_equity is not None else "Unknown",
    )
    p2.metric("Open paper positions", len(inputs.paper_open_positions))
    p3.metric(
        "Open paper risk",
        f"₹{inputs.paper_open_risk:,.0f}" if inputs.paper_open_risk is not None else "Unknown",
    )
    p4.metric(
        "Qualified opportunities",
        inputs.qualified_opportunities if inputs.qualified_opportunities is not None else "Not verified",
    )

    if view.attention_items:
        with st.expander(f"Needs attention ({len(view.attention_items)})", expanded=True):
            for item in view.attention_items:
                st.write(f"• {item}")

    with st.expander("Technical evidence", expanded=False):
        st.json(
            {
                "observed_at": inputs.observed_at.isoformat(),
                "snapshot_id": inputs.snapshot_id,
                "snapshot_last_trading_date": (
                    inputs.snapshot_last_trading_date.isoformat()
                    if inputs.snapshot_last_trading_date else None
                ),
                "instrument_master_source": inputs.instrument_master_source,
                "instrument_master_count": inputs.instrument_master_count,
                "paper_mode": inputs.paper_mode,
                "paper_auto_enabled": inputs.paper_auto_enabled,
                "runtime_reconciled": inputs.runtime_reconciled,
                "last_completed_cycle": inputs.last_completed_cycle,
            }
        )

    st.caption(
        "Ready means the required evidence and paper controls are available. It never means "
        "QuantTerm must force a trade."
    )
