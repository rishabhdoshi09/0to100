"""Retail product projections. Backend trading state remains authoritative."""
from product.gather import gather_product_inputs
from product.plain_language import (
    FIELD_LABELS,
    NAV_JOBS,
    PlainCard,
    bulk_explain,
    explain_decision,
    explain_metric,
    explain_pit_state,
    explain_research_verdict,
    explain_trust_class,
    label_for,
    render_layers,
    research_report_blurb,
)
from product.projection import ProductInputs, ProductState, SetupStep, TERMINOLOGY, build_product_state

__all__ = [
    "ProductInputs", "ProductState", "SetupStep", "TERMINOLOGY",
    "build_product_state", "gather_product_inputs",
    "FIELD_LABELS", "NAV_JOBS", "PlainCard",
    "bulk_explain", "explain_decision", "explain_metric", "explain_pit_state",
    "explain_research_verdict", "explain_trust_class", "label_for",
    "render_layers", "research_report_blurb",
]
