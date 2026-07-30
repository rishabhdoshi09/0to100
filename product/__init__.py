"""Retail product projections for QuantTerm.

This package is deliberately read-only. It translates canonical backend state
into plain-language product views; it does not persist trading state, calculate
signals, or import broker order-placement code.
"""

from product.projection import (
    ProductInputs,
    ProductStatus,
    ReadinessCard,
    HomeProjection,
    PaperTradingProjection,
    build_home_projection,
    build_paper_trading_projection,
)

__all__ = [
    "ProductInputs",
    "ProductStatus",
    "ReadinessCard",
    "HomeProjection",
    "PaperTradingProjection",
    "build_home_projection",
    "build_paper_trading_projection",
]
