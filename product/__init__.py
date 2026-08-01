"""Retail product projections. Backend trading state remains authoritative."""
from __future__ import annotations

import sys

from product.gather import gather_product_inputs
from product.projection import ProductInputs, ProductState, SetupStep, TERMINOLOGY, build_product_state

# The dedicated terminal imports ``terminal_api`` before importing product projections. In that
# one context, install the read-only observer lifecycle and endpoint on the existing app. Ordinary
# research/worker imports do not pull in the API or start a process.
_terminal_api = sys.modules.get("terminal_api")
if _terminal_api is not None and hasattr(_terminal_api, "app"):
    from product.observer_api import install as _install_observer_api

    _install_observer_api(_terminal_api.app)

__all__ = [
    "ProductInputs", "ProductState", "SetupStep", "TERMINOLOGY",
    "build_product_state", "gather_product_inputs",
]
