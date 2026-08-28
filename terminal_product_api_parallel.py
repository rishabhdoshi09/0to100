"""Canonical terminal API with performance-safe operation routing.

This wrapper preserves every route from ``terminal_product_api`` and fixes the
long-term control so it enters the dedicated ``long_term`` market-operations
lane instead of accidentally starting another MARKET_SCAN.
"""
from __future__ import annotations

import terminal_api as core
import terminal_product_api as product

# The base API intentionally keeps the control mapping in one mutable registry.
# Patch only the incorrect operation kind; all endpoint/dedup/priority semantics
# remain unchanged.
core._OPERATION_CONTROLS["RUN_LONG_TERM_SCAN_NOW"] = "LONG_TERM_SCAN"

app = product.app
