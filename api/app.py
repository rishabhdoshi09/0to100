"""Stable public entrypoint for the QuantTerm product API.

This module intentionally does not construct another FastAPI application.  It
re-exports the existing, safety-hardened application object so callers have one
stable name while the historical implementation can be simplified behind the
boundary in later tranches.

Production invariant: importing this module must never alter broker authority,
risk gates, evidence classes, or live-execution state.
"""
from __future__ import annotations

from terminal_product_api_parallel import app as app

__all__ = ["app"]
