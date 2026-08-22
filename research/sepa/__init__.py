"""SEPA compute helpers used by FEATURE-001/002 snapshots.

Not a live strategy package. Eligibility engines, VCP/buy-zone studies,
and experiment runners are deliberately absent from this runtime port.
"""
from __future__ import annotations

from research.sepa.config import DEFAULT_CONFIG, SepaConfig

__all__ = ["DEFAULT_CONFIG", "SepaConfig"]
