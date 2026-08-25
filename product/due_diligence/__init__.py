"""Second-stage due diligence — after a scanner already shortlisted a name.

This package never scans the market. It reads persisted fundamentals, news,
and the current scan/long-term rows, then returns an evidence-backed
SUPPORTS / NEUTRAL / CONTRADICTS view of the technical setup.
"""
from product.due_diligence.engine import build_due_diligence

__all__ = ["build_due_diligence"]
