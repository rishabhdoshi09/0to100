"""Second-stage due diligence — after a scanner already shortlisted a name.

This package never scans the market. It reads persisted fundamentals, news,
and the current scan/long-term rows, then returns an evidence-backed
SUPPORTS / NEUTRAL / CONTRADICTS view of the technical setup.

StockResearchEngine is the public name. Scanners and manual search share it.
"""
from product.due_diligence.engine import build_due_diligence
from product.due_diligence.research_engine import StockResearchEngine, investigate_stock
from product.due_diligence.suggest import suggest_tickers

__all__ = [
    "build_due_diligence",
    "investigate_stock",
    "StockResearchEngine",
    "suggest_tickers",
]
