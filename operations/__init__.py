"""QuantTerm market-operations plane.

This package owns user-requested market scans, long-term research, news refresh,
F&O universe refresh and historical-data preparation. It is deliberately separate
from PAPER autonomy so execution/research jobs can never block an operator scan.
"""
from .store import OperationStore

__all__ = ["OperationStore"]
