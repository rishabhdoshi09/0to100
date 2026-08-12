"""Research-only market structure discovery (Phase A / A5)."""
from research.market_structure.engine import (
    MarketStructureResult,
    discover_structure,
    returns_matrix_from_closes,
)
from research.market_structure.benchmarks import compare_to_labels

__all__ = [
    "MarketStructureResult",
    "discover_structure",
    "returns_matrix_from_closes",
    "compare_to_labels",
]
