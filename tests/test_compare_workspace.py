"""Compare workspace projections."""
from __future__ import annotations

from product.compare_workspace import build_compare_workspace


def test_compare_limits_symbol_count():
    payload = build_compare_workspace(["A", "B", "C", "D", "E", "F"], max_symbols=5)
    assert len(payload["symbols"]) == 5


def test_compare_empty_symbols_returns_structure():
    payload = build_compare_workspace([])
    assert payload["rows"] == []
    assert "disclaimer" in payload
