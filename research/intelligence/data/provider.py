"""
🔌 SnapshotBarProvider — the production provider bound to ONE pinned snapshot.

Invariants: reads only the pinned, committed snapshot; never returns bars after `through`;
never falls back to the internet or synthetic data; deterministic; Streamlit-free. If the
snapshot lacks something (e.g. a benchmark), it says so honestly rather than returning zeros.
"""
from __future__ import annotations


class SnapshotBarProvider:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.snapshot_id = snapshot.snapshot_id

    def bars(self, symbol: str, through: str, adjustment: str = "raw") -> list:
        return self.snapshot.bars(symbol, through, adjustment)

    def universe(self, on_date: str) -> list:
        return self.snapshot.universe(on_date)

    def benchmark(self, through: str, name: str | None = None) -> list:
        return self.snapshot.benchmark(through, name)

    def latest_available_date(self):
        return self.snapshot.latest_available_date()

    def health(self) -> dict:
        return self.snapshot.health()

    def coverage_for(self, spec) -> dict:
        return self.snapshot.coverage_for(spec)

    def universe_history(self, symbol: str, through: str) -> list:
        """Full PIT bar history for a symbol through `through` (for cross-sectional ranking)."""
        return self.snapshot.bars(symbol, through)
