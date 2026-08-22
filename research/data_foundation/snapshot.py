"""Read-only EvidenceSnapshot. Historical research uses frozen local evidence only."""
from __future__ import annotations

from typing import Any

from research.data_foundation.gates import evaluate as evaluate_gates
from research.data_foundation.manifest import build_manifest, snapshot_hash
from research.data_foundation.network import forbid_network


class EvidenceSnapshot:
    """Point-in-time research facade. No silent network I/O."""

    def __init__(
        self,
        as_of: str,
        *,
        config: dict | None = None,
        fundamentals_path=None,
        events_path=None,
        sector_snapshot: dict | None = None,
        price_frames: dict | None = None,
        universe_path=None,
        ca_path=None,
        index_dir=None,
        guard_network: bool = True,
    ):
        self.as_of = str(as_of)[:10]
        self.fundamentals_path = fundamentals_path
        self.events_path = events_path
        self.sector_snapshot = sector_snapshot
        self.price_frames = price_frames
        self.universe_path = universe_path
        self.ca_path = ca_path
        self.index_dir = index_dir
        self.guard_network = guard_network
        self.manifest = build_manifest(as_of=self.as_of, config=config)
        self.snapshot_hash = self.manifest["snapshot_hash"]

    def _guard(self):
        return forbid_network() if self.guard_network else _nullcontext()

    def prices(self, symbol: str):
        """OHLCV known on or before as_of. Never fabricates a later bar."""
        import pandas as pd
        with self._guard():
            df = None
            if self.price_frames is not None:
                df = self.price_frames.get(str(symbol).upper())
            else:
                from data.bhavcopy_store import get_ohlcv
                df = get_ohlcv(symbol)
            if df is None or len(df) == 0:
                return None
            idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            cut = pd.Timestamp(self.as_of)
            mask = idx <= cut
            out = df.loc[mask].copy()
            return out if len(out) else None

    def universe(self) -> dict:
        with self._guard():
            from data.nse_universe import point_in_time_universe
            return point_in_time_universe(self.as_of, path=self.universe_path)

    def fundamentals(self, symbol: str) -> dict | None:
        with self._guard():
            from data.pit_fundamentals import fundamentals_with_ratios
            return fundamentals_with_ratios(
                symbol, self.as_of, path=self.fundamentals_path,
            )

    def earnings_events(self, symbol: str) -> list[dict]:
        with self._guard():
            from data.earnings_events import timeline
            return timeline(symbol, self.as_of, path=self.events_path)

    def sector(self, symbol: str) -> dict:
        with self._guard():
            from data.sector_map import sector_of
            return sector_of(symbol, self.sector_snapshot)

    def benchmark(self, name: str = "NIFTY500") -> dict:
        with self._guard():
            from data.benchmarks import load_index
            return load_index(name, as_of=self.as_of, index_dir=self.index_dir)

    def data_quality(self, symbol: str, *, required: tuple[str, ...] | None = None) -> dict:
        frame = self.prices(symbol)
        req = required or ("PRICE_OK", "UNIVERSE_OK")
        return evaluate_gates(symbol, self.as_of, required=req, frame=frame)

    def replay_hash(self) -> str:
        return snapshot_hash(self.manifest)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
