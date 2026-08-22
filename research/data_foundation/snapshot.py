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
        bind = price_frames is None and fundamentals_path is None and index_dir is None
        self.manifest = build_manifest(as_of=self.as_of, config=config, bind_live_stores=bind)
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
            bundle = fundamentals_with_ratios(
                symbol, self.as_of, path=self.fundamentals_path,
            )
            if not bundle:
                return None
            if bundle.get("current"):
                bundle["current"] = self._with_provenance("fundamentals", bundle["current"])
            return bundle

    def earnings_events(self, symbol: str) -> list[dict]:
        with self._guard():
            from data.earnings_events import timeline
            return [self._with_provenance("earnings_event", e) for e in timeline(
                symbol, self.as_of, path=self.events_path,
            )]

    def sector(self, symbol: str) -> dict:
        with self._guard():
            from data.sector_map import sector_of
            return self._with_provenance("sector", sector_of(symbol, self.sector_snapshot))

    def benchmark(self, name: str = "NIFTY500") -> dict:
        with self._guard():
            from data.benchmarks import load_index
            return load_index(name, as_of=self.as_of, index_dir=self.index_dir)

    def latest_reported_quarter(self, symbol: str) -> dict | None:
        """Latest quarter known by as_of. Skips YTD/9-month/annual-only rows."""
        with self._guard():
            from data.pit_fundamentals import known_as_of
            rows = known_as_of(symbol, self.as_of, path=self.fundamentals_path)
            usable = [
                r for r in rows
                if r.get("quarterly_usable") is True
                or (r.get("period_kind") == "quarter")
                or (
                    str(r.get("period") or "").lower() == "quarterly"
                    and r.get("quarterly_usable") is not False
                )
            ]
            if not usable:
                return None
            # prefer consolidated on the latest available_at
            usable.sort(key=lambda r: (r.get("available_at") or "", r.get("consol_basis") == "CONSOLIDATED"))
            row = usable[-1]
            return self._with_provenance("fundamentals_quarter", row)

    def quality(self, symbol: str, *, required: tuple[str, ...] | None = None) -> dict:
        return self.data_quality(symbol, required=required)

    def data_quality(self, symbol: str, *, required: tuple[str, ...] | None = None) -> dict:
        frame = self.prices(symbol)
        req = required or ("PRICE_OK", "UNIVERSE_OK")
        return evaluate_gates(symbol, self.as_of, required=req, frame=frame)

    def replay_hash(self) -> str:
        return snapshot_hash(self.manifest)

    def _with_provenance(self, kind: str, row: dict) -> dict:
        out = dict(row)
        out.setdefault("provenance", {
            "kind": kind,
            "source": row.get("source"),
            "available_at": row.get("available_at"),
            "version": row.get("parser_version") or row.get("mapping_version"),
            "quality": row.get("field_quality") or row.get("data_quality") or row.get("pit_status"),
            "pit": bool(row.get("available_at")),
            "static_backfill": row.get("pit_status") == "STATIC_BACKFILL",
            "raw_evidence_id": row.get("raw_hash") or row.get("row_id") or row.get("event_id"),
            "as_of": self.as_of,
        })
        return out


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
