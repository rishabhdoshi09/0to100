"""
📖 Snapshot — read-only, point-in-time accessor over one committed snapshot.

Loads the immutable CSVs once and answers PIT queries. It NEVER returns a bar dated after the
`through` date, never reaches the network, never fabricates data. Bars are returned as
`strategy_runtime.Bar` so the adapters consume them directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from research.intelligence.strategy_runtime import Bar


class Snapshot:
    def __init__(self, sdir):
        self.dir = Path(sdir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.snapshot_id = self.manifest["snapshot_id"]
        self._equity = _load(self.dir / "bars_equity.csv",
                             ["symbol", "date", "open", "high", "low", "close", "volume", "series"])
        self._index = _load(self.dir / "index_daily.csv",
                            ["name", "date", "open", "high", "low", "close"])
        # index per symbol for fast PIT slicing (sorted by date)
        self._by_symbol: dict[str, list] = {}
        for r in self._equity:
            self._by_symbol.setdefault(r["symbol"], []).append(r)
        for rows in self._by_symbol.values():
            rows.sort(key=lambda r: r["date"])
        self._bench_rows = sorted(self._index, key=lambda r: r["date"])

    # ── point-in-time reads ──────────────────────────────────────────────────────
    def bars(self, symbol: str, through: str, adjustment: str = "raw") -> list:
        rows = self._by_symbol.get(symbol.upper(), [])
        return [_bar(r) for r in rows if r["date"] <= through]     # never past `through`

    def symbols(self) -> list:
        return sorted(self._by_symbol)

    def universe(self, on_date: str) -> list:
        """Symbols that actually trade ON `on_date` (point-in-time — no future members)."""
        return sorted(s for s, rows in self._by_symbol.items()
                      if any(r["date"] == on_date for r in rows))

    def benchmark(self, through: str, name: str | None = None) -> list:
        rows = [r for r in self._bench_rows
                if r["date"] <= through and (name is None or r["name"] == name)]
        return [_bar(r) for r in rows]

    def has_benchmark(self) -> bool:
        return bool(self._bench_rows)

    def latest_available_date(self) -> str | None:
        return self.manifest.get("last_trading_date")

    def health(self) -> dict:
        return {"has_prices": bool(self._equity),
                "has_benchmark": self.has_benchmark(),
                "instrument_count": self.manifest.get("instrument_count", 0),
                "last_trading_date": self.latest_available_date(),
                "date_range": self.manifest.get("date_range"),
                **{k: self.manifest.get(k) for k in
                   ("has_universe_history", "adjustment_consistent",
                    "corporate_action_coverage", "missing_session_rate",
                    "validation_errors", "freshness_days") if k in self.manifest}}

    def coverage_for(self, spec) -> dict:
        """What this snapshot can offer a strategy (used by data-aware registry readiness)."""
        from research.intelligence import strategy_runtime as RT
        cross = RT.is_cross_sectional(getattr(spec, "family", ""))
        return {"runtime_supported": RT.is_supported(getattr(spec, "family", "")),
                "has_benchmark": self.has_benchmark(),
                "cross_sectional": cross,
                "instrument_count": self.manifest.get("instrument_count", 0)}


def _load(path: Path, header: list) -> list:
    if not path.exists():
        return []
    out = []
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split(",")
        out.append(dict(zip(header, parts)))
    return out


def _bar(r: dict) -> Bar:
    return Bar(date=r["date"], open=float(r["open"]), high=float(r["high"]),
               low=float(r["low"]), close=float(r["close"]))
