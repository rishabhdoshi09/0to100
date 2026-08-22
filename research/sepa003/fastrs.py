"""Vectorized as-of RS using FastInvestable close arrays. Same formula as rs_cs_v1."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from research.sepa.config import SepaConfig
from research.sepa.rs import percentile_rank


class FastRS:
    def __init__(self, fast, config: SepaConfig):
        self.fast = fast
        self.horizons = tuple(int(x) for x in config.rs_horizons)
        self.weights = tuple(float(x) for x in config.rs_weights)

    def table(self, as_of, universe: Sequence[str]) -> dict:
        scores: dict[str, float] = {}
        comps: dict[str, dict[str, float]] = {}
        as_ns = None
        from research.sepa.universe_pit import _asof_ns
        as_ns = _asof_ns(as_of)
        need = max(self.horizons)
        for sym in universe:
            i = self.fast._pos.get(str(sym).upper())
            if i is None:
                continue
            j = self.fast.loc_as_of(self.fast._dates[i], as_ns)
            close = self.fast._close[i]
            if j < need:
                continue
            last = float(close[j])
            if last <= 0 or last != last:
                continue
            ok = True
            parts = {}
            acc = 0.0
            for n, w in zip(self.horizons, self.weights):
                start = float(close[j - n])
                if start <= 0 or start != start:
                    ok = False
                    break
                r = last / start - 1.0
                parts[f"r{n}"] = r
                acc += w * r
            if not ok:
                continue
            scores[str(sym).upper()] = acc
            comps[str(sym).upper()] = parts
        uni = list(scores.values())
        table = {"scores": scores, "components": comps, "percentiles": {}, "n_ranked": len(uni)}
        for sym, sc in scores.items():
            table["percentiles"][sym] = percentile_rank(sc, uni)
        return table


def lookup_fast(table: Mapping, symbol: str) -> dict:
    sym = str(symbol).upper()
    if sym not in (table.get("scores") or {}):
        return {"available": False, "percentile": None, "score": None, "components": {}, "n_ranked": table.get("n_ranked")}
    return {
        "available": True,
        "percentile": table["percentiles"].get(sym),
        "score": table["scores"].get(sym),
        "components": table.get("components", {}).get(sym) or {},
        "n_ranked": table.get("n_ranked"),
    }
