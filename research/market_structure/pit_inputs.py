"""Build PIT close panels for market-structure research via PitContract."""
from __future__ import annotations

from typing import Iterable

from research.intelligence.data.pit_contract import PitContract, DOMAIN_BARS
from research.intelligence import data_state as DS


def closes_from_pit(
    pit: PitContract,
    symbols: Iterable[str],
    *,
    as_of: str,
) -> dict[str, list[float]]:
    """Return ``{symbol: [closes ≤ as_of]}`` skipping blocked/incomplete reads."""
    out: dict[str, list[float]] = {}
    for sym in symbols:
        result = pit.as_of(DOMAIN_BARS, when=as_of, symbol=sym)
        if result.status not in (DS.READY, DS.DEGRADED, DS.STALE) or not result.data:
            continue
        out[str(sym).upper()] = [float(b.close) for b in result.data]
    return out
