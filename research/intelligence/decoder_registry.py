"""
🧩 Decoder registry — maps a source kind to its deterministic decoder and runs them.

`decode(kind, raw, ctx)` returns canonical records; feeding those to an EventStore is
idempotent, so re-decoding the same raw input adds nothing new. Read-only decoding jobs are
independent and may be parallelised safely; the store write stays single-writer.
"""
from __future__ import annotations

from research.intelligence.decoders import (
    signal_decoder, market_decoder, strategy_decoder,
    execution_decoder, outcome_decoder, explanation_decoder,
)

_REGISTRY = {
    "signal": signal_decoder.decode,
    "market": market_decoder.decode,
    "strategy": strategy_decoder.decode,
    "execution": execution_decoder.decode,
    "outcome": outcome_decoder.decode,
    "explanation": explanation_decoder.decode,
}


def kinds() -> list:
    return sorted(_REGISTRY)


def decode(kind: str, raw, *, ctx: dict | None = None) -> list:
    if kind not in _REGISTRY:
        raise KeyError(f"no decoder registered for source kind {kind!r}")
    return _REGISTRY[kind](raw, ctx=ctx or {})


def decode_into(store, kind: str, raws, *, ctx: dict | None = None) -> int:
    """Decode a batch and append to the store. Returns the count of NEW records stored
    (duplicates from re-decoding are ignored by the store)."""
    stored = 0
    for raw in (raws or []):
        for rec in decode(kind, raw, ctx=ctx):
            stored += int(store.append(rec))
    return stored
