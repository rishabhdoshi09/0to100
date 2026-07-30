"""Regime / breadth read → MarketContext. Deterministic."""
from __future__ import annotations

from research.intelligence import schemas as SC


def decode(raw: dict, *, ctx: dict | None = None) -> list:
    ctx = ctx or {}
    if not raw:
        return []
    return [SC.MarketContext(
        source="market", event_ts=str(raw.get("date", ctx.get("event_ts", ""))),
        data_snapshot_id=str(ctx.get("data_snapshot_id", "")),
        regime=str(raw.get("regime", "UNKNOWN")), breadth=str(raw.get("breadth", "")),
        nifty_trend=str(raw.get("nifty_trend", "")), vix=float(raw.get("vix", 0.0) or 0.0))]
