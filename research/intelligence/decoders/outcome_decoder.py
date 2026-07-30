"""Closed paper trade → OutcomeObservation (realized R / pnl / split)."""
from __future__ import annotations

from research.intelligence import schemas as SC


def decode(trade: dict, *, ctx: dict | None = None) -> list:
    ctx = ctx or {}
    return [SC.OutcomeObservation(
        strategy_id=str(trade.get("strategy_id", "")),
        strategy_version=int(ctx.get("strategy_version", 0)),
        rules_hash=str(ctx.get("rules_hash", "")),
        data_snapshot_id=str(ctx.get("data_snapshot_id", "")),
        source="outcome", event_ts=str(trade.get("exit_date", "")),
        symbol=str(trade.get("symbol", "")), split=str(ctx.get("split", "forward")),
        realized_R=float(trade.get("realized_R", 0.0)), pnl=float(trade.get("pnl", 0.0)),
        regime=str(ctx.get("regime", "")), sector=str(ctx.get("sector", "")))]
