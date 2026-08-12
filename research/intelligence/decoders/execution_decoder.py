"""Closed paper trade → ExecutionAssessment (fill quality / cost / exit reason)."""
from __future__ import annotations

from research.intelligence import schemas as SC


def decode(trade: dict, *, ctx: dict | None = None) -> list:
    ctx = ctx or {}
    return [SC.ExecutionAssessment(
        strategy_id=str(trade.get("strategy_id", "")),
        strategy_version=int(ctx.get("strategy_version", 0)),
        rules_hash=str(ctx.get("rules_hash", "")),
        data_snapshot_id=str(ctx.get("data_snapshot_id", "")),
        source="execution", event_ts=str(trade.get("exit_date", "")),
        symbol=str(trade.get("symbol", "")),
        intended_price=float(trade.get("entry_price", 0.0)),
        fill_price=float(trade.get("exit_price", 0.0)),
        slippage_bps=float(ctx.get("slippage_bps", 0.0)),
        cost=float(ctx.get("cost", 0.0)),
        exit_reason=str(trade.get("exit_reason", "")))]
