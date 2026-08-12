"""Frozen StrategySpec → StrategyDefinition (rules frozen, rules_hash = config hash)."""
from __future__ import annotations

from research.intelligence import schemas as SC


def decode(spec, *, ctx: dict | None = None) -> list:
    ctx = ctx or {}
    return [SC.StrategyDefinition(
        strategy_id=spec.strategy_id, strategy_version=spec.version,
        rules_hash=spec.config_hash(), data_snapshot_id=str(ctx.get("data_snapshot_id", "")),
        source="strategy", event_ts=str(ctx.get("event_ts", "")),
        family=spec.family, entry_rules=tuple(spec.entry_rules),
        exit_rules=tuple(spec.exit_rules), stop_rules=tuple(spec.stop_rules),
        max_holding_days=int(spec.max_holding_days), frozen=True)]
