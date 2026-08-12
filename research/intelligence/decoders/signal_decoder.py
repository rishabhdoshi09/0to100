"""Raw entry signal → CanonicalSignal. Deterministic; empty in, empty out (honest)."""
from __future__ import annotations

from research.intelligence import schemas as SC


def decode(raw: dict, *, ctx: dict | None = None) -> list:
    ctx = ctx or {}
    sym = str(raw.get("symbol", "")).strip().upper()
    entry, stop, target = raw.get("entry"), raw.get("stop"), raw.get("target")
    if not (sym and entry and stop and target and float(entry) > float(stop) > 0):
        return []                              # invalid/empty ⇒ no event, never fabricate
    return [SC.CanonicalSignal(
        strategy_id=str(raw.get("strategy_id", ctx.get("strategy_id", ""))),
        strategy_version=int(ctx.get("strategy_version", 0)),
        rules_hash=str(ctx.get("rules_hash", "")),
        data_snapshot_id=str(ctx.get("data_snapshot_id", "")),
        source="signal", event_ts=str(raw.get("date", ctx.get("event_ts", ""))),
        symbol=sym, direction=str(raw.get("direction", "LONG")),
        entry=float(entry), stop=float(stop), target=float(target),
        max_hold=int(raw.get("max_hold", 0)),
        rationale=str(raw.get("rationale", ""))[:200])]
