"""Structured rationale dict → ResearchRationale. NEVER stores raw chain-of-thought."""
from __future__ import annotations

from research.intelligence import schemas as SC

_ALLOWED = ("observation", "hypothesis", "supporting_evidence", "conflicting_evidence",
            "decision", "uncertainty", "next_test")


def decode(raw: dict, *, ctx: dict | None = None) -> list:
    ctx = ctx or {}
    if not raw:
        return []
    def _t(v):  # tuples for the evidence lists, str for the rest
        return tuple(v) if isinstance(v, (list, tuple)) else v
    fields = {k: _t(raw.get(k, () if "evidence" in k else "")) for k in _ALLOWED}
    return [SC.ResearchRationale(
        strategy_id=str(raw.get("strategy_id", ctx.get("strategy_id", ""))),
        strategy_version=int(ctx.get("strategy_version", 0)),
        rules_hash=str(ctx.get("rules_hash", "")), source="explanation",
        event_ts=str(ctx.get("event_ts", "")), **fields)]
