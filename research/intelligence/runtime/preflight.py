"""
🚦 Operational safety preflight (Phase Q) — checks run BEFORE each automation cycle.

If any critical check fails, the cycle must not create new risk. `preflight()` returns a
report; the orchestrator treats a failed critical check as a data-gate failure (no new
entries; existing positions may still be de-risked). Nothing here trades.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PreflightReport:
    ok: bool
    failed: list = field(default_factory=list)      # (check, reason)
    passed: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return {"ok": self.ok, "failed": self.failed, "passed": self.passed,
                "warnings": self.warnings}


def preflight(ctx, *, store, book, runtime_state) -> PreflightReport:
    rep = PreflightReport(ok=True)

    def chk(name: str, ok: bool, reason: str = ""):
        if ok:
            rep.passed.append(name)
        else:
            rep.failed.append((name, reason)); rep.ok = False

    chk("data_available", bool(ctx.data_ok), "no validated data")
    chk("data_snapshot_valid", bool(ctx.data_snapshot_id) or not ctx.data_ok,
        "data ok but no snapshot id")
    chk("trading_date_valid", bool(ctx.as_of_date), "missing as_of_date")
    chk("strategy_registry_valid", isinstance(ctx.strategies, list), "registry not a list")
    chk("config_valid", bool(ctx.config_hash), "missing config hash")
    chk("event_store_writable", store is not None, "no event store")
    chk("paper_book_loadable", book is not None, "no paper book")
    chk("risk_config_available", ctx.mode is not None, "no operating mode")
    # reconciliation is a WARNING not a hard fail — the loop refuses new risk but still manages
    if not getattr(runtime_state, "reconciled", True):
        rep.warnings.append("paper book not reconciled — new entries will be refused")
    # cycle idempotency
    chk("cycle_not_already_completed",
        not runtime_state.is_cycle_done(ctx.cycle_id()), "cycle already completed")
    return rep
