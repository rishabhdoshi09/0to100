"""Second-stage due diligence — after a scanner already shortlisted a name.

This package never scans the market. It reads persisted fundamentals, news,
and the current scan/long-term rows, then returns an evidence-backed
SUPPORTS / NEUTRAL / CONTRADICTS view of the technical setup.

StockResearchEngine is the public name. Scanners and manual search share it.

Long-running market-operations acquisition calls are wrapped here at the
package boundary so the existing public ``product.due_diligence.acquire`` API
stays compatible while production jobs receive an enforceable process-level
wall-clock deadline.  Short unit/replay calls keep the established in-process
path; the isolated child explicitly opts out to avoid recursion.
"""
from __future__ import annotations

import os
import time


# Market-operations calls use 90s per-symbol budgets and a 12-minute overall
# deadline.  A remaining budget >=30s identifies that durable/provider path
# without changing deterministic tests/replays that intentionally exercise the
# cooperative acquire functions with tiny synthetic deadlines.
_ISOLATION_MIN_REMAINING_S = 30.0


def _install_acquisition_isolation() -> None:
    if os.environ.get("QT_DD_ISOLATED_CHILD") == "1":
        return

    from product.due_diligence import acquire as acquire_mod
    from product.due_diligence.isolation import (
        DEFAULT_SYMBOL_TIMEOUT_S,
        acquire_shortlist_isolated,
        acquire_symbol_isolated,
    )

    original_symbol = acquire_mod.acquire_symbol
    original_shortlist = acquire_mod.acquire_shortlist

    def should_isolate(deadline_monotonic: float | None, *, budget_s: float) -> bool:
        if deadline_monotonic is None:
            return False
        remaining = float(deadline_monotonic) - time.monotonic()
        return remaining >= _ISOLATION_MIN_REMAINING_S and float(budget_s) >= _ISOLATION_MIN_REMAINING_S

    def acquire_symbol(
        symbol: str,
        *,
        force: bool = False,
        datasets: list[str] | None = None,
        now=None,
        deadline_monotonic: float | None = None,
    ):
        # Historical/deterministic callers can inject ``now``; keep those
        # in-process.  Production market-ops does not inject a synthetic clock.
        if now is None and should_isolate(deadline_monotonic, budget_s=DEFAULT_SYMBOL_TIMEOUT_S):
            return acquire_symbol_isolated(
                symbol,
                force=force,
                datasets=datasets,
                deadline_monotonic=deadline_monotonic,
                timeout_s=DEFAULT_SYMBOL_TIMEOUT_S,
            )
        return original_symbol(
            symbol,
            force=force,
            datasets=datasets,
            now=now,
            deadline_monotonic=deadline_monotonic,
        )

    def acquire_shortlist(
        *,
        limit: int = acquire_mod.ACQUIRE_CAP,
        force: bool = False,
        scan_payload=None,
        deadline_monotonic: float | None = None,
        per_symbol_s: float = DEFAULT_SYMBOL_TIMEOUT_S,
        progress_cb=None,
    ):
        if should_isolate(deadline_monotonic, budget_s=per_symbol_s):
            return acquire_shortlist_isolated(
                limit=limit,
                force=force,
                scan_payload=scan_payload,
                deadline_monotonic=deadline_monotonic,
                per_symbol_s=per_symbol_s,
                progress_cb=progress_cb,
            )
        return original_shortlist(
            limit=limit,
            force=force,
            scan_payload=scan_payload,
            deadline_monotonic=deadline_monotonic,
            per_symbol_s=per_symbol_s,
            progress_cb=progress_cb,
        )

    acquire_symbol.__name__ = "acquire_symbol"
    acquire_symbol.__module__ = acquire_mod.__name__
    acquire_shortlist.__name__ = "acquire_shortlist"
    acquire_shortlist.__module__ = acquire_mod.__name__
    acquire_mod.acquire_symbol = acquire_symbol
    acquire_mod.acquire_shortlist = acquire_shortlist


_install_acquisition_isolation()

from product.due_diligence.engine import build_due_diligence
from product.due_diligence.research_engine import StockResearchEngine, investigate_stock
from product.due_diligence.suggest import suggest_tickers

__all__ = [
    "build_due_diligence",
    "investigate_stock",
    "StockResearchEngine",
    "suggest_tickers",
]
