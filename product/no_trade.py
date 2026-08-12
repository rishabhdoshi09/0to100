"""Explain a no-trade outcome without inventing missing funnel counts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FunnelStage:
    label: str
    count: int | None
    detail: str


@dataclass(frozen=True)
class NoTradeExplanation:
    headline: str
    stages: tuple[FunnelStage, ...]
    top_reasons: tuple[tuple[str, int], ...]


def _cycle_value(cycle: Any, *keys: str) -> Any:
    if not isinstance(cycle, Mapping):
        return None
    for key in keys:
        if cycle.get(key) is not None:
            return cycle.get(key)
    return None


def build_no_trade_explanation(
    scan_payload: Mapping[str, Any] | None,
    refusals: Iterable[Any] = (),
    last_cycle: Mapping[str, Any] | None = None,
    open_positions: int = 0,
) -> NoTradeExplanation:
    summary = dict((scan_payload or {}).get("summary", {}))
    universe = (scan_payload or {}).get("universe_size")
    setup_count = summary.get("with_any_setup")
    momentum = summary.get("momentum")
    ready = summary.get("ready_to_trade")

    refusal_reasons: list[str] = []
    for item in refusals or ():
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            refusal_reasons.append(str(item[1]))
        elif isinstance(item, Mapping):
            refusal_reasons.append(str(item.get("reason") or item.get("message") or "Unknown safety refusal"))
        else:
            refusal_reasons.append(str(item))
    reason_counts = tuple(Counter(refusal_reasons).most_common(5))

    decision = str(_cycle_value(last_cycle, "decision", "status", "result") or "").upper()
    intents = _cycle_value(last_cycle, "intents", "eligible_intents", "proposals")
    if isinstance(intents, (list, tuple, set, dict)):
        intent_count: int | None = len(intents)
    elif isinstance(intents, (int, float)):
        intent_count = int(intents)
    else:
        intent_count = None

    if open_positions:
        headline = f"No new trade was needed; {open_positions} existing paper position(s) are being managed."
    elif reason_counts:
        headline = f"No trade passed the final safety checks. Main reason: {reason_counts[0][0]}."
    elif decision in {"NO_ELIGIBLE_TRADE", "NO_TRADE", "NO SAFE TRADE FOUND"}:
        headline = "The latest autonomous cycle found no safe trade."
    elif ready == 0:
        headline = "The latest scan found no entry-ready setup."
    elif not scan_payload:
        headline = "Run a momentum scan to see the full candidate funnel."
    else:
        headline = "No paper trade was opened; the final backend decision did not qualify an entry."

    stages = (
        FunnelStage("Stocks scanned", int(universe) if universe is not None else None,
                    "Broad NSE universe evaluated in the latest saved scan."),
        FunnelStage("Stocks with any setup", int(setup_count) if setup_count is not None else None,
                    "Passed the scanner's basic price, liquidity and setup checks."),
        FunnelStage("Momentum candidates", int(momentum) if momentum is not None else None,
                    "Met the underlying-stock momentum condition."),
        FunnelStage("Entry-ready candidates", int(ready) if ready is not None else None,
                    "Not extended and marked ready by the scanner."),
        FunnelStage("Backend trade proposals", intent_count,
                    "Only shown when the latest PAPER_AUTO cycle exposes this count."),
        FunnelStage("Safety refusals", len(refusal_reasons),
                    "Portfolio and risk checks that explicitly rejected an entry."),
    )
    return NoTradeExplanation(headline=headline, stages=stages, top_reasons=reason_counts)
