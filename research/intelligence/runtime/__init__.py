"""Runtime: the authoritative, broker/UI-free orchestration of the two-brain paper loop."""
from research.intelligence.runtime.autonomous_loop import run_intelligence_cycle as _run_intelligence_cycle
from research.intelligence.runtime.cycle_result import IntelligenceCycleResult
from research.intelligence.runtime.cycle_context import CycleContext
from research.intelligence.runtime.runtime_state import RuntimeState, StrategyState
from research.intelligence.runtime.decision_accounting import finalize_cycle_decisions
from research.intelligence.runtime import modes, controls, preflight, events


def run_intelligence_cycle(ctx, *, store, book, runtime_state, knowledge=None,
                           alloc_cfg=None, backtest_R=None, backtest_trades=None):
    """Run the canonical cycle and close decision accounting before returning.

    The base loop remains the only mutation/orchestration authority.  This wrapper
    performs a deterministic post-cycle classification of signals the loop already
    generated, ensuring non-taken opportunities reach the paper self-feed instead
    of disappearing between allocation and learning.
    """
    result = _run_intelligence_cycle(
        ctx,
        store=store,
        book=book,
        runtime_state=runtime_state,
        knowledge=knowledge,
        alloc_cfg=alloc_cfg,
        backtest_R=backtest_R,
        backtest_trades=backtest_trades,
    )
    try:
        finalize_cycle_decisions(ctx, result, store=store)
        result.events_emitted = len(store)
        # The base loop persists before post-cycle accounting. Persist once more so
        # rejection events and the typed decision projection survive a restart.
        if hasattr(store, "save"):
            store.save()
        if hasattr(runtime_state, "save"):
            runtime_state.save()
    except Exception as exc:
        # Decision accounting must never turn a safe paper cycle into a failed
        # trading mutation. Surface the degradation loudly in the result instead.
        result.warnings.append(f"decision accounting unavailable: {type(exc).__name__}: {exc}")
    return result


__all__ = ["run_intelligence_cycle", "IntelligenceCycleResult", "CycleContext",
           "RuntimeState", "StrategyState", "modes", "controls", "preflight", "events",
           "finalize_cycle_decisions"]
