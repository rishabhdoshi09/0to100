"""
🏗️ Production cycle-context builder — turns a pinned snapshot into a real CycleContext.

Replaces fixture injection: given a verified snapshot (via a `SnapshotBarProvider`) and the
validated strategy registry, it assembles the point-in-time universe data, benchmark, dataset
tier and per-strategy readiness for one cycle — pinned to exactly ONE snapshot id.

A strategy that the snapshot can't serve is disabled with an exact reason (Part 13); the rest
still run. Forward eligibility (Part 14) gates whether NEW entries may open — historical
research eligibility is separate from current forward-trading eligibility.
"""
from __future__ import annotations

from research.intelligence.runtime.cycle_context import CycleContext
from research.intelligence import data_state as DS
from research.intelligence import strategy_runtime as RT

# per-strategy runtime readiness (Part 13)
READY = "READY"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
MISSING_BENCHMARK = "MISSING_BENCHMARK"
UNSUPPORTED_RUNTIME = "UNSUPPORTED_RUNTIME"
OWNER_DISABLED = "OWNER_DISABLED"

_MIN_HISTORY = {"cross_sectional_momentum": 121, "relative_strength": 121,
                "sector_rotation": 121, "trend_following": 51, "pullback": 51,
                "breakout": 21, "volatility_contraction": 21}
_BENCHMARK_FAMILIES = {"relative_strength", "sector_rotation"}


def strategy_readiness(spec, provider, as_of: str) -> str:
    fam = getattr(spec, "family", "")
    if not RT.is_supported(fam):
        return UNSUPPORTED_RUNTIME
    if fam in _BENCHMARK_FAMILIES and not provider.benchmark(as_of):
        return MISSING_BENCHMARK
    need = _MIN_HISTORY.get(fam, 21)
    universe = provider.universe(as_of)
    if not universe:
        return INSUFFICIENT_HISTORY
    # at least one symbol must carry the required lookback as-of the date
    deepest = max((len(provider.universe_history(s, as_of)) for s in universe), default=0)
    return READY if deepest >= need else INSUFFICIENT_HISTORY


def _freshness_days(provider, as_of: str) -> float:
    latest = provider.latest_available_date()
    if not latest:
        return 999.0
    return 0.0 if latest >= as_of else 999.0        # calendar-accurate freshness is deferred


def build_context_from_snapshot(provider, registry, *, as_of: str, mode: str,
                                market_regime: str = "RISK_ON", config_hash: str = "cfg0",
                                clusters: dict | None = None, max_universe: int = 500
                                ) -> tuple:
    """Return (CycleContext, readiness_map). `provider` is a SnapshotBarProvider; `registry`
    a StrategyRegistry. Pins the snapshot id into the context."""
    sid = provider.snapshot_id
    universe = provider.universe(as_of)[:max_universe]
    bench = provider.benchmark(as_of)

    strategies, data, readiness = [], {}, {}
    for spec in registry.deployable_specs():
        rr = strategy_readiness(spec, provider, as_of)
        readiness[spec.strategy_id] = rr
        if rr not in (READY,):
            continue
        strategies.append(spec)
        # point-in-time universe bar history for this strategy
        data[spec.strategy_id] = {sym: provider.universe_history(sym, as_of)
                                  for sym in universe}

    # dataset tier + forward eligibility from real snapshot health
    health = dict(provider.health())
    health["freshness_days"] = _freshness_days(provider, as_of)
    tier, _reasons = DS.classify_tier(health)
    forward = DS.forward_eligible(tier)

    ctx = CycleContext(
        as_of_date=as_of, cycle_type="paper_session", mode=mode,
        data_ok=bool(provider.snapshot and universe), data_snapshot_id=sid,
        market_regime=market_regime, dataset_tier=tier, config_hash=config_hash,
        registry_version=f"reg{len(registry.all())}", strategies=strategies, data=data,
        clusters=clusters or {}, benchmark=bench)
    # forward eligibility is a distinct gate on NEW entries (historical research ≠ forward)
    ctx.forward_eligible = forward
    return ctx, readiness
