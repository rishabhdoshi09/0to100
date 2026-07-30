"""
⚙️ Per-strategy rule→signal evaluator (Phase 2, minimal).

Closes the signal-realism gap: instead of trading the scanner's setups, each FROZEN strategy
derives its OWN entries/exits from its rules, bar-by-bar, chronologically. No future bar can
affect an earlier decision. Fills, slippage, costs and gap-through-stop reuse the existing
realistic PaperBook — this module only decides WHEN to enter/exit.

Coverage is honest: only families with a registered adapter are supported. An unsupported
family raises `UnsupportedStrategy` — it NEVER silently falls back to scanner signals.
"""
from __future__ import annotations

from dataclasses import dataclass


class UnsupportedStrategy(Exception):
    """This strategy family has no bar-by-bar adapter yet — fail loud, never fall back."""


@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float


# ── family adapters: given the history SO FAR (point-in-time), decide an entry ───

def _breakout_adapter(spec, history: list, i: int) -> dict | None:
    """Enter when today's close breaks above the highest high of the prior `lookback` bars —
    a transparent, point-in-time breakout. Uses only bars[:i+1]. Stop below the base low."""
    lookback = 20
    if i < lookback:
        return None
    prior = history[i - lookback:i]                 # strictly BEFORE today — no look-ahead
    pivot = max(b.high for b in prior)
    base_low = min(b.low for b in prior)
    today = history[i]
    if today.close > pivot > 0 and base_low < today.close:
        return {"symbol": None, "entry": today.close, "stop": base_low,
                "target": today.close + 2 * (today.close - base_low),
                "max_hold": spec.max_holding_days}
    return None


_ADAPTERS = {
    "breakout": _breakout_adapter,
    "volatility_contraction": _breakout_adapter,   # same structural entry (base breakout)
}


def supported_families() -> list:
    return sorted(_ADAPTERS)


def is_supported(family: str) -> bool:
    return family in _ADAPTERS


def entries_for(spec, symbol: str, history: list) -> list:
    """Return the list of entry signals this frozen strategy fires over `history` (a
    chronological list of Bar). Point-in-time: bar i sees only bars[:i+1]. Raises for an
    unsupported family rather than inventing generic signals."""
    if spec.family not in _ADAPTERS:
        raise UnsupportedStrategy(
            f"no bar-by-bar adapter for family {spec.family!r}; "
            f"supported: {supported_families()}")
    adapter = _ADAPTERS[spec.family]
    out = []
    for i in range(len(history)):                   # serial + chronological, by contract
        sig = adapter(spec, history, i)
        if sig is not None:
            sig = dict(sig); sig["symbol"] = symbol
            sig["strategy_id"] = spec.strategy_id
            sig["date"] = history[i].date
            out.append(sig)
    return out
