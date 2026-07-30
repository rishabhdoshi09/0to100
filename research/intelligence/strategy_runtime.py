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


# ── shared point-in-time feature helpers (Phase 13 — one calc, reused by adapters) ─
# All read only history[:i+1]; none can look ahead.

def _sma(history, i, n):
    if i + 1 < n:
        return None
    return sum(b.close for b in history[i - n + 1:i + 1]) / n


def _ema(history, i, n):
    if i + 1 < n:
        return None
    k = 2.0 / (n + 1)
    e = history[i - n + 1].close
    for b in history[i - n + 2:i + 1]:
        e = b.close * k + e * (1 - k)
    return e


def _atr(history, i, n=14):
    if i < n:
        return None
    trs = []
    for j in range(i - n + 1, i + 1):
        pc = history[j - 1].close
        h, l = history[j].high, history[j].low
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / n


def _ret(history, i, n):
    """Simple return over the prior n bars, ending at bar i (point-in-time)."""
    if i < n or history[i - n].close <= 0:
        return None
    return history[i].close / history[i - n].close - 1.0


def _bars_through(history, as_of):
    """The sublist of `history` up to AND INCLUDING the bar dated `as_of`, or None if the
    symbol has no bar that day (⇒ not in the tradable universe on `as_of`)."""
    for idx, b in enumerate(history):
        if b.date == as_of:
            return history[:idx + 1], idx
    return None, None


# ── single-symbol adapters (adapter(spec, history, i) -> signal|None) ─────────────

def _trend_following_adapter(spec, history, i):
    """Long a stock in a confirmed uptrend that breaks a short-term high. Stop = 2·ATR;
    target = 3·ATR (a volatility-adjusted, frozen exit)."""
    sma = _sma(history, i, 50)
    atr = _atr(history, i, 14)
    if sma is None or atr is None or atr <= 0:
        return None
    prior_high = max(b.high for b in history[i - 20:i]) if i >= 20 else None
    today = history[i]
    if prior_high and today.close > sma and today.close > prior_high:
        stop = today.close - 2 * atr
        if stop <= 0:
            return None
        return {"symbol": None, "entry": today.close, "stop": stop,
                "target": today.close + 3 * atr, "max_hold": spec.max_holding_days,
                "exit_policy": "atr_2_3"}
    return None


def _pullback_adapter(spec, history, i):
    """In an uptrend (above 50-SMA), buy a pullback that tags the 20-EMA and closes back up
    (confirmation). Invalidation = the pullback low."""
    sma = _sma(history, i, 50)
    ema = _ema(history, i, 20)
    if sma is None or ema is None:
        return None
    today = history[i]
    tol = 0.02 * ema
    tagged = today.low <= ema + tol
    if today.close > sma and tagged and today.close > today.open:
        stop = min(today.low, ema) * 0.995
        if stop <= 0 or stop >= today.close:
            return None
        return {"symbol": None, "entry": today.close, "stop": stop,
                "target": today.close + 2 * (today.close - stop),
                "max_hold": spec.max_holding_days, "exit_policy": "structural_2R"}
    return None


_SINGLE_ADAPTERS = {
    "breakout": _breakout_adapter,
    "volatility_contraction": _breakout_adapter,   # same structural entry (base breakout)
    "trend_following": _trend_following_adapter,
    "pullback": _pullback_adapter,
}


# ── cross-sectional adapters (rank the contemporaneous universe, pick top N) ──────

def _rank_and_pick(spec, as_of, universe_data, score_fn, top_n=5):
    """Score every symbol that has a bar on `as_of` (point-in-time universe), rank, and emit
    entries for the top N. Never uses a symbol that isn't trading on `as_of`."""
    scored = []
    for symbol, history in universe_data.items():
        sub, i = _bars_through(history, as_of)
        if sub is None:
            continue                                # not in the universe on as_of
        s = score_fn(sub, i)
        if s is None:
            continue
        atr = _atr(sub, i, 14)
        if atr is None or atr <= 0:
            continue
        scored.append((s, symbol, sub[i], atr))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for _s, symbol, bar, atr in scored[:top_n]:
        stop = bar.close - 2 * atr
        if stop <= 0:
            continue
        out.append({"symbol": symbol, "entry": bar.close, "stop": stop,
                    "target": bar.close + 3 * atr, "max_hold": spec.max_holding_days,
                    "date": as_of, "strategy_id": spec.strategy_id,
                    "exit_policy": "atr_2_3"})
    return out


def _xsec_momentum(spec, as_of, universe_data, benchmark=None):
    lookback = int((spec.thresholds or {}).get("lookback", 120))
    return _rank_and_pick(spec, as_of, universe_data,
                          lambda h, i: _ret(h, i, lookback))


def _relative_strength(spec, as_of, universe_data, benchmark=None):
    """Rank by return RELATIVE to the benchmark over the lookback (needs a benchmark series)."""
    lookback = int((spec.thresholds or {}).get("lookback", 120))
    bench_ret = None
    if benchmark:
        bsub, bi = _bars_through(benchmark, as_of)
        if bsub is not None:
            bench_ret = _ret(bsub, bi, lookback)
    if bench_ret is None:
        return []                                   # no benchmark ⇒ honest no-signal, not zero
    return _rank_and_pick(spec, as_of, universe_data,
                          lambda h, i: (_ret(h, i, lookback) - bench_ret)
                          if _ret(h, i, lookback) is not None else None)


_CROSS_ADAPTERS = {
    "cross_sectional_momentum": _xsec_momentum,
    "relative_strength": _relative_strength,
    "sector_rotation": _relative_strength,          # same PIT relative-strength ranking
}


def supported_families() -> list:
    return sorted(set(_SINGLE_ADAPTERS) | set(_CROSS_ADAPTERS))


def is_supported(family: str) -> bool:
    return family in _SINGLE_ADAPTERS or family in _CROSS_ADAPTERS


def is_cross_sectional(family: str) -> bool:
    return family in _CROSS_ADAPTERS


def signals(spec, as_of_date: str, universe_data: dict, benchmark=None) -> list:
    """Unified entry point the loop calls per strategy per cycle. Returns the entry signals a
    FROZEN strategy fires on `as_of_date` — single-symbol adapters loop the universe; cross-
    sectional adapters rank it. Raises for an unsupported family (never scanner fallback)."""
    fam = spec.family
    if fam in _CROSS_ADAPTERS:
        return _CROSS_ADAPTERS[fam](spec, as_of_date, universe_data, benchmark)
    if fam in _SINGLE_ADAPTERS:
        adapter = _SINGLE_ADAPTERS[fam]
        out = []
        for symbol, history in universe_data.items():
            sub, i = _bars_through(history, as_of_date)
            if sub is None:
                continue
            sig = adapter(spec, sub, i)
            if sig is not None:
                sig = dict(sig); sig["symbol"] = symbol
                sig["strategy_id"] = spec.strategy_id; sig["date"] = as_of_date
                out.append(sig)
        return out
    raise UnsupportedStrategy(
        f"no bar-by-bar adapter for family {fam!r}; supported: {supported_families()}")


def entries_for(spec, symbol: str, history: list) -> list:
    """Single-symbol convenience wrapper (also used by tests). Returns every entry a frozen
    SINGLE-symbol strategy fires over `history`. Cross-sectional families must use `signals()`
    (they need the whole universe); calling this for one raises."""
    if spec.family in _CROSS_ADAPTERS:
        raise UnsupportedStrategy(
            f"family {spec.family!r} is cross-sectional — use signals(spec, as_of, universe)")
    if spec.family not in _SINGLE_ADAPTERS:
        raise UnsupportedStrategy(
            f"no bar-by-bar adapter for family {spec.family!r}; "
            f"supported: {supported_families()}")
    adapter = _SINGLE_ADAPTERS[spec.family]
    out = []
    for i in range(len(history)):                   # serial + chronological, by contract
        sig = adapter(spec, history, i)
        if sig is not None:
            sig = dict(sig); sig["symbol"] = symbol
            sig["strategy_id"] = spec.strategy_id
            sig["date"] = history[i].date
            out.append(sig)
    return out
