"""
🕰️ Point-in-time primitives for the Institutional Momentum Breakout framework.

WHY THIS MODULE EXISTS (architecture decision — see ADR-002).
The repository already has ATR / moving-average / relative-strength / breakout
implementations, but they are OPERATIONAL, not research-grade:
  • `scan/relative_strength.py` fetches Nifty live (Kite/yfinance) and anchors on
    `date.today()` — it cannot be evaluated "as of" a historical bar.
  • `scan/unified_scanner.py` / `scan/breakout_sniper.py` ATR & pattern code is
    coupled to live scan context and score calibration.
None of them can guarantee "use only bars available at the observation timestamp",
which is the whole point of this milestone. Reusing them would silently leak the
future into a historical study.

So this module is the ONE canonical set of *pure, point-in-time-safe* primitives
the research framework uses. Every function operates on a plain array and an
explicit integer index `i` — the observation bar (the most recent CLOSED bar known
at signal time). A primitive may read only `arr[: i + 1]`; reading `arr[i + 1:]`
is a temporal violation. `assert_no_future_read()` and the guards at the bottom
make that contract testable rather than a matter of trust.

Pure: numpy only, no I/O, no network, no clock.
"""
from __future__ import annotations

import numpy as np

# ── module version — any change to a computation here bumps this, which flows
#    into the framework config hash so historical observations stay attributable.
PIT_INDICATORS_VERSION = 1


# ══════════════════════════════════════════════════════════════════════════════
# The point-in-time contract
# ══════════════════════════════════════════════════════════════════════════════

class FutureLeak(Exception):
    """Raised when a computation is asked to read a bar at/after the future
    boundary. Fail CLOSED — a leaked future is a void observation, never a
    silently optimistic one."""


def window(arr, i: int, lookback: int) -> np.ndarray:
    """The only sanctioned way to grab history: the `lookback` bars ending at and
    INCLUDING bar `i`. Never returns anything past `i`. Raises if `i` is out of
    range so an off-by-one can't silently read the wrong bar."""
    a = np.asarray(arr, dtype=float)
    if i < 0 or i >= a.size:
        raise FutureLeak(f"observation index {i} out of range for size {a.size}")
    lo = max(0, i - lookback + 1)
    return a[lo: i + 1]


def assert_no_future_read(i: int, accessed_index: int) -> None:
    """Guard callers can use when they compute an index by hand: proves a read
    never touched a bar after the observation boundary `i`."""
    if accessed_index > i:
        raise FutureLeak(f"read bar {accessed_index} > observation bar {i}")


# ══════════════════════════════════════════════════════════════════════════════
# Trend / moving averages
# ══════════════════════════════════════════════════════════════════════════════

def sma(close, i: int, window_n: int) -> float:
    """Simple moving average of the `window_n` closes ending at bar `i`. NaN until
    there are enough bars (never fabricates a value from a short window)."""
    w = window(close, i, window_n)
    if w.size < window_n:
        return float("nan")
    return float(np.mean(w))


def ema(close, i: int, span: int) -> float:
    """Exponential moving average (span→alpha=2/(span+1)) over bars ≤ i. Warmed up
    from the first available bar; NaN if fewer than `span` bars exist."""
    a = np.asarray(close, dtype=float)[: i + 1]
    if a.size < span:
        return float("nan")
    alpha = 2.0 / (span + 1.0)
    e = a[0]
    for x in a[1:]:
        if np.isnan(x):
            continue
        e = alpha * x + (1 - alpha) * e
    return float(e)


def slope_pct(close, i: int, window_n: int, ago: int) -> float:
    """Slope of the `window_n`-SMA: today's SMA vs the SMA `ago` bars back, as a
    percentage. Positive = rising trend. NaN if either endpoint is unavailable."""
    if i - ago < 0:
        return float("nan")
    now = sma(close, i, window_n)
    then = sma(close, i - ago, window_n)
    if not np.isfinite(now) or not np.isfinite(then) or then <= 0:
        return float("nan")
    return float((now / then - 1.0) * 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# Volatility / range
# ══════════════════════════════════════════════════════════════════════════════

def true_range(high, low, close, i: int) -> float:
    """True range of bar `i` (needs the prior close). NaN at the first bar."""
    h = np.asarray(high, float); l = np.asarray(low, float); c = np.asarray(close, float)
    if i <= 0:
        return float("nan")
    prev_c = c[i - 1]
    return float(max(h[i] - l[i], abs(h[i] - prev_c), abs(l[i] - prev_c)))


def atr(high, low, close, i: int, window_n: int = 14) -> float:
    """Average true range over the `window_n` bars ending at `i` (Wilder-style
    simple mean of true ranges — deterministic, PIT-safe). NaN until warmed up."""
    if i < window_n:
        return float("nan")
    trs = [true_range(high, low, close, j) for j in range(i - window_n + 1, i + 1)]
    trs = [t for t in trs if np.isfinite(t)]
    if len(trs) < window_n:
        return float("nan")
    return float(np.mean(trs))


def atr_pct(high, low, close, i: int, window_n: int = 14) -> float:
    """ATR as a percentage of the close at bar `i`."""
    a = atr(high, low, close, i, window_n)
    c = float(np.asarray(close, float)[i])
    if not np.isfinite(a) or c <= 0:
        return float("nan")
    return float(a / c * 100.0)


def realised_vol_pct(close, i: int, window_n: int) -> float:
    """Annualisation-free realised volatility: std of daily returns over the
    `window_n` bars ending at `i`, in percent."""
    w = window(close, i, window_n + 1)
    if w.size < window_n + 1:
        return float("nan")
    rets = np.diff(w) / w[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size < 2:
        return float("nan")
    return float(np.std(rets, ddof=1) * 100.0)


def clv(high, low, close, i: int) -> float:
    """Close Location Value of bar `i`: where the close sat in the bar's range,
    −1 (at the low) … +1 (at the high). 0 when high==low."""
    h = float(np.asarray(high, float)[i]); l = float(np.asarray(low, float)[i])
    c = float(np.asarray(close, float)[i])
    if h <= l:
        return 0.0
    return float(((c - l) - (h - c)) / (h - l))


# ══════════════════════════════════════════════════════════════════════════════
# Returns / momentum / drawdown
# ══════════════════════════════════════════════════════════════════════════════

def ret_pct(close, i: int, lookback: int) -> float:
    """Percentage return from bar `i-lookback` to bar `i`. NaN if unavailable."""
    if i - lookback < 0:
        return float("nan")
    c = np.asarray(close, float)
    base = c[i - lookback]
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float((c[i] / base - 1.0) * 100.0)


def ret_pct_skip(close, i: int, lookback: int, skip: int) -> float:
    """`lookback`-to-`skip` return (e.g. 12-1 momentum): return from
    `i-lookback` to `i-skip`, excluding the most recent `skip` bars. NaN if
    unavailable."""
    if i - lookback < 0 or i - skip < 0:
        return float("nan")
    c = np.asarray(close, float)
    base = c[i - lookback]; end = c[i - skip]
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float((end / base - 1.0) * 100.0)


def max_drawdown_pct(close, i: int, lookback: int) -> float:
    """Worst peak-to-trough drawdown over the `lookback` window ending at `i`, as a
    positive percentage. 0 for a monotonically rising window."""
    w = window(close, i, lookback)
    if w.size < 2:
        return float("nan")
    peak = np.maximum.accumulate(w)
    dd = (peak - w) / peak
    dd = dd[np.isfinite(dd)]
    return float(np.max(dd) * 100.0) if dd.size else 0.0


def dist_from_high_pct(high, i: int, lookback: int) -> float:
    """% below the highest HIGH over the `lookback` bars ending at `i` (0 = at a
    new high). Point-in-time: the 52-week high uses only past bars."""
    w = window(high, i, lookback)
    if w.size == 0:
        return float("nan")
    hi = float(np.max(w))
    h_i = float(np.asarray(high, float)[i])
    if hi <= 0:
        return float("nan")
    return float((hi - h_i) / hi * 100.0)


def is_new_high(high, i: int, lookback: int) -> bool:
    """Did bar `i`'s high match/exceed the highest high of the PRIOR `lookback`
    bars? Uses [i-lookback, i-1] as the reference so a new high is genuinely new."""
    if i - lookback < 0:
        return False
    ref = np.asarray(high, float)[i - lookback: i]
    if ref.size == 0:
        return False
    return float(np.asarray(high, float)[i]) >= float(np.max(ref))


# ══════════════════════════════════════════════════════════════════════════════
# Volume / participation
# ══════════════════════════════════════════════════════════════════════════════

def volume_z(volume, i: int, window_n: int) -> float:
    """Z-score of bar `i`'s volume vs the `window_n` bars ending at `i-1` (the
    reference EXCLUDES the day itself, so a breakout-volume spike is measured
    against its own quiet base). NaN if unavailable."""
    if i - 1 < 0:
        return float("nan")
    ref = window(volume, i - 1, window_n)
    ref = ref[np.isfinite(ref)]
    if ref.size < max(5, window_n // 2):
        return float("nan")
    mu = float(np.mean(ref)); sd = float(np.std(ref, ddof=1))
    if sd <= 0:
        return 0.0
    return float((float(np.asarray(volume, float)[i]) - mu) / sd)


def rel_volume(volume, i: int, window_n: int) -> float:
    """Bar `i` volume ÷ average of the prior `window_n` bars (excludes today).
    1.0 = average; 2.0 = twice normal. NaN if unavailable."""
    if i - 1 < 0:
        return float("nan")
    ref = window(volume, i - 1, window_n)
    ref = ref[np.isfinite(ref)]
    if ref.size < max(5, window_n // 2):
        return float("nan")
    mu = float(np.mean(ref))
    if mu <= 0:
        return float("nan")
    return float(float(np.asarray(volume, float)[i]) / mu)


def volume_dryup(volume, i: int, base_n: int, ref_n: int) -> float:
    """Volume contraction inside a base: mean volume over the last `base_n` bars ÷
    mean volume over the `ref_n` bars before the base. <1 = dry-up (constructive).
    NaN if unavailable."""
    if i - base_n - ref_n + 1 < 0:
        return float("nan")
    base = window(volume, i, base_n)
    ref = np.asarray(volume, float)[i - base_n - ref_n + 1: i - base_n + 1]
    base = base[np.isfinite(base)]; ref = ref[np.isfinite(ref)]
    if base.size == 0 or ref.size == 0:
        return float("nan")
    mb = float(np.mean(base)); mr = float(np.mean(ref))
    if mr <= 0:
        return float("nan")
    return float(mb / mr)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-sectional relative strength (point-in-time, pure)
# ══════════════════════════════════════════════════════════════════════════════

def rel_strength_vs_benchmark(close, bench_close, i: int, lookback: int) -> float:
    """Stock return minus benchmark return over `lookback` bars ending at `i`, in
    percentage points. `close` and `bench_close` MUST be aligned to the same bar
    index. Positive = leadership. Pure & PIT (no live fetch, unlike
    scan/relative_strength.py). NaN if unavailable."""
    r_s = ret_pct(close, i, lookback)
    r_b = ret_pct(bench_close, i, lookback)
    if not np.isfinite(r_s) or not np.isfinite(r_b):
        return float("nan")
    return float(r_s - r_b)


def percentile_rank(value: float, population) -> float:
    """Percentile (0–100) of `value` within `population` (rank method). Used for
    cross-sectional RS percentile at a single as-of date — the caller must pass a
    population measured at THAT date (point-in-time). NaN if empty."""
    pop = np.asarray([p for p in population if p is not None and np.isfinite(p)],
                     dtype=float)
    if pop.size == 0 or not np.isfinite(value):
        return float("nan")
    return float((np.sum(pop < value) + 0.5 * np.sum(pop == value)) / pop.size * 100.0)
