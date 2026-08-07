"""
🛡️ Point-in-time safety — the temporal firewall.

The framework distinguishes six clocks:
  market_ts        — the bar's own time
  signal_ts        — when the signal became known (the breakout close)
  data_avail_ts    — when the data used was actually available
  ingestion_ts     — when we stored it
  entry_ts         — when a trade could first be entered (next tradable bar)
  fundamental_ts   — when a financial report became public

A study is only honest if information never travels backwards across these. This
module FAILS CLOSED: a proven temporal violation voids the observation (raises /
returns a violation), it is never silently tolerated as an optimistic result.

Structural guarantees already enforced elsewhere (documented here so the contract
is legible): base features read only [base_start, base_end=i-1]; the pivot is the
base's max high strictly before the breakout bar; entry is the NEXT bar's price,
never the signal bar's close. This module adds the checkable assertions and the
event-deduplication registry.
"""
from __future__ import annotations

from dataclasses import dataclass


class PITViolation(Exception):
    """A proven point-in-time violation. Fail closed."""


@dataclass
class PITCheck:
    ok: bool
    violations: tuple = ()

    def raise_if_bad(self) -> "PITCheck":
        if not self.ok:
            raise PITViolation("; ".join(self.violations))
        return self


def _le(a, b) -> bool:
    """String/ISO-date safe ≤ (dates compare lexicographically when ISO)."""
    return str(a) <= str(b)


def check_timestamps(*, market_ts, signal_ts, data_avail_ts, entry_ts,
                     valuation_ts=None) -> PITCheck:
    """The core clock ordering. Fail closed on any inversion:
      data_avail_ts ≤ signal_ts   (can't use data before it existed)
      signal_ts     ≤ entry_ts    (can't enter before the signal is known)
      signal_ts     == market_ts  (the signal is dated to its bar)
      valuation_ts  ≤ signal_ts   (fundamentals never used before publication)
    """
    v = []
    if not _le(data_avail_ts, signal_ts):
        v.append(f"data available {data_avail_ts} AFTER signal {signal_ts}")
    if not _le(signal_ts, entry_ts):
        v.append(f"signal {signal_ts} AFTER entry {entry_ts}")
    if str(market_ts) != str(signal_ts):
        v.append(f"signal {signal_ts} not dated to its market bar {market_ts}")
    if valuation_ts is not None and not _le(valuation_ts, signal_ts):
        v.append(f"valuation dated {valuation_ts} AFTER signal {signal_ts} (future fundamental)")
    return PITCheck(ok=not v, violations=tuple(v))


def check_pivot_pre_existing(pivot: float, base_high: float,
                             breakout_high: float) -> PITCheck:
    """The pivot must be the base's PRE-EXISTING resistance — derived from base
    bars, not from the breakout bar's own (future-relative-to-the-base) high."""
    v = []
    if pivot is None or base_high is None:
        return PITCheck(ok=True)             # nothing to contradict
    if abs(float(pivot) - float(base_high)) > 1e-9 and float(pivot) > float(base_high):
        v.append(f"pivot {pivot} exceeds base high {base_high} — built from a future bar")
    return PITCheck(ok=not v, violations=tuple(v))


def check_entry_not_signal_bar(entry_index: int, signal_index: int) -> PITCheck:
    """A historical trade must not assume execution at the signal bar's close.
    Entry index must be strictly AFTER the signal (breakout) bar."""
    if entry_index <= signal_index:
        return PITCheck(ok=False,
                        violations=(f"entry bar {entry_index} not after signal bar "
                                    f"{signal_index} — same-day-close leakage",))
    return PITCheck(ok=True)


def check_stop_below_entry(stop: float, entry_ref: float) -> PITCheck:
    if stop is None:
        return PITCheck(ok=False, violations=("no structural stop selected",))
    if not (0 < float(stop) < float(entry_ref)):
        return PITCheck(ok=False,
                        violations=(f"stop {stop} not strictly below entry {entry_ref}",))
    return PITCheck(ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Event deduplication — one breakout event → one observation
# ══════════════════════════════════════════════════════════════════════════════

class EventRegistry:
    """Deduplicates breakout events across a scan. Keyed by the observation's
    canonical `event_id()` (symbol+base+pivot+breakout-day+detector/config). A stock
    can requalify only after a documented reset AND a genuinely new base/pivot — the
    detector enforces the cooldown; this registry catches any residual duplicate.

    NOTE: this is a RESEARCH construct. It has NOTHING to do with paper/live journals
    or their own de-duplication — candidate-event identity is never derived from an
    order book."""

    def __init__(self):
        self._seen: set[str] = set()

    def is_new(self, event_id: str) -> bool:
        return event_id not in self._seen

    def register(self, event_id: str) -> bool:
        """Return True if newly registered, False if it was a duplicate."""
        if event_id in self._seen:
            return False
        self._seen.add(event_id)
        return True

    def __len__(self) -> int:
        return len(self._seen)
