"""Which markets QuantTerm may open NEW paper risk in (platform authority).

QuantTerm is India-first: NSE equities and NSE F&O. The US virtual-paper
lane is paused for new entries unless the operator explicitly opts back in.

    QT_MARKET_SCOPE=INDIA_ONLY      (default) no new US paper entries
    QT_MARKET_SCOPE=INDIA_AND_US    US paper lane may open new positions

Pausing never deletes or rewrites US state: existing US paper positions keep
being supervised to their stops/targets/time-stops and settled, so their
records stay true. An unrecognised value fails closed to INDIA_ONLY, because
the safe reading of an unclear instruction is "do not open new risk".

Every consumer asks this module; none re-reads the environment itself.
"""
from __future__ import annotations

import os

ENV_VAR = "QT_MARKET_SCOPE"
INDIA_ONLY = "INDIA_ONLY"
INDIA_AND_US = "INDIA_AND_US"
PAUSE_REASON = "INDIA_ONLY_MODE"


def _raw() -> str:
    return os.environ.get(ENV_VAR, "").strip().upper()


def market_scope() -> str:
    raw = _raw()
    return INDIA_AND_US if raw == INDIA_AND_US else INDIA_ONLY


def us_new_entries_allowed() -> bool:
    return market_scope() == INDIA_AND_US


def scope_status() -> dict:
    raw = _raw()
    scope = market_scope()
    return {
        "market_scope": scope,
        "us_new_entries_allowed": scope == INDIA_AND_US,
        "us_existing_positions_supervised": True,
        "reason": "" if scope == INDIA_AND_US else PAUSE_REASON,
        "configured_value_invalid": bool(raw) and raw not in {INDIA_ONLY, INDIA_AND_US},
    }
