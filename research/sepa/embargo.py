"""Exchange-session embargo — never convert hold sessions to calendar days."""
from __future__ import annotations

from typing import Any

import pandas as pd

from research.sepa.frames import iso_date


def attach_session_path(sim: dict[str, Any] | None, fwd: pd.DataFrame | None, *, as_of) -> dict[str, Any] | None:
    """Stamp entry/exit exchange sessions onto a simulator result."""
    if sim is None or fwd is None or len(fwd) == 0:
        return sim
    hold = int(sim.get("hold_sessions") or sim.get("hold") or 1)
    hold = max(1, hold)
    entry_idx = int(sim.get("entry_index") or 0)
    exit_idx = int(sim.get("exit_index") if sim.get("exit_index") is not None else hold - 1)
    entry_idx = max(0, min(entry_idx, len(fwd) - 1))
    exit_idx = max(entry_idx, min(exit_idx, len(fwd) - 1))
    sim["entry_index"] = entry_idx
    sim["exit_index"] = exit_idx
    sim["hold_sessions"] = int(exit_idx - entry_idx + 1)
    sim["entry_date"] = iso_date(fwd.index[entry_idx])
    sim["exit_date"] = iso_date(fwd.index[exit_idx])
    sim["signal_date"] = iso_date(as_of)
    return sim


def calendar_day_embargo_until(as_of, hold_sessions: int) -> str:
    """The **incorrect** approximation: as_of + Timedelta(days=hold). Tests only."""
    hold = max(1, int(hold_sessions))
    return str((pd.Timestamp(iso_date(as_of)) + pd.Timedelta(days=hold)).date())


def session_embargo_blocks(*, as_of, last_exit_session: str | None) -> bool:
    """True while a position is still open through ``last_exit_session``."""
    if not last_exit_session:
        return False
    return iso_date(as_of) <= str(last_exit_session)


def embargo_lifts_after(exit_date) -> str:
    """First as_of that may take a new signal is the next session after exit."""
    return iso_date(exit_date)
