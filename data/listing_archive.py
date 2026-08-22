"""Listing / delisting honesty layer.

Local bhav first/last appearance is not an official listing archive.
`point_in_time_universe_v2` is **not** created — evidence is not strong enough.
"""
from __future__ import annotations

from typing import Any

from data.universe_history import history_path, ledger_status


def universe_pit_class(path=None) -> dict[str, Any]:
    st = ledger_status(path)
    src = str(st.get("source") or "")
    official = bool(st.get("research_grade"))
    return {
        "path": str(history_path(path)),
        "rows": st.get("rows"),
        "source": src,
        "research_grade": official,
        "pit_class": "PIT_STRONG" if official else "PIT_DEGRADED",
        "point_in_time_universe_v2": False,
        "status": "RESEARCH_READY" if official else "DESCRIPTIVE_ONLY",
        "note": (
            "v2 withheld. Inferred first/last bhav session is not an official "
            "listing/delisting/suspension/symbol-change archive. Membership "
            "remain PIT_DEGRADED until an NSE/BSE listing file is ingested."
        ),
        **{k: st.get(k) for k in ("date_range", "n_delisted", "generated_at") if k in st},
    }


def is_investable(symbol: str, as_of, *, path=None, max_stale_sessions: int = 40) -> dict[str, Any]:
    """Stale / delisted names must not remain investable indefinitely."""
    import pandas as pd
    from data.nse_universe import point_in_time_universe

    pit = point_in_time_universe(as_of, path=path)
    sym = str(symbol).upper()
    members = set(pit.get("symbols") or [])
    listed = sym in members
    return {
        "symbol": sym,
        "as_of": str(pd.Timestamp(as_of).date()),
        "in_universe": listed,
        "survivorship_complete": bool(pit.get("survivorship_complete")),
        "research_grade": bool(pit.get("research_grade")),
        "pit_class": "PIT_DEGRADED" if not pit.get("research_grade") else "PIT_STRONG",
        "max_stale_sessions_policy": max_stale_sessions,
        "note": pit.get("note"),
    }
