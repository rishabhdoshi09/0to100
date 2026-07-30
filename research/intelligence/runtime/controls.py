"""
🎛️ Owner controls (Phase P) — manual interventions that ALWAYS produce canonical events.

Routine paper trading is automatic; these are the owner's override handles. Each one is an
audited action: it records a CanonicalEvent with actor="user" so every manual intervention is
in the same append-only trail as the automatic decisions. None of these can reach live.
"""
from __future__ import annotations

from research.intelligence.runtime import events as EV
from research.intelligence.runtime import modes as MODES


def _audit(store, action: str, *, strategy_id="", symbol="", reason="", summary=None):
    return EV.emit(store, cycle_id="manual", event_type=f"MANUAL_{action}", actor="user",
                   strategy_id=strategy_id, symbol=symbol, reason=reason, summary=summary or {})


def set_mode(store, runtime_state, mode: str) -> str:
    if not MODES.is_valid(mode):
        raise ValueError(f"unknown mode {mode!r}")
    MODES.assert_no_live(mode)                       # owner cannot flip to live here
    _audit(store, "SET_MODE", reason=mode)
    return mode


def pause_all_entries(store) -> None:
    _audit(store, "PAUSE_ALL_ENTRIES")


def resume_entries(store) -> None:
    _audit(store, "RESUME_ENTRIES")


def pause_strategy(store, runtime_state, strategy_id: str, reason="owner pause") -> None:
    runtime_state.get(strategy_id).pause_reason = reason
    _audit(store, "PAUSE_STRATEGY", strategy_id=strategy_id, reason=reason)


def retire_strategy(store, runtime_state, strategy_id: str, reason="owner retire") -> None:
    st = runtime_state.get(strategy_id)
    st.retire_reason = reason; st.lifecycle = "RETIRED"
    _audit(store, "RETIRE_STRATEGY", strategy_id=strategy_id, reason=reason)


def block_symbol(store, blocklist: set, symbol: str) -> None:
    blocklist.add(symbol.upper())
    _audit(store, "BLOCK_SYMBOL", symbol=symbol.upper())


def block_sector(store, blocklist: set, sector: str) -> None:
    blocklist.add(sector)
    _audit(store, "BLOCK_SECTOR", reason=sector)


def close_position(store, book, strategy_id: str, symbol: str, date: str) -> bool:
    """Force-close one paper position at its last close (owner action)."""
    key = (strategy_id, symbol)
    pos = book.open.get(key)
    if pos is None:
        return False
    t = book._close(key, pos, pos.entry_price, "OWNER_CLOSE", date)
    _audit(store, "CLOSE_POSITION", strategy_id=strategy_id, symbol=symbol,
           summary={"realized_R": round(t.realized_R, 4)})
    return True


def close_all(store, book, date: str) -> int:
    n = 0
    for (sid, sym) in list(book.open.keys()):
        n += int(close_position(store, book, sid, sym, date))
    _audit(store, "CLOSE_ALL", summary={"closed": n})
    return n
