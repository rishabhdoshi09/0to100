"""Unique SEPA setup identity and lifecycle. One VCP → one observation."""
from __future__ import annotations

import hashlib
from typing import Any


TERMINAL = frozenset({
    "FILLED", "GAP_THROUGH", "MISSED", "EXTENDED", "INVALIDATED", "FAILED", "EXPIRED",
})


def setup_id(
    symbol: str,
    base_start_date: str | None,
    *,
    eligibility_version: str,
    vcp_version: str,
    pivot_version: str,
) -> str:
    """Stable ID frozen at first structural detection of a base.

    Pivot *price* is deliberately excluded so daily noise / extra confirmed
    contractions on the same base do not mint a new observation.
    """
    raw = "|".join([
        str(symbol or "").upper().strip(),
        str(base_start_date or ""),
        str(eligibility_version or ""),
        str(vcp_version or ""),
        str(pivot_version or ""),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def setup_key(symbol: str, base_start_date: str | None) -> tuple[str, str]:
    return (str(symbol or "").upper().strip(), str(base_start_date or ""))


class SetupRegistry:
    """Deduplicate daily scans: one lifecycle per (symbol, base_start)."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], dict[str, Any]] = {}
        self._open_symbol: dict[str, str] = {}

    def see(self, *, symbol: str, vcp: dict[str, Any], versions: dict[str, str]) -> dict[str, Any] | None:
        base = vcp.get("base_start_date")
        if not base or not vcp.get("pivot"):
            return None
        key = setup_key(symbol, str(base))
        sid = setup_id(
            symbol, str(base),
            eligibility_version=versions.get("eligibility_version") or "",
            vcp_version=versions.get("vcp_version") or "",
            pivot_version=versions.get("pivot_version") or "",
        )
        row = self._by_key.get(key)
        if row is None:
            row = {
                "setup_id": sid,
                "symbol": str(symbol).upper(),
                "base_start_date": str(base),
                "first_detected_date": None,
                "pivot_knowable_date": vcp.get("pivot_knowable_date"),
                "vcp_knowable_date": vcp.get("vcp_knowable_date"),
                "status": "FORMING",
                "pivot": vcp.get("pivot"),
                "stop": vcp.get("stop"),
            }
            self._by_key[key] = row
        if vcp.get("detected") and row["first_detected_date"] is None:
            row["first_detected_date"] = vcp.get("vcp_knowable_date") or versions.get("as_of")
            if row["status"] == "FORMING":
                row["status"] = "DETECTED"
        if vcp.get("pivot") is not None:
            row["pivot"] = vcp.get("pivot")
            row["stop"] = vcp.get("stop")
        return row

    def mark(self, symbol: str, base_start_date: str | None, status: str) -> None:
        key = setup_key(symbol, base_start_date)
        row = self._by_key.get(key)
        if row is None:
            return
        row["status"] = status
        if status in TERMINAL:
            self._open_symbol.pop(str(symbol).upper(), None)
        else:
            self._open_symbol[str(symbol).upper()] = row["setup_id"]

    def is_terminal(self, symbol: str, base_start_date: str | None) -> bool:
        row = self._by_key.get(setup_key(symbol, base_start_date))
        return bool(row and row.get("status") in TERMINAL)

    def get(self, symbol: str, base_start_date: str | None) -> dict[str, Any] | None:
        return self._by_key.get(setup_key(symbol, base_start_date))

    def all_rows(self) -> list[dict[str, Any]]:
        return list(self._by_key.values())
