"""Persistent SEPA base lifecycle — one economic setup, not one rolling window."""
from __future__ import annotations

from typing import Any

from research.sepa.setups import setup_id as hash_setup_id


CORE_TERMINAL = frozenset({
    "FILLED", "GAP_THROUGH", "MISSED", "EXTENDED", "INVALIDATED", "FAILED", "EXPIRED",
})
RESEARCH_ONLY = frozenset({"PIVOT_RETEST", "LEFT_CENSORED"})


def _dateset(vcp: dict[str, Any]) -> set[str]:
    highs = []
    ev = vcp.get("evidence") or {}
    if vcp.get("base_start_date"):
        highs.append(str(vcp.get("base_start_date")))
    last = ev.get("last_contraction_high_date") or ev.get("active_last_high_date")
    if last:
        highs.append(str(last))
    for d in vcp.get("dates") or []:
        highs.append(str(d))
    return {d for d in highs if d}


class PersistentSetupLedger:
    """One ID per continuing base even when the 120-bar lookback rolls.

    A new ID requires a structural reset, not the first contraction ageing
    out of a rolling array.
    """

    def __init__(self, *, versions: dict[str, str] | None = None) -> None:
        self.versions = dict(versions or {})
        self._open: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []

    def _new_row(self, symbol: str, vcp: dict[str, Any], as_of: str) -> dict[str, Any]:
        origin = str(vcp.get("base_start_date") or as_of)
        sid = hash_setup_id(
            symbol, origin,
            eligibility_version=self.versions.get("eligibility_version") or "",
            vcp_version=self.versions.get("vcp_version") or "",
            pivot_version=self.versions.get("pivot_version") or "",
        )
        return {
            "setup_id": sid,
            "symbol": str(symbol).upper(),
            "original_base_start": origin,
            "first_knowable_date": vcp.get("vcp_knowable_date") or as_of,
            "contraction_dates": sorted(_dateset(vcp)),
            "pivot": vcp.get("pivot"),
            "pivot_revisions": [],
            "stop": vcp.get("stop"),
            "state": vcp.get("state") or "FORMING",
            "status": "FORMING",
            "terminal_reason": None,
            "left_censored": False,
            "saw_entry_ready": False,
            "saw_forming": False,
            "first_seen_as_of": as_of,
            "first_seen_state": vcp.get("state"),
        }

    def observe(
        self,
        *,
        symbol: str,
        vcp: dict[str, Any],
        as_of: str,
        evaluation_start: str | None = None,
        price: float | None = None,
        zone_hi: float | None = None,
        in_eval_window: bool = True,
    ) -> dict[str, Any] | None:
        if not vcp.get("base_start_date") and not vcp.get("pivot"):
            return None
        if not vcp.get("pivot") and int(vcp.get("contraction_count") or 0) < 1:
            return None
        if not vcp.get("base_start_date"):
            return None
        sym = str(symbol).upper()
        cur_dates = _dateset(vcp)
        last_high = str((vcp.get("evidence") or {}).get("last_contraction_high_date") or "")
        open_row = self._open.get(sym)

        continuing = False
        if open_row is not None and open_row.get("status") not in CORE_TERMINAL:
            prev = set(open_row.get("contraction_dates") or [])
            if prev & cur_dates or (last_high and last_high in prev):
                continuing = True
            elif open_row.get("original_base_start") and open_row["original_base_start"] in cur_dates:
                continuing = True
            elif open_row.get("status") == "FORMING" and vcp.get("state") in {
                "PIVOT_DEFINED", "ENTRY_READY", "VCP_FORMING", "CONTRACTION_2", "BASE_FORMING",
            }:
                # Rolling window dropped the first high but the live coil continues.
                continuing = True

        if open_row is not None and open_row.get("status") in CORE_TERMINAL:
            # After a canonical breakout/fail, a return to the zone is a retest.
            if vcp.get("state") == "ENTRY_READY" and open_row.get("status") == "EXTENDED":
                rec = dict(open_row)
                rec["status"] = "PIVOT_RETEST"
                rec["state"] = "PIVOT_RETEST"
                rec["as_of"] = as_of
                rec["core_sepa_entry"] = False
                return rec
            continuing = False

        if continuing and open_row is not None:
            row = open_row
            row["contraction_dates"] = sorted(set(row.get("contraction_dates") or []) | cur_dates)
            if vcp.get("pivot") is not None and vcp.get("pivot") != row.get("pivot"):
                row.setdefault("pivot_revisions", []).append({
                    "as_of": as_of, "from": row.get("pivot"), "to": vcp.get("pivot"),
                })
            row["pivot"] = vcp.get("pivot") if vcp.get("pivot") is not None else row.get("pivot")
            row["stop"] = vcp.get("stop") if vcp.get("stop") is not None else row.get("stop")
            row["state"] = vcp.get("state") or row.get("state")
        else:
            row = self._new_row(sym, vcp, as_of)
            self._open[sym] = row
            self._history.append(row)

        state = str(vcp.get("state") or "")
        if state in {"CONTRACTION_1", "CONTRACTION_2", "BASE_FORMING", "VCP_FORMING", "PIVOT_DEFINED"}:
            row["saw_forming"] = True
        if state == "ENTRY_READY":
            row["saw_entry_ready"] = True

        if in_eval_window and not row.get("left_censored"):
            first = row.get("first_seen_as_of")
            at_boundary = (
                evaluation_start
                and first
                and str(first) <= str(evaluation_start)
                and as_of == first
            )
            already_through = state in {"EXTENDED", "BROKEN_OUT"} or (
                zone_hi is not None and price is not None and float(price) > float(zone_hi)
            )
            if (at_boundary or row.get("first_seen_as_of") == as_of) and already_through and not row.get("saw_entry_ready"):
                row["left_censored"] = True
                row["status"] = "LEFT_CENSORED"
                row["terminal_reason"] = "LEFT_CENSORED"

        row["as_of"] = as_of
        row["window_base_start"] = vcp.get("base_start_date")
        return row

    def mark(self, symbol: str, status: str, *, reason: str | None = None) -> None:
        row = self._open.get(str(symbol).upper())
        if row is None:
            return
        if row.get("left_censored") and status in CORE_TERMINAL:
            # Left-censored observations are not opportunities.
            row["status"] = "LEFT_CENSORED"
            row["terminal_reason"] = row.get("terminal_reason") or "LEFT_CENSORED"
            return
        if status == "PIVOT_RETEST":
            row["status"] = "PIVOT_RETEST"
            row["core_sepa_entry"] = False
            return
        row["status"] = status
        row["terminal_reason"] = reason or status
        if status in CORE_TERMINAL:
            row["closed"] = True

    def is_core_opportunity(self, symbol: str) -> bool:
        row = self._open.get(str(symbol).upper())
        if row is None:
            return False
        if row.get("left_censored"):
            return False
        if row.get("status") in {"PIVOT_RETEST", "LEFT_CENSORED"}:
            return False
        if row.get("status") in CORE_TERMINAL:
            return False
        return True

    def get(self, symbol: str) -> dict[str, Any] | None:
        return self._open.get(str(symbol).upper())

    def all_rows(self) -> list[dict[str, Any]]:
        return list(self._history)
