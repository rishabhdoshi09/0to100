"""Cheap, durable "what did the tournament conclude about this symbol?"
lookup -- written once per real tournament cycle (product.evolution.
autonomy_hook, right after a production run_tournament_cycle() call), read
by Home/recommendations enrichment (section 17/42/45).

Read-only for consumers and purely additive: Home must still decide
eligibility from the Champion's real decision alone (product.paper_autopilot).
This board can EXPLAIN that decision (policy consensus, dominant dissent)
but must never be consulted to override it -- there is deliberately no
function here that returns anything resembling an eligibility verdict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_dir


def board_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_CONSENSUS_BOARD")
    if override:
        return Path(override)
    return logs_dir() / "product" / "evolution_consensus_board.json"


def save_latest_consensus(
    tournament_result: Mapping[str, Any], *, path: str | Path | None = None,
) -> None:
    """Overwrite each symbol's entry with its LATEST tournament read -- this
    is a live snapshot of "right now", not a history (that lives in the
    shadow-decision ledger itself)."""
    target = board_path(path)
    try:
        board = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(board, dict):
            board = {}
    except Exception:
        board = {}

    champion_policy_id = str(tournament_result.get("champion_policy_id") or "")
    as_of = str(tournament_result.get("as_of") or "")
    for row in tournament_result.get("results") or []:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        consensus = row.get("consensus") or {}
        board[symbol] = {
            "symbol": symbol,
            "market_snapshot_id": row.get("market_snapshot_id"),
            "champion_policy_id": champion_policy_id,
            "champion_decision": (row.get("champion") or {}).get("decision"),
            "consensus_pct": consensus.get("consensus_pct"),
            "qualified_count": consensus.get("qualified_count"),
            "selecting_count": consensus.get("selecting_count"),
            "main_dissent_reason": consensus.get("main_dissent_reason"),
            "as_of": as_of,
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(board, indent=2, default=str, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def get_consensus(symbol: str, *, path: str | Path | None = None) -> dict[str, Any] | None:
    """Read-only. None if the symbol was never part of a tournament cycle
    yet (e.g. right after a fresh install, before the first PAPER_CYCLE) --
    callers must treat that as "no research context available", never as a
    negative signal."""
    try:
        board = json.loads(board_path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(board, dict):
        return None
    return board.get(symbol.upper())
