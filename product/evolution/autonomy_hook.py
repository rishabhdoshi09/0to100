"""Production wiring: run the tournament alongside the REAL paper cycle.

Compatibility hook for callers outside the canonical PAPER path. The canonical
product.paper_autopilot.run_reco_paper_cycle() now runs the tournament itself
BEFORE any PAPER mutation so Champion and Challengers see the same account
state. This module never re-scans the market and never influences execution --
it only re-reads the same already-persisted recommendations payload
(a cheap JSON read, not a re-scan) to freeze the Champion's real decisions
and bounded Challenger shadows for later grading.

Entirely best-effort by design (section 32, failure isolation): any failure
here is caught and logged, never allowed to raise into or affect the real
paper-cycle result that already happened before this runs.
"""
from __future__ import annotations

from typing import Any, Mapping

from logger import get_logger

log = get_logger(__name__)

CHAMPION_BOOTSTRAP_ID = "EQUITY_CHAMPION_BASELINE_V1"


def _ensure_champion_exists(domain: str) -> str:
    from product.evolution import policy_registry as PR

    population = PR.ensure_seed_population(domain)
    return str((population.get("champion") or {}).get("policy_id") or CHAMPION_BOOTSTRAP_ID)


def run_tournament_for_reco_cycle(
    reco: Mapping[str, Any], *, book: Any = None, regime: str = "", as_of: str = "",
) -> dict[str, Any] | None:
    """Best-effort: freeze the Champion's real decisions + bounded Challenger
    shadows for this cycle's candidates. Returns None (and logs) on any
    failure -- never raises into the real paper-cycle caller."""
    try:
        from product.autopilot_journal import flatten_cards
        from product.evolution import policy_registry as PR
        from product.evolution import tournament as T
        from product.recommendations_store import load_recommendations

        domain = PR.EQUITY
        champion_policy_id = _ensure_champion_exists(domain)

        payload = load_recommendations() or {}
        card_list = flatten_cards(payload)
        if not card_list:
            return None

        champion_decisions_by_symbol: dict[str, dict[str, Any]] = {}
        for row in [
            *(reco.get("taken") or []), *(reco.get("rejections") or []), *(reco.get("waits") or []),
        ]:
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                champion_decisions_by_symbol[symbol] = row

        result = T.run_tournament_cycle(
            card_list, champion_decisions_by_symbol,
            champion_policy_id=champion_policy_id, domain=domain,
            book=book, regime=regime, as_of=as_of,
        )
        try:
            from product.evolution.consensus_board import save_latest_consensus
            save_latest_consensus(result)
        except Exception as exc:
            log.debug("evolution_consensus_board_save_failed", error=str(exc))
        return result
    except Exception as exc:
        log.warning("evolution_tournament_cycle_failed", error=str(exc))
        return None
