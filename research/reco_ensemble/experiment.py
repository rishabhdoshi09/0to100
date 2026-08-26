"""EXP-RECO-001 — historical comparison of recommendation experts.

Compares SEPA-only, the previous Reco overlay, cross-sectional momentum,
momentum+quality, earnings momentum, breakout+sector, and the multi-expert
ensemble on the **same** universe, timestamps, costs, and horizon.

This module does not promote the ensemble because it sounds better.
If point-in-time bhavcopy / universe history is missing, the result is
INCONCLUSIVE / BLOCKED — never a toy backtest with invented prices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "EXP-RECO-001"
EXPERIMENT_TITLE = "Recommendation mixture-of-experts vs single-method baselines"

VARIANTS = (
    "A_sepa_only",
    "B_previous_reco_overlay",
    "C_xs_momentum",
    "D_momentum_quality",
    "E_earnings_momentum",
    "F_breakout_sector",
    "G_multi_expert_ensemble",
)

SUCCESS_CRITERIA = {
    "same_universe": True,
    "same_timestamps": True,
    "same_costs": True,
    "same_horizon": True,
    "min_trades": 30,
    "out_of_sample": True,
    "do_not_promote_on_in_sample_auc": True,
}


def _bhav_ready() -> tuple[bool, str]:
    store = ROOT / "data" / "bhavcopy_store"
    if not store.exists():
        return False, "data/bhavcopy_store is missing — no official NSE history on this machine."
    csvs = list(store.glob("**/*.csv"))
    if len(csvs) < 60:
        return False, (
            f"data/bhavcopy_store has {len(csvs)} CSV file(s); "
            "walk-forward comparison needs a multi-year official bhavcopy archive."
        )
    universe = ROOT / "data" / "universe_history.json"
    if not universe.exists():
        alt = ROOT / "logs" / "research" / "universe_history.json"
        if not alt.exists():
            return False, "universe_history.json is missing — cannot rebuild a point-in-time NSE universe."
    return True, "bhavcopy and universe history are present"


def experiment_spec() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "title": EXPERIMENT_TITLE,
        "hypothesis": (
            "A mixture of independent recommendation experts, ranked by evidence-family "
            "agreement rather than a weighted indicator soup, produces higher "
            "out-of-sample expectancy after costs than SEPA-only or the previous "
            "nine-method overlay, on the same NSE universe and holding horizon."
        ),
        "variants": list(VARIANTS),
        "success_criteria": dict(SUCCESS_CRITERIA),
        "comparisons": [
            "SEPA overlay shortlist only",
            "Previous Reco Buy rule (two method passes + weighted composite)",
            "Pure cross-sectional 5d/6m/12m momentum (persisted fields only)",
            "Momentum + quality class with ≥50% coverage",
            "Earnings momentum (sequential prints only — no fabricated consensus)",
            "Breakout + sector leadership",
            "Multi-expert ensemble with family agreement",
        ],
        "costs": "Same as research.momentum_breakout config (round-trip + slippage)",
        "integrity": [
            "point-in-time features only",
            "no future fundamentals",
            "no survivorship-blind constituents",
            "empty high-conviction days remain empty",
        ],
        "no_post_result_optimisation": True,
        "llm_not_in_money_path": True,
    }


def run_comparison(*, force: bool = False) -> dict[str, Any]:
    """Evaluate variants or honestly refuse when history is not on disk."""
    ready, reason = _bhav_ready()
    spec = experiment_spec()
    if not ready and not force:
        return {
            "experiment_id": EXPERIMENT_ID,
            "verdict": "INCONCLUSIVE",
            "blocked": True,
            "reason": reason,
            "spec": spec,
            "variants": {name: {"status": "not_run"} for name in VARIANTS},
            "note": (
                "The ensemble is not promoted. A missing archive is not evidence "
                "that the mixture of experts is better or worse than SEPA."
            ),
        }
    # A full walk-forward lives behind the existing momentum-breakout PIT harness.
    # This phase registers the experiment and refuses to fake a result.
    return {
        "experiment_id": EXPERIMENT_ID,
        "verdict": "INCONCLUSIVE",
        "blocked": True,
        "reason": (
            "Walk-forward comparison is registered but not executed in this phase. "
            "Existing research.momentum_breakout PIT harness must be reused; "
            "this function does not invent a parallel backtester."
        ),
        "spec": spec,
        "data_ready": ready,
        "data_note": reason,
        "variants": {name: {"status": "registered_not_run"} for name in VARIANTS},
        "note": "Do not treat registration as a performance claim.",
    }
