"""D5–D6 / D8–D10 readiness assessments (no alpha experiments)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.data_expansion.classify import WINDOW_END, WINDOW_START
from research.phase_a5.scoped_certification import FROZEN_PANEL

REPO_ROOT = Path(__file__).resolve().parents[2]


def assess_sector_history() -> dict[str, Any]:
    """D5 — PIT sector membership readiness."""
    static_map = REPO_ROOT / "logs" / "phase_a5" / "sector_map.json"
    has_static = static_map.exists()
    # No dated sector membership ledger in-repo
    dated = list((REPO_ROOT / "logs").glob("*sector*histor*")) + list(
        (REPO_ROOT / "logs").glob("*pit*sector*")
    )
    return {
        "status": "NOT_RESEARCH_READY",
        "pit_sector_history": False,
        "static_sector_map_available": has_static,
        "static_map_path": str(static_map) if has_static else None,
        "dated_ledgers_found": [str(p) for p in dated],
        "plain": {
            "label": "Sector history",
            "explanation": (
                "QuantTerm knows today's sector labels for many stocks, but does "
                "not yet have a trustworthy year-by-year sector membership history."
            ),
            "implication": (
                "Price-only research can proceed. Tests that need historical "
                "sector-neutral portfolios stay blocked."
            ),
            "technical": "NOT_RESEARCH_READY",
        },
        "blocks_ohlcv_research": False,
    }


def assess_fundamentals_events() -> dict[str, Any]:
    """D6 — future datasets readiness (AVAILABLE_AT focus). No strategies built."""
    fund_cache = REPO_ROOT / "fundamentals"
    rows = [
        {
            "dataset": "financial_statements",
            "status": "OPERATIONAL_ONLY",
            "available_at": "MISSING",
            "note": "As-of-now caches / scrapes; no publication-dated ledger",
        },
        {
            "dataset": "profitability_metrics",
            "status": "OPERATIONAL_ONLY",
            "available_at": "MISSING",
            "note": "Derived from current fundamentals cache",
        },
        {
            "dataset": "valuation_multiples",
            "status": "OPERATIONAL_ONLY",
            "available_at": "MISSING",
            "note": "PitContract refuses current fundamentals for research",
        },
        {
            "dataset": "earnings_results",
            "status": "MISSING",
            "available_at": "MISSING",
            "note": "Need official results timestamps + AVAILABLE_AT",
        },
        {
            "dataset": "corporate_announcements",
            "status": "PARTIAL",
            "available_at": "PARTIAL",
            "note": "Some news/CA feeds exist; not a PIT announcement ledger",
        },
        {
            "dataset": "shareholding_ownership",
            "status": "OPERATIONAL_ONLY",
            "available_at": "MISSING",
            "note": "Current shareholding scrapes; no quarter AVAILABLE_AT history",
        },
        {
            "dataset": "earnings_dates",
            "status": "MISSING",
            "available_at": "MISSING",
            "note": "Calendar without verified first-available timestamps",
        },
        {
            "dataset": "reported_vs_available_timestamps",
            "status": "MISSING",
            "available_at": "MISSING",
            "note": "Core requirement for any fundamental/event factor research",
        },
    ]
    return {
        "key_requirement": "AVAILABLE_AT",
        "fundamentals_tree_present": fund_cache.exists(),
        "datasets": rows,
        "plain": {
            "label": "Company fundamentals & events",
            "explanation": (
                "QuantTerm can show some company numbers for today's screening, "
                "but it cannot yet prove when those numbers first became public "
                "in the past."
            ),
            "implication": (
                "Serious historical tests of value, quality, or earnings reactions "
                "must wait for dated 'available-at' records."
            ),
            "technical": "AVAILABLE_AT missing → OPERATIONAL_ONLY / MISSING",
        },
    }


def research_power(
    *,
    n_securities: int,
    n_sessions: int,
    security_sessions: int,
    date_start: str,
    date_end: str,
    prior_n: int = 29,
    prior_sessions: int = 764,
) -> dict[str, Any]:
    """D8 — quantify what the broader dataset enables (no experiments)."""
    # Approximate calendar years covered
    y0 = int(date_start[:4])
    y1 = int(date_end[:4])
    calendar_years = max(1, y1 - y0 + 1)
    security_years = security_sessions / 252.0 if security_sessions else 0.0
    prior_sec_years = (prior_n * prior_sessions) / 252.0
    sample_gain = (security_years / prior_sec_years) if prior_sec_years else None
    # Cross-sectional breadth ≈ median names (approx n_securities for liquid panel)
    cs_breadth = n_securities
    # Independent periods: ~ calendar years minus 1 (warmup) as rough research power
    independent_periods = max(1, calendar_years - 1)

    regimes = [
        {"period": "2020", "label": "COVID crash + recovery"},
        {"period": "2021", "label": "Post-stimulus / reopening equity strength"},
        {"period": "2022", "label": "Rate-hike / risk-off year"},
        {"period": "2023", "label": "Recovery / consolidation"},
        {"period": "2024", "label": "Election / mid-cycle tape"},
        {"period": "2025-2026", "label": "Recent live research window"},
    ]

    families = [
        {
            "family": "cross_sectional_price_factors_ohlcv",
            "class": "READY_TO_TEST",
            "note": "Low-vol / other price-only CS factors with broader panel+years",
        },
        {
            "family": "short_horizon_reversal",
            "class": "CLOSED_REJECTED",
            "note": "Do not reopen; data expansion ≠ retune",
        },
        {
            "family": "momentum_60d",
            "class": "CLOSED_REJECTED",
            "note": "Do not reopen",
        },
        {
            "family": "dynamic_market_structure",
            "class": "CLOSED_REJECTED",
            "note": "Do not reopen",
        },
        {
            "family": "standalone_network_alpha",
            "class": "CLOSED_REJECTED",
            "note": "Do not reopen",
        },
        {
            "family": "network_concentration_interaction",
            "class": "CLOSED_REJECTED",
            "note": "Do not reopen",
        },
        {
            "family": "logistic_challenger",
            "class": "CLOSED_REJECTED",
            "note": "Do not reopen",
        },
        {
            "family": "volatility_compression_risk",
            "class": "CLOSED_REJECTED",
            "note": "Do not reopen",
        },
        {
            "family": "value_quality_profitability",
            "class": "DATA_MISSING",
            "note": "Need PIT fundamentals with AVAILABLE_AT",
        },
        {
            "family": "post_earnings_drift_events",
            "class": "DATA_MISSING",
            "note": "Need earnings AVAILABLE_AT timestamps",
        },
        {
            "family": "sector_neutral_factors",
            "class": "PIT_UNSAFE",
            "note": "PIT sector history NOT_RESEARCH_READY",
        },
        {
            "family": "ownership_shareholding_effects",
            "class": "DATA_MISSING",
            "note": "No PIT ownership ledger",
        },
    ]

    return {
        "securities": n_securities,
        "sessions": n_sessions,
        "security_sessions": security_sessions,
        "security_years": round(security_years, 1),
        "prior_29_security_years": round(prior_sec_years, 1),
        "approx_sample_size_gain": round(sample_gain, 2) if sample_gain else None,
        "effective_cross_sectional_breadth": cs_breadth,
        "independent_calendar_years": calendar_years,
        "independent_time_periods_approx": independent_periods,
        "date_range": [date_start, date_end],
        "market_regimes_represented": regimes,
        "hypothesis_family_readiness": families,
    }


def low_vol_retest_readiness(
    *,
    n_securities: int,
    n_sessions: int,
    prior_rebalances: int = 13,
    prior29_full_window_blocked: list[str] | None = None,
    prior29_post_ca_sessions: int = 1056,
) -> dict[str, Any]:
    """D9 — readiness only; do NOT rerun EXP-NEXT-02."""
    # Preserve frozen protocol constants by reference
    from research.phase_next import protocol as P

    frozen = {
        "experiment": "EXP-NEXT-02",
        "status": "INCONCLUSIVE",
        "next_action": "HOLD_NO_TUNING",
        "hypothesis": "unchanged",
        "volatility_definition": f"realized_vol lookback={P.LOWVOL_LOOKBACK}",
        "hold": P.LOWVOL_HOLD,
        "rebalance": P.LOWVOL_REBALANCE,
        "universe_construction": (
            "FIXED_PREREGISTERED_29 — preserved for identical-protocol retest; "
            "expanded CERTIFIABLE panel is a separate broader scope requiring a "
            "new experiment id if used"
        ),
        "cost_assumptions": "CNC round_trip — unchanged",
        "metrics": "mean_net, DSR, discovery+confirm gates — unchanged",
        "significance_thresholds": "unchanged",
        "success_failure_criteria": "unchanged",
        "prior_snapshot": "a7a9828ec37e09e4",
        "prior_n_rebalances": prior_rebalances,
    }

    blocked = prior29_full_window_blocked or ["TATASTEEL", "BAJAJFINSV"]
    # Path A: identical 29-name universe on post-2022-09-15 CA-clean window
    approx_reb_29 = max(
        0, (prior29_post_ca_sessions - P.LOWVOL_LOOKBACK) // P.LOWVOL_REBALANCE
    )
    path_a_ready = approx_reb_29 >= 40  # material vs ~13

    # Path B: expanded certifiable liquid panel (broader universe → new registration)
    approx_reb = max(0, (n_sessions - P.LOWVOL_LOOKBACK) // P.LOWVOL_REBALANCE)
    cs_ok = n_securities >= 100
    time_ok = approx_reb >= 40
    path_b_ready = cs_ok and time_ok and n_sessions >= 1000

    ready = path_a_ready or path_b_ready
    verdict = "LOW_VOL_RETEST_READY" if ready else "LOW_VOL_STILL_TOO_THIN"
    return {
        "verdict": verdict,
        "frozen_protocol_preserved": True,
        "do_not_rerun_in_this_task": True,
        "paths": {
            "identical_29_post_2022_09_15": {
                "ready": path_a_ready,
                "approx_rebalances": approx_reb_29,
                "note": (
                    "All 29 are CA-clean after 2022-09-15. Full 2020-01-01 window "
                    f"still BLOCKED_CA for {blocked} until official CA factors are ingested."
                ),
            },
            "expanded_certifiable_panel": {
                "ready": path_b_ready,
                "approx_rebalances": approx_reb,
                "n_securities": n_securities,
                "note": "Broader universe — register as a new experiment id; do not silently retune EXP-NEXT-02.",
            },
        },
        "approx_rebalances_on_expanded_history": approx_reb,
        "cross_section_names": n_securities,
        "material_improvement_vs_prior": {
            "rebalance_count_gain_path_a": (
                approx_reb_29 / prior_rebalances if prior_rebalances else None
            ),
            "rebalance_count_gain_path_b": (
                approx_reb / prior_rebalances if prior_rebalances else None
            ),
            "cross_section_gain_path_b": n_securities / 29.0,
            "history_sessions_gain_path_b": n_sessions / 764.0 if n_sessions else None,
        },
        "gates": {
            "path_a_rebalances_ge_40": path_a_ready,
            "path_b_cross_section_ge_100": cs_ok,
            "path_b_approx_rebalances_ge_40": time_ok,
            "path_b_sessions_ge_1000": n_sessions >= 1000,
        },
        "frozen": frozen,
        "plain": {
            "label": "Low-volatility retest readiness",
            "explanation": (
                "The earlier low-volatility test did not have enough independent "
                "rebalances to decide. QuantTerm now has enough cleaner history to "
                "retest the same idea honestly — either on the original 29 names "
                "after mid-2022, or on a much larger certified stock group."
                if ready else
                "Even with more data, the certified surface is still too thin to "
                "retest low-volatility honestly."
            ),
            "implication": (
                "A retest is now scientifically allowed — but it has not been run yet. "
                "Do not change the frozen rules when you do run it."
                if ready else
                "Do not rerun yet; keep HOLD_NO_TUNING."
            ),
            "technical": verdict,
        },
    }


def future_research_families(
    *,
    n_certifiable: int,
    low_vol_verdict: str,
) -> list[dict[str, Any]]:
    """D10 — rank next potential families (no implementation)."""
    families = [
        {
            "family": "low_volatility_retest",
            "priority": 1 if low_vol_verdict == "LOW_VOL_RETEST_READY" else 3,
            "economic_rationale": (
                "Investors may prefer smoother stocks; a cost-aware long-short "
                "of low vs high realized vol is a classic risk-based premium test."
            ),
            "required_data": "PIT-safe adjusted OHLCV; fixed certified universe",
            "current_data_readiness": (
                "READY_TO_TEST" if low_vol_verdict == "LOW_VOL_RETEST_READY" else "STILL_TOO_THIN"
            ),
            "sample_potential": f"~{n_certifiable} names × multi-year sessions",
            "pit_risk": "LOW if scoped snapshot used",
            "cost_sensitivity": "HIGH (quintile turnover)",
            "priority_note": "First honest retest candidate; HOLD_NO_TUNING until run",
        },
        {
            "family": "value",
            "priority": 4,
            "economic_rationale": "Cheap vs expensive securities may earn a premium after costs.",
            "required_data": "Valuation multiples with AVAILABLE_AT timestamps",
            "current_data_readiness": "DATA_MISSING",
            "sample_potential": "Large once PIT ledger exists",
            "pit_risk": "HIGH today (as-of-now caches)",
            "cost_sensitivity": "MEDIUM",
            "priority_note": "Blocked on AVAILABLE_AT fundamentals",
        },
        {
            "family": "quality_profitability",
            "priority": 4,
            "economic_rationale": "High-quality / profitable firms may outperform low-quality peers.",
            "required_data": "PIT profitability & balance-sheet ratios + AVAILABLE_AT",
            "current_data_readiness": "DATA_MISSING",
            "sample_potential": "Large once PIT ledger exists",
            "pit_risk": "HIGH today",
            "cost_sensitivity": "MEDIUM",
            "priority_note": "Blocked on fundamentals ledger",
        },
        {
            "family": "earnings_growth",
            "priority": 5,
            "economic_rationale": "Markets may under-react to sustained earnings growth.",
            "required_data": "PIT earnings series with AVAILABLE_AT",
            "current_data_readiness": "DATA_MISSING",
            "sample_potential": "Event + CS hybrid",
            "pit_risk": "HIGH without timestamps",
            "cost_sensitivity": "MEDIUM",
            "priority_note": "Requires earnings AVAILABLE_AT",
        },
        {
            "family": "post_earnings_drift",
            "priority": 3,
            "economic_rationale": "Prices may drift after earnings surprises.",
            "required_data": "Earnings surprise + first-available timestamp + OHLCV",
            "current_data_readiness": "DATA_MISSING",
            "sample_potential": "Many events/year once ledger exists",
            "pit_risk": "CRITICAL without AVAILABLE_AT",
            "cost_sensitivity": "HIGH (event trading)",
            "priority_note": "Top data-investment after OHLCV breadth",
        },
        {
            "family": "event_reactions",
            "priority": 3,
            "economic_rationale": "Official corporate announcements can move prices with lag.",
            "required_data": "Announcement ledger with AVAILABLE_AT",
            "current_data_readiness": "PARTIAL",
            "sample_potential": "Dense once filings are timed",
            "pit_risk": "HIGH if scrape time ≠ public time",
            "cost_sensitivity": "HIGH",
            "priority_note": "Invest in official announcement timestamps",
        },
        {
            "family": "ownership_shareholding_effects",
            "priority": 5,
            "economic_rationale": "Promoter/FII ownership shifts may correlate with returns.",
            "required_data": "Shareholding filings with AVAILABLE_AT",
            "current_data_readiness": "DATA_MISSING",
            "sample_potential": "Quarterly panels",
            "pit_risk": "HIGH (current scrapes)",
            "cost_sensitivity": "LOW-MEDIUM",
            "priority_note": "After fundamentals/events infrastructure",
        },
    ]
    # Do not recommend closed families
    return sorted(families, key=lambda x: x["priority"])
