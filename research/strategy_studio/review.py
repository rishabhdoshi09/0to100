"""
⚖️ Review layer — separated confidence, the "Convince Me" case, and comparison.

A strategy is NEVER summarised by one vague "confidence" number, and NEVER sold with a
score or AI sales language. QuantTerm presents the PROSECUTION and the DEFENCE of every
strategy, labels evidence vs plausibility vs speculation, and refuses to call synthetic
software-test results market evidence.

Pure: no streamlit, no network, no order path.
"""
from __future__ import annotations

from research.strategy_studio import spec as S
from research.strategy_studio.discovery import EvidenceReport


# ══════════════════════════════════════════════════════════════════════════════
# Five SEPARATE confidences (never collapsed into one)
# ══════════════════════════════════════════════════════════════════════════════

def confidences(ev: EvidenceReport, data_status: dict | None, n_attempts: int,
                paper_reproducibility: float | None = None) -> dict:
    """Return the five confidence dimensions separately. A high model probability alone
    must NOT make a strategy look trustworthy — that is only 'prediction'."""
    ev_conf = 0.0
    if not ev.is_synthetic:                       # synthetic never earns evidence conf
        base = {"PROMOTE": 80, "PASS": 80}.get(ev.verdict, 0)
        base += min(15, max(0, (ev.n_trades - 30) / 20))
        base -= min(30, n_attempts / 2000.0)      # big search → lower evidence conf
        ev_conf = max(0.0, min(100.0, base))
    data_conf = {"green": 85.0, "amber": 55.0}.get((data_status or {}).get("color"), 10.0)
    stability = 100.0 * (0.5 * ev.regime_consistency + 0.5 * ev.sector_consistency)
    if ev.depends_on_one_period:
        stability *= 0.4
    exec_conf = (100.0 * paper_reproducibility if paper_reproducibility is not None
                 else max(0.0, 70.0 - ev.turnover * 10.0))   # high turnover = harder to reproduce
    return {
        "evidence": round(ev_conf, 0),
        "prediction": None,                       # rule-based: no per-trade model prob
        "data": round(data_conf, 0),
        "stability": round(min(100.0, stability), 0),
        "execution": round(max(0.0, min(100.0, exec_conf)), 0),
        "note": "Five separate measures — a strategy is not trustworthy just because one "
                "is high.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Convince Me — prosecution AND defence
# ══════════════════════════════════════════════════════════════════════════════

_REJECT_WORDS = ("guaranteed", "certain", "safe profit", "assured", "risk-free")

RECO_REJECT = "Reject"
RECO_MORE_EVIDENCE = "Needs more evidence"
RECO_REVIEW = "Promising enough for user review"
RECO_PAPER = "Suitable for paper evaluation"
RECO_UNSUITABLE = "Not suitable for implementation"


def recommendation(ev: EvidenceReport, data_status: dict | None) -> str:
    if ev.is_synthetic:
        return RECO_UNSUITABLE            # synthetic demo is not market evidence
    if ev.verdict in ("REJECT", "FAIL"):
        return RECO_REJECT
    if ev.verdict in ("PROMOTE", "PASS"):
        if (data_status or {}).get("color") == "green":
            return RECO_PAPER
        return RECO_REVIEW
    return RECO_MORE_EVIDENCE              # INCONCLUSIVE / UNDERPOWERED


def convince_me(spec: S.StrategySpec, ev: EvidenceReport, *, data_status: dict | None,
                n_attempts: int, n_rejected: int, trades: list | None = None,
                baselines: dict | None = None, dataset_period: str = "",
                validation=None, untouched=None, limitations=None) -> dict:
    """Build the full investment-committee case. Includes BOTH the defence (why it may
    work + evidence) and the prosecution (what could go wrong + why-not-luck limits)."""
    trades = trades or []
    limitations = list(limitations or [])
    synth = ev.is_synthetic

    how = [f"Enter on the next day after: {r.replace('require:', '')}"
           for r in spec.entry_rules] or ["Enter when the setup qualifies (next day)."]
    how.append(f"Exit-if-wrong: {', '.join(spec.stop_rules) or 'structural stop'}.")
    how.append(f"Take profit / trend exit: {', '.join(spec.exit_rules) or 'trend exit'}; "
               f"max hold {spec.max_holding_days} days.")

    # why it may work — LABELLED
    why = {
        "evidence_supported": ([] if synth else
                               [f"On {dataset_period or 'the tested period'} it produced "
                                f"{ev.net_expectancy_R:+.2f}R per trade over {ev.n_trades} "
                                "trades after costs."]),
        "plausible": [f"{spec.family.replace('_',' ').title()} ideas exploit a widely "
                      "documented tendency for strength/weakness to persist."],
        "speculation": ["Whether this specific rule set keeps working in future is "
                        "unknown — markets adapt."],
    }

    evidence_summary = {
        "is_market_evidence": not synth,
        "synthetic_note": ("These numbers come from SYNTHETIC test data and are NOT "
                           "market evidence." if synth else ""),
        "dataset_period": dataset_period, "n_trades": ev.n_trades,
        "net_expectancy_R": ev.net_expectancy_R, "cost_drag_R": ev.cost_drag_R,
        "max_drawdown": ev.max_drawdown, "benchmark": spec.benchmark,
        "benchmark_relative": (baselines or {}).get("vs_nifty"),
        "validation": validation, "untouched_test": untouched,
        "regime_consistency": ev.regime_consistency,
        "sector_consistency": ev.sector_consistency,
        "multiple_testing": {"attempts": n_attempts, "rejected": n_rejected},
        "paper_evidence": None,
    }

    what_could_go_wrong = {
        "worst_period": ev.worst_period or "unknown",
        "worst_drawdown": ev.max_drawdown,
        "longest_losing_streak": (ev.as_dict().get("longest_losing_streak")
                                  if hasattr(ev, "as_dict") else None),
        "gap_risk": "A stock can gap through the stop and exit worse than planned.",
        "liquidity_risk": "Thin names can be hard to enter/exit at the assumed price.",
        "concentration_risk": f"Largest single-name share was {ev.max_symbol_share:.0%}.",
        "evidence_limitations": limitations,
        "invalidation_conditions": [
            "A run of losses beyond the historical worst.",
            "The edge disappearing in live/paper vs history.",
            "The market regime it relies on ending."],
        "depends_on_few_trades": ev.depends_on_few_trades,
    }

    why_not_luck = {
        "total_strategies_attempted": n_attempts,
        "strategies_rejected": n_rejected,
        "controls_applied": ["realistic costs", "minimum trade count",
                             "multiple-testing adjustment", "untouched-test isolation",
                             "simpler-baseline comparison"],
        "untouched_test_result": untouched,
        "depends_on_a_few_trades": ev.depends_on_few_trades,
    }

    simpler = baselines or {"vs_nifty": None, "vs_simple_momentum": None,
                            "vs_simple_trend": None, "vs_closest_simpler": None}

    example_trades = _representative(trades)

    return {
        "one_line_idea": spec.hypothesis,
        "how_it_works": how,
        "why_it_may_work": why,                          # DEFENCE
        "evidence_summary": evidence_summary,
        "what_could_go_wrong": what_could_go_wrong,      # PROSECUTION
        "why_this_is_not_luck": why_not_luck,
        "simpler_alternatives": simpler,
        "example_trades": example_trades,
        "system_recommendation": recommendation(ev, data_status),
        "user_decision_options": ["Reject Strategy", "Save for Later",
                                  "Tweak Strategy", "Approve for Paper Testing"],
        "confidences": confidences(ev, data_status, n_attempts),
        "no_live_option": True,
        "synthetic_labelled_non_evidence": synth,
    }


def _representative(trades: list) -> dict:
    """A NON-cherry-picked sample: guarantee wins AND losses are shown when both exist."""
    wins = [t for t in trades if t.get("net_R", 0) > 0]
    losses = [t for t in trades if t.get("net_R", 0) <= 0]
    ordinary = sorted(trades, key=lambda t: abs(t.get("net_R", 0)))[:2]
    gaps = [t for t in trades if t.get("exit_reason") == "gap_stop"]
    return {"wins": wins[:2], "losses": losses[:2], "ordinary": ordinary,
            "gap_cases": gaps[:1], "includes_losses": bool(losses),
            "cherry_picked": False}


def sanitize_language(text: str) -> str:
    """Guard against over-promising words in any generated blurb."""
    low = text.lower()
    for w in _REJECT_WORDS:
        if w in low:
            return "[removed over-promising claim] " + text
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Comparison (never auto-pick the highest return)
# ══════════════════════════════════════════════════════════════════════════════

def compare(strategies: list[dict]) -> dict:
    """Side-by-side. `strategies` = [{name, spec, ev}]. Flags when differences are NOT
    statistically meaningful, and NEVER auto-selects a winner."""
    rows = []
    for s in strategies:
        ev: EvidenceReport = s["ev"]; sp: S.StrategySpec = s["spec"]
        rows.append({"name": s["name"], "complexity": S.complexity_of(sp),
                     "n_trades": ev.n_trades, "net_expectancy_R": ev.net_expectancy_R,
                     "max_drawdown": ev.max_drawdown, "turnover": ev.turnover,
                     "regime_consistency": ev.regime_consistency,
                     "is_synthetic": ev.is_synthetic})
    # meaningfulness: crude but honest — if best two are within 0.05R, not meaningful
    exps = sorted((r["net_expectancy_R"] for r in rows), reverse=True)
    meaningful = len(exps) < 2 or (exps[0] - exps[1]) >= 0.05
    return {"rows": rows, "auto_selected": None,           # never auto-choose
            "difference_is_meaningful": meaningful,
            "note": ("Differences are within noise — prefer the simpler / more stable one."
                     if not meaningful else "There is a meaningful difference.")}
