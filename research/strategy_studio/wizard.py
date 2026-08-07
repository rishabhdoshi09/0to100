"""
🧙 Guided strategy creation — a plain-language wizard.

Turns seven simple questions into a canonical StrategySpec. A manually-created strategy
faces the SAME evidence standards as an autonomous one — the wizard only builds the spec;
it does not grant any evidence or approval.

Pure: no streamlit, no network, no order path.
"""
from __future__ import annotations

from research.strategy_studio import spec as S

WIZARD_QUESTIONS = [
    {"key": "behaviour", "q": "What behaviour are you trying to capture?",
     "options": ["strength continuing (momentum)", "a pullback in a strong stock",
                 "a breakout from a base", "a bounce from oversold (mean reversion)",
                 "a strong sector leading"]},
    {"key": "universe", "q": "Which stocks should be eligible?",
     "options": ["liquid_nse", "nifty500", "nifty200"]},
    {"key": "entry", "q": "What should trigger an entry?",
     "options": ["breakout above resistance", "pullback to a moving average",
                 "new 6-month strength", "oversold bounce"]},
    {"key": "stop", "q": "Where is the idea proven wrong (the stop)?",
     "options": ["structural_stop", "atr_stop", "ema_stop"]},
    {"key": "exit", "q": "How should winners be held or exited?",
     "options": ["trend_exit", "ema_trail", "time_stop"]},
    {"key": "regime", "q": "In which market conditions should it operate?",
     "options": ["any market", "only strong markets"]},
    {"key": "risk", "q": "How much turnover and drawdown are acceptable?",
     "options": ["low (patient)", "medium", "higher (active)"]},
]

_BEHAVIOUR_FAMILY = {
    "strength continuing (momentum)": "cross_sectional_momentum",
    "a pullback in a strong stock": "pullback",
    "a breakout from a base": "breakout",
    "a bounce from oversold (mean reversion)": "mean_reversion",
    "a strong sector leading": "sector_rotation",
}
_ENTRY_BLOCK = {"breakout above resistance": "breakout_pivot",
                "pullback to a moving average": "pullback_to_ema",
                "new 6-month strength": "ret_6m", "oversold bounce": "mean_reversion_low"}
_RISK = {"low (patient)": (1.0, 0.20), "medium": (2.0, 0.30), "higher (active)": (3.5, 0.40)}


def wizard_to_spec(answers: dict, strategy_id: str = "STR-USER") -> S.StrategySpec:
    fam = _BEHAVIOUR_FAMILY.get(answers.get("behaviour"), "rule_combo")
    entry_block = _ENTRY_BLOCK.get(answers.get("entry"), "breakout_pivot")
    turnover, drawdown = _RISK.get(answers.get("risk"), (2.0, 0.30))
    regime = ("require_strong_market",) if answers.get("regime") == "only strong markets" else ()
    return S.StrategySpec(
        strategy_id=strategy_id, name="My strategy", version=1,
        hypothesis=f"A {fam.replace('_',' ')} idea entering on {entry_block.replace('_',' ')}.",
        family=fam, eligible_universe=answers.get("universe", "liquid_nse"),
        entry_rules=(f"require:{entry_block}",),
        stop_rules=(answers.get("stop", "structural_stop"),),
        exit_rules=(answers.get("exit", "trend_exit"),),
        regime_conditions=regime, feature_defs=(entry_block,),
        thresholds={"entry_timing": "next_bar_open"}, turnover_cap=turnover,
        generation_method="user_wizard")
