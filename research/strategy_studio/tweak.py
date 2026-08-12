"""
🛠️ Strategy Tuner — comfortable, no-code tweaking with an explicit audit trail.

Two ways to modify a strategy: guided controls (sliders/dropdowns with safe ranges) and
natural-language requests. A natural-language request is mapped to an EXPLICIT proposed
diff from a FIXED vocabulary — it never executes arbitrary code. Every MATERIAL change
creates a NEW version (new config hash), preserves the parent, invalidates the parent's
evidence for the new version, and must be retested from the beginning.

Pure: no streamlit, no network, no order path.
"""
from __future__ import annotations

from research.strategy_studio import spec as S

# guided controls with SAFE ranges (no arbitrary unrestricted values)
GUIDED_CONTROLS = {
    "eligible_universe": {"kind": "dropdown", "options": ["liquid_nse", "nifty500", "nifty200"]},
    "min_liquidity_cr": {"kind": "slider", "min": 1.0, "max": 50.0, "field": "liquidity_conditions"},
    "entry_timing": {"kind": "dropdown", "options": ["next_bar_open"], "field": "thresholds"},
    "max_holding_days": {"kind": "slider", "min": 2, "max": 120},
    "stop_method": {"kind": "dropdown", "options": ["structural_stop", "atr_stop", "ema_stop"],
                    "field": "stop_rules"},
    "max_initial_risk_pct": {"kind": "slider", "min": 1.0, "max": 8.0, "field": "thresholds"},
    "exit_method": {"kind": "dropdown", "options": ["trend_exit", "ema_trail", "time_stop"],
                    "field": "exit_rules"},
    "regime_filter": {"kind": "checkbox", "field": "regime_conditions",
                      "on": ("require_strong_market",)},
    "sector_requirement": {"kind": "checkbox", "field": "sector_conditions",
                           "on": ("require_strong_sector",)},
    "max_positions": {"kind": "slider", "min": 1, "max": 10},
    "rebalance_days": {"kind": "slider", "min": 0, "max": 60},
    "turnover_cap": {"kind": "slider", "min": 0.2, "max": 5.0},
}

# natural-language → explicit field change (FIXED vocabulary; never arbitrary code)
_NL_RULES = [
    (("deeper pullback", "wait for a deeper"), "thresholds", "pullback_depth_pct", 12.0,
     "waits for a bigger dip before buying — fewer, later entries"),
    (("avoid weak market", "avoid weak markets", "skip weak market"), "regime_conditions",
     None, ("require_strong_market",), "skips trades when the overall market is weak"),
    (("reduce the maximum stop", "max stop to 5", "tighter stop", "stop to 5"),
     "thresholds", "max_initial_risk_pct", 5.0, "smaller planned losses, more rejects"),
    (("hold winners longer", "hold longer"), "max_holding_days", None, 90,
     "keeps winning trades open longer"),
    (("remove the sector filter", "drop the sector"), "sector_conditions", None, (),
     "no longer requires a strong sector — more trades, weaker filter"),
    (("fewer but stronger", "fewer stocks", "use fewer"), "max_positions", None, 3,
     "holds fewer, higher-conviction names"),
    (("less complicated", "make this simpler", "simplify"), "__simplify__", None, None,
     "removes the least-important rule to reduce complexity"),
]

_UNSAFE = ("always win", "guarantee", "no losses", "risk-free", "delete everything",
           "import ", "os.", "exec(", "__")


class TweakRejected(Exception):
    pass


def parse_nl(request: str, spec: S.StrategySpec) -> dict:
    """Map a plain-language request to an EXPLICIT proposed diff. Returns a diff dict
    with status 'ready' | 'needs_clarification' | 'rejected'. Never runs arbitrary code."""
    text = (request or "").strip().lower()
    if not text:
        return {"status": "needs_clarification", "why": "empty request"}
    if any(u in text for u in _UNSAFE):
        return {"status": "rejected",
                "why": "that request is unsafe or impossible (no strategy can guarantee "
                       "wins, and free-text cannot run code)"}
    for keys, field_, key, value, effect in _NL_RULES:
        if any(k in text for k in keys):
            return _diff_for(spec, field_, key, value, effect)
    return {"status": "needs_clarification",
            "why": "I couldn't map that to a known change. Try one of the guided controls, "
                   "or rephrase (e.g. 'reduce the maximum stop to 5%')."}


def _diff_for(spec: S.StrategySpec, field_, key, value, effect) -> dict:
    if field_ == "__simplify__":
        if not spec.entry_rules:
            return {"status": "needs_clarification", "why": "nothing to simplify"}
        old = spec.entry_rules
        proposed = tuple(old[:-1])
        return {"status": "ready", "field": "entry_rules", "old": old, "proposed": proposed,
                "expected_effect": effect, "invalidates_evidence": True,
                "new_version_required": True}
    if field_ == "thresholds":
        old = dict(spec.thresholds)
        proposed = {**old, key: value}
        return {"status": "ready", "field": "thresholds", "old": old, "proposed": proposed,
                "expected_effect": effect, "invalidates_evidence": True,
                "new_version_required": True}
    old = getattr(spec, field_)
    return {"status": "ready", "field": field_, "old": old, "proposed": value,
            "expected_effect": effect, "invalidates_evidence": True,
            "new_version_required": True}


def tweak_impact_preview(spec: S.StrategySpec, diff: dict) -> dict:
    """The 'You changed … / Likely effect … / Required action …' preview."""
    if diff.get("status") != "ready":
        return {"status": diff.get("status"), "why": diff.get("why")}
    return {
        "you_changed": {"field": diff["field"], "from": diff["old"], "to": diff["proposed"]},
        "likely_effect": diff["expected_effect"],
        "evidence_note": "The previous evidence no longer applies to the changed strategy.",
        "required_action": (f"Create {spec.strategy_id} v{spec.version + 1} and run a new "
                            "historical test."),
        "new_version_required": True,
    }


def apply_diff(spec: S.StrategySpec, diff: dict) -> S.StrategySpec:
    """Apply a READY diff → a NEW version (new config hash), preserving the parent. Old
    evidence does NOT transfer. Raises if the diff is not ready."""
    if diff.get("status") != "ready":
        raise TweakRejected(diff.get("why", "diff not ready"))
    field_ = diff["field"]; proposed = diff["proposed"]
    new = spec.bump_version(**{field_: proposed})
    return new


def apply_display_change(spec: S.StrategySpec, *, name=None, hypothesis=None) -> S.StrategySpec:
    """A NON-material change (rename / reword). Does NOT bump the version and does NOT
    change the config hash — the evidence identity is preserved."""
    import dataclasses
    kw = {}
    if name is not None:
        kw["name"] = name
    if hypothesis is not None:
        kw["hypothesis"] = hypothesis
    return dataclasses.replace(spec, **kw)
