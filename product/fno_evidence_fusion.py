"""The evidence-fusion policy: how historical (COUNTERFACTUAL) and forward
(PAPER_FORWARD) F&O evidence combine into one ranking adjustment.

This is the explicit implementation of the desk's evidence hierarchy for
F&O, replacing "historical evidence can never move ranking" (too
conservative to ever get wiser from simulation) with "historical evidence
may act as a small, bounded PRIOR; real forward paper evidence is always the
stronger vote; live-money authority is unaffected either way."

    Historical walk-forward evidence  = hypothesis / prior / weak vote
    Real PAPER_FORWARD evidence       = stronger vote
    Live-money authority              = still locked / unaffected

Nothing in this module ever authorizes live money, and nothing here can
promote OR demote past its explicit cap -- see the constants below. Historical
evidence is DELIBERATELY capped far smaller than forward evidence for two
concrete, checkable reasons, not merely convention:
  1. It carries no cost model (product.fno_historical_walkforward is a pure
     price simulation); forward evidence's expectancy_R is already net of
     the same cost model real paper trades pay.
  2. It was never actually acted on, so it cannot confirm execution/slippage/
     path realism the way a genuinely settled trade does.

Regime dependence is handled by construction, not by a separate check: every
cell this module reads is already keyed by product.fno_evidence
.fno_context_key, which buckets market regime, sector state, volatility and
extension into the key itself. Two different regimes are, by definition,
two different cells -- there is no pooled statistic to accidentally
generalise across them.

Stability/drift is checked directly: each cell's r_values are stored in
settlement order, so splitting them in half and comparing the two halves'
sign catches a setup whose edge reversed mid-sample (a regime break, a
market-structure shift) before that reversal gets averaged away.

Policy table (see fuse_fno_ranking_evidence's docstring for the exact
branches):
  no evidence                              -> 0 (baseline)
  historical only, thin sample             -> 0
  historical only, large + stable          -> small bounded prior
  forward usable (any direction)           -> forward dominates
  historical agrees with forward           -> historical prior ADDS to it
  historical disagrees with forward        -> historical is ignored entirely
  forward positive but thin/unstable/weak  -> 0 (never promote on thin ice)
"""
from __future__ import annotations

from typing import Any, Mapping

from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD
from product.fno_evidence import fno_context_key

# Forward (PAPER_FORWARD) evidence -- the stronger vote.
FORWARD_MIN_SAMPLE_DEMOTE = 30      # matches product.conditional_evidence.MIN_SAMPLE
FORWARD_MIN_SAMPLE_PROMOTE = 40     # promotion is held to a stricter floor than demotion
FORWARD_NEGATIVE_CAP = -30.0        # matches the equity desk's existing demote cap
FORWARD_POSITIVE_CAP = 15.0         # promotion is capped at HALF the demotion magnitude
FORWARD_PROMOTE_WILSON_MIN = 0.50   # even the pessimistic win-rate estimate must clear a coin flip

# Historical (COUNTERFACTUAL) evidence -- a weak, bounded prior only.
HISTORICAL_MIN_SAMPLE = 50          # stricter than forward: uncosted, never actually traded
HISTORICAL_CAP = 5.0                # a fraction of even the forward promotion cap

# The combined adjustment (historical_prior + forward_adjustment) is bounded
# by the sum of each side's own cap -- agreement between the two is allowed
# to matter MORE than either alone, but never runs away.
COMBINED_POSITIVE_CAP = FORWARD_POSITIVE_CAP + HISTORICAL_CAP
COMBINED_NEGATIVE_CAP = FORWARD_NEGATIVE_CAP - HISTORICAL_CAP

_MIN_SPLIT_SAMPLE = 4  # below this a half/half stability read is not meaningful


def _same_sign(a: float, b: float) -> bool:
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def _stability(r_values: list[float]) -> dict[str, Any]:
    """Chronological first-half vs second-half sign check.

    r_values are appended to the cell in settlement order (never reordered),
    so this catches a setup whose edge reversed partway through its sample --
    a real regime break -- before the average quietly smooths it over.
    """
    n = len(r_values)
    if n < _MIN_SPLIT_SAMPLE:
        return {
            "checked": False,
            "stable": True,  # too few points to call it unstable; caller still gates on min sample
            "first_half_expectancy": None,
            "second_half_expectancy": None,
        }
    mid = n // 2
    first = r_values[:mid]
    second = r_values[mid:]
    first_exp = sum(first) / len(first)
    second_exp = sum(second) / len(second)
    stable = not ((first_exp > 0 and second_exp < 0) or (first_exp < 0 and second_exp > 0))
    return {
        "checked": True,
        "stable": stable,
        "first_half_expectancy": round(first_exp, 4),
        "second_half_expectancy": round(second_exp, 4),
    }


def _forward_component(cell: Mapping[str, Any]) -> dict[str, Any]:
    count = int(cell.get("count") or 0)
    r_values = [float(r) for r in (cell.get("r_values") or [])]
    stability = _stability(r_values)
    base = {"count": count, "stability": stability, "evidence_class": PAPER_FORWARD}
    if count < FORWARD_MIN_SAMPLE_DEMOTE:
        return {**base, "usable": False, "direction": "NONE", "adjustment": 0.0,
                "reason": "INSUFFICIENT_FORWARD_SAMPLE"}

    expectancy = float(cell.get("expectancy_R") or 0.0)
    wilson = cell.get("wilson_lower_bound")
    base.update({"expectancy_R": expectancy, "wilson_lower_bound": wilson})

    if expectancy < 0:
        # Demotion keeps the exact formula/cap the desk has always used --
        # this path is unchanged from before evidence fusion existed.
        adjustment = max(FORWARD_NEGATIVE_CAP, round(expectancy * 20.0, 4))
        return {**base, "usable": True, "direction": "DEMOTE", "adjustment": adjustment,
                "reason": "FORWARD_NEGATIVE_EXPECTANCY"}

    if expectancy > 0:
        if count < FORWARD_MIN_SAMPLE_PROMOTE:
            return {**base, "usable": False, "direction": "NONE", "adjustment": 0.0,
                    "reason": "INSUFFICIENT_SAMPLE_FOR_PROMOTION"}
        if wilson is None or float(wilson) < FORWARD_PROMOTE_WILSON_MIN:
            return {**base, "usable": False, "direction": "NONE", "adjustment": 0.0,
                    "reason": "WIN_RATE_LOWER_BOUND_TOO_WEAK_FOR_PROMOTION"}
        if not stability["stable"]:
            return {**base, "usable": False, "direction": "NONE", "adjustment": 0.0,
                    "reason": "FORWARD_UNSTABLE_ACROSS_PERIODS"}
        # Promotion multiplier (10x) is intentionally half the demotion
        # multiplier (20x): earning a bonus is held to a higher bar than
        # losing one.
        adjustment = min(FORWARD_POSITIVE_CAP, round(expectancy * 10.0, 4))
        return {**base, "usable": True, "direction": "PROMOTE", "adjustment": adjustment,
                "reason": "FORWARD_POSITIVE_EXPECTANCY_VALIDATED"}

    return {**base, "usable": True, "direction": "NONE", "adjustment": 0.0, "reason": "FORWARD_FLAT"}


def _historical_component(cell: Mapping[str, Any]) -> dict[str, Any]:
    count = int(cell.get("count") or 0)
    r_values = [float(r) for r in (cell.get("r_values") or [])]
    stability = _stability(r_values)
    base = {"count": count, "stability": stability, "evidence_class": COUNTERFACTUAL}
    if count < HISTORICAL_MIN_SAMPLE:
        return {**base, "usable": False, "direction": "NONE", "adjustment": 0.0,
                "reason": "INSUFFICIENT_HISTORICAL_SAMPLE"}
    if not stability["stable"]:
        return {**base, "usable": False, "direction": "NONE", "adjustment": 0.0,
                "reason": "HISTORICAL_UNSTABLE_ACROSS_PERIODS"}
    expectancy = float(cell.get("expectancy_R") or 0.0)
    # Small, symmetric, uncosted prior -- see the module docstring for why
    # this cap is a fraction of even the forward promotion cap.
    adjustment = max(-HISTORICAL_CAP, min(HISTORICAL_CAP, round(expectancy * 5.0, 4)))
    if adjustment == 0.0:
        return {**base, "usable": False, "direction": "NONE", "adjustment": 0.0,
                "expectancy_R": expectancy, "reason": "HISTORICAL_FLAT"}
    direction = "PROMOTE" if adjustment > 0 else "DEMOTE"
    return {**base, "usable": True, "direction": direction, "adjustment": adjustment,
            "expectancy_R": expectancy, "reason": "HISTORICAL_ROBUST_PRIOR"}


def fuse_fno_ranking_evidence(
    setup: Mapping[str, Any] | None,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    """The single place F&O ranking reads to decide historical_prior,
    forward_adjustment and their sum. Never called for equity; never reads
    or writes anything at the broker boundary.

    Branches, in order:
      1. No context key (no direction on the setup) -> zero everything.
      2. Forward evidence usable (enough sample, and if positive: enough
         sample AND a Wilson floor above 0.50 AND stable across the sample's
         two halves) -> forward's own adjustment is used as-is (demote or
         promote). Historical is folded in ONLY if it agrees in sign with
         forward -- disagreement means forward dominates completely and
         historical contributes nothing, exactly the "historical positive,
         forward negative -> forward wins" and "historical negative, forward
         strongly positive -> forward can recover the ranking" rules.
      3. Forward NOT usable (too thin, too unstable, or simply doesn't exist
         yet) but historical IS usable (large + stable) -> historical's own
         small, bounded prior acts alone. This is the only path through
         which historical evidence can move a ranking with zero forward
         confirmation, and it is capped an order of magnitude below what
         forward evidence alone can do.
      4. Neither usable -> 0, the untouched baseline score stands.
    """
    setup = setup or {}
    key = fno_context_key(setup)
    if not key:
        empty = {"usable": False, "direction": "NONE", "adjustment": 0.0,
                 "count": 0, "reason": "NO_CONTEXT_KEY", "evidence_class": PAPER_FORWARD}
        return {
            "context_key": "",
            "usable": False,
            "evidence_class": PAPER_FORWARD,
            "count": 0,
            "expectancy_R": None,
            "wilson_lower_bound": None,
            "forward": empty,
            "historical": dict(empty, evidence_class=COUNTERFACTUAL),
            "historical_prior": 0.0,
            "forward_adjustment": 0.0,
            "adjustment": 0.0,
            "status": "NO_CONTEXT_KEY",
            "reason": "NO_CONTEXT_KEY",
            "historical_note": "NONE",
        }

    from product.conditional_evidence import load as _load, read as _read

    store = _load(path)
    forward_cell = _read(key, evidence_class=PAPER_FORWARD, path=path, store=store)
    historical_cell = _read(key, evidence_class=COUNTERFACTUAL, path=path, store=store)
    forward = _forward_component(forward_cell)
    historical = _historical_component(historical_cell)

    if forward["usable"]:
        forward_adjustment = float(forward["adjustment"])
        if historical["usable"] and _same_sign(historical["adjustment"], forward_adjustment):
            historical_prior = float(historical["adjustment"])
            historical_note = "AGREES_WITH_FORWARD_ADDED"
        elif historical["usable"]:
            historical_prior = 0.0
            historical_note = "DISAGREES_WITH_FORWARD_IGNORED"
        else:
            historical_prior = 0.0
            historical_note = "NO_USABLE_HISTORICAL_EVIDENCE"
        status = "FORWARD_DOMINATES"
        reason = forward["reason"]
    elif historical["usable"]:
        historical_prior = float(historical["adjustment"])
        forward_adjustment = 0.0
        historical_note = "ACTING_ALONE_NO_FORWARD_SIGNAL"
        status = "HISTORICAL_PRIOR_ONLY"
        reason = historical["reason"]
    else:
        historical_prior = 0.0
        forward_adjustment = 0.0
        historical_note = "NONE"
        status = "NO_EVIDENCE" if (forward["count"] == 0 and historical["count"] == 0) else "INSUFFICIENT_EVIDENCE"
        reason = forward["reason"] if forward["count"] else historical["reason"]

    combined = historical_prior + forward_adjustment
    if combined > 0:
        combined = min(combined, COMBINED_POSITIVE_CAP)
    elif combined < 0:
        combined = max(combined, COMBINED_NEGATIVE_CAP)

    usable = bool(forward["usable"] or historical["usable"])
    # Flat convenience fields mirror whichever side actually decided the
    # outcome (forward when it dominates or ties, historical only in the
    # historical-prior-only branch), so a caller that just wants "how much
    # evidence, from which class" does not have to know the fusion internals.
    dominant = forward if status in ("FORWARD_DOMINATES", "NO_EVIDENCE", "INSUFFICIENT_EVIDENCE") else historical

    return {
        "context_key": key,
        "usable": usable,
        "evidence_class": dominant.get("evidence_class", PAPER_FORWARD),
        "count": dominant.get("count", 0),
        "expectancy_R": dominant.get("expectancy_R"),
        "wilson_lower_bound": forward.get("wilson_lower_bound"),
        "forward": forward,
        "historical": historical,
        "historical_prior": round(historical_prior, 4),
        "forward_adjustment": round(forward_adjustment, 4),
        "adjustment": round(combined, 4),
        "status": status,
        "reason": reason,
        "historical_note": historical_note,
    }
