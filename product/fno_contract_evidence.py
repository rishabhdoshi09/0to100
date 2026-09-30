"""Evidence-aware learning for OPTION CONTRACT selection.

Deliberately separate from product.fno_evidence_fusion, which grades the
UNDERLYING directional call (was LONG/SHORT the right side, in this regime,
at this extension). This module grades a different question entirely: given
that a direction was called, was THIS contract (this strike/expiry/delta/IV/
liquidity shape) the right one to buy? A losing option trade does not prove
the underlying call was wrong -- the underlying can move exactly as
predicted while theta, a wide spread, or a too-low delta still lose the
trade. Pooling the two questions into one statistic hides which one to fix.

Same evidence-class discipline as the rest of the desk: this only ever reads
and writes PAPER_FORWARD cells (real settled paper option trades -- no
broker order path, no live money). There is no historical NSE option-chain
data source anywhere in this repository, live or replayed, so there is no
COUNTERFACTUAL/BACKTEST contract cell to fuse against -- fabricating one
would be inventing evidence. Contract-selection learning is forward-only by
construction until a trustworthy historical option-chain source exists.

The context key (CTXCTX_V1) buckets the contract's own shape -- option type,
moneyness, delta, days-to-expiry, IV percentile, bid/ask spread -- entirely
independent of which underlying/setup/regime chose it, so a lesson learned
about "deep-OTM, <2 DTE, wide-spread" contracts generalises across every
setup that might pick one, rather than being re-learned per setup.

Caps are deliberately smaller than the underlying-ranking caps in
product.fno_evidence_fusion (-20/+10 vs -30/+15): a contract-quality signal
should nudge which strike/expiry gets picked, never override a strong
underlying signal.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.conditional_evidence import load as _load, read as _read, record_outcome
from product.decision_chain import Outcome
from product.evidence_class import PAPER_FORWARD

CONTRACT_CONTEXT_SCHEMA_VERSION = "CTXCTX_V1"

# Forward (PAPER_FORWARD) contract evidence -- the only evidence class this
# module ever produces or reads. Same demote/promote asymmetry as the
# underlying-ranking policy (product.fno_evidence_fusion), scaled to a
# smaller cap: a contract-shape lesson refines selection, it does not
# override the underlying call.
MIN_SAMPLE_DEMOTE = 30
MIN_SAMPLE_PROMOTE = 40
NEGATIVE_CAP = -20.0
POSITIVE_CAP = 10.0
PROMOTE_WILSON_MIN = 0.50
_MIN_SPLIT_SAMPLE = 4

# Classification thresholds -- each one reads a REAL field captured at entry
# (see FoPaperPosition/FoPaperTrade's entry_* fields) or a REAL exit fact the
# paper book itself already computed (exit_reason, exit_underlying_spot).
# None of these invent information the desk did not actually observe.
WIDE_SPREAD_PCT = 3.0
LOW_DELTA_ABS = 0.25
HIGH_DELTA_ABS = 0.85
SHORT_DTE_DAYS = 1
THIN_LIQUIDITY_OI = 500

CLASSIFICATION_UNKNOWN = "INSUFFICIENT_EVIDENCE_FOR_CLASSIFICATION"
CLASSIFICATION_UNDERLYING_RIGHT_CONTRACT_RIGHT = "UNDERLYING_RIGHT_CONTRACT_RIGHT"
CLASSIFICATION_UNDERLYING_RIGHT_CONTRACT_POOR = "UNDERLYING_RIGHT_CONTRACT_POOR"
CLASSIFICATION_UNDERLYING_WRONG_CONTRACT_OK_MECHANICS = "UNDERLYING_WRONG_CONTRACT_OK_MECHANICS"
CLASSIFICATION_UNDERLYING_WRONG_CONTRACT_WRONG = "UNDERLYING_WRONG_CONTRACT_WRONG"
CLASSIFICATION_THETA_DAMAGE = "THETA_DAMAGE"
CLASSIFICATION_IV_CRUSH = "IV_CRUSH"
CLASSIFICATION_WIDE_SPREAD_COST = "WIDE_SPREAD_COST"
CLASSIFICATION_BAD_EXPIRY_CHOICE = "BAD_EXPIRY_CHOICE"
CLASSIFICATION_DELTA_TOO_LOW = "DELTA_TOO_LOW"
CLASSIFICATION_DELTA_TOO_HIGH = "DELTA_TOO_HIGH"
CLASSIFICATION_POOR_LIQUIDITY = "POOR_LIQUIDITY"
# ENTRY_TIMING_POOR is intentionally never assigned: no field captures
# intended-vs-actual entry timing quality today, and inventing a timing
# verdict from data that doesn't exist would violate the "never invent"
# rule this module is built around. It stays defined so a future field
# addition has a name to write to.
CLASSIFICATION_ENTRY_TIMING_POOR = "ENTRY_TIMING_POOR"


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0  # filters NaN


def _delta_bucket(delta_abs: float | None) -> str:
    if delta_abs is None:
        return "UNKNOWN"
    value = abs(float(delta_abs))
    if value < 0.30:
        return "LOW"
    if value < 0.50:
        return "MID"
    if value < 0.70:
        return "HIGH"
    return "DEEP"


def _dte_bucket(dte: int | None) -> str:
    if dte is None:
        return "UNKNOWN"
    value = int(dte)
    if value <= 1:
        return "0_1D"
    if value <= 7:
        return "2_7D"
    if value <= 15:
        return "8_15D"
    return "16D_PLUS"


def _iv_percentile_bucket(iv_pct: float | None) -> str:
    if iv_pct is None:
        return "UNKNOWN"
    value = float(iv_pct)
    if value < 30.0:
        return "LOW"
    if value < 70.0:
        return "MID"
    return "HIGH"


def _spread_bucket(spread_pct: float | None) -> str:
    if spread_pct is None:
        return "UNKNOWN"
    value = float(spread_pct)
    if value < 1.0:
        return "TIGHT"
    if value < WIDE_SPREAD_PCT:
        return "NORMAL"
    return "WIDE"


def contract_context_key(contract: Mapping[str, Any] | None) -> str:
    """Bucket a REAL option contract by its own shape, not by the setup.

    ``contract`` is the dict options.directional_selector.score_option_contract
    already computes (or the equivalent fields captured on a FoPaperPosition/
    FoPaperTrade at entry). Returns "" when there is not even an option type
    to key on -- callers must treat that as "cannot record contract evidence
    for this trade", not as a valid cell.
    """
    contract = contract or {}
    option_type = str(contract.get("option_type") or "").upper()
    if option_type not in ("CE", "PE"):
        return ""
    delta = contract.get("delta")
    dte = contract.get("dte")
    iv_percentile = contract.get("iv_percentile")
    spread_pct = contract.get("spread_pct")
    moneyness = str(contract.get("moneyness") or "UNKNOWN").upper()
    parts = [
        CONTRACT_CONTEXT_SCHEMA_VERSION,
        f"type={option_type}",
        f"money={moneyness or 'UNKNOWN'}",
        f"delta={_delta_bucket(_f(delta) if delta is not None else None)}",
        f"dte={_dte_bucket(int(_f(dte)) if dte is not None else None)}",
        f"iv={_iv_percentile_bucket(_f(iv_percentile) if iv_percentile is not None else None)}",
        f"spread={_spread_bucket(_f(spread_pct) if spread_pct is not None else None)}",
    ]
    return "|".join(parts)


def record_contract_settlement(
    trade_row: Mapping[str, Any],
    *,
    context_key: str,
    path: str | None = None,
) -> Any:
    """Fold one settled F&O paper trade into CONTRACT evidence, or skip it.

    Mirrors product.fno_evidence.record_fno_settlement's honesty rules
    exactly (same eligibility gate, same R computation), but keys the cell by
    the contract's own shape instead of the underlying setup, so the two
    statistics never pool.
    """
    if not context_key:
        return None
    if not bool(trade_row.get("production_evidence_eligible")):
        return None
    entry = _f(trade_row.get("entry_price"))
    stop = _f(trade_row.get("stop_price"))
    risk_per_unit = abs(entry - stop)
    quantity = _f(trade_row.get("quantity"))
    risk_amount = risk_per_unit * quantity
    if risk_amount <= 0:
        return None
    net_pnl = _f(trade_row.get("net_pnl"))
    realized_R = net_pnl / risk_amount
    mfe_pct = trade_row.get("mfe_pct")
    mae_pct = trade_row.get("mae_pct")
    mfe_R = (
        (_f(mfe_pct) / 100.0 * entry) / risk_per_unit
        if mfe_pct is not None and risk_per_unit > 0 else None
    )
    mae_R = (
        (_f(mae_pct) / 100.0 * entry) / risk_per_unit
        if mae_pct is not None and risk_per_unit > 0 else None
    )
    trade_id = str(trade_row.get("trade_id") or "")
    if not trade_id:
        return None
    outcome = Outcome(
        position_id=f"contract::{trade_id}",
        paper_order_id=trade_id,
        paper_intent_id=trade_id,
        decision_id=trade_id,
        symbol=str(trade_row.get("option_symbol") or trade_row.get("underlying") or ""),
        exit_reason=str(trade_row.get("exit_reason") or ""),
        realized_R=realized_R,
        entry_price=entry or None,
        exit_price=_f(trade_row.get("exit_price")) or None,
        entry_session=str(trade_row.get("opened_at") or "")[:10],
        exit_session=str(trade_row.get("settled_at") or "")[:10],
        mae_R=mae_R,
        mfe_R=mfe_R,
        evidence_class=PAPER_FORWARD,
        resolved_at=str(trade_row.get("settled_at") or ""),
    )
    return record_outcome(
        outcome,
        context_key=context_key,
        evidence_class=PAPER_FORWARD,
        path=path,
        # The contract cell is a refinement signal on top of the setup
        # ladder, not a second vote in it -- the setup ladder already
        # records this same settlement under ranking_context_key via
        # record_fno_settlement.
        also_update_policy_ladder=False,
    )


def _stability(r_values: list[float]) -> dict[str, Any]:
    n = len(r_values)
    if n < _MIN_SPLIT_SAMPLE:
        return {"checked": False, "stable": True}
    mid = n // 2
    first = r_values[:mid]
    second = r_values[mid:]
    first_exp = sum(first) / len(first)
    second_exp = sum(second) / len(second)
    stable = not ((first_exp > 0 and second_exp < 0) or (first_exp < 0 and second_exp > 0))
    return {"checked": True, "stable": stable}


def contract_ranking_adjustment(
    contract: Mapping[str, Any] | None,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    """The learned contract-selection adjustment, on top of raw_contract_score.

    Same shape as product.fno_evidence_fusion's forward component: demote is
    held to a lower sample floor than promote, and promotion additionally
    requires a Wilson lower bound clearing a coin flip and stability across
    the sample's two chronological halves. Below every floor this returns
    adjustment=0.0 -- a tiny sample can never move contract selection.
    """
    key = contract_context_key(contract)
    if not key:
        return {
            "context_key": "", "usable": False, "count": 0,
            "adjustment": 0.0, "direction": "NONE",
            "reason": "NO_CONTRACT_CONTEXT_KEY",
        }
    store = _load(path)
    cell = _read(key, evidence_class=PAPER_FORWARD, path=path, store=store)
    count = int(cell.get("count") or 0)
    r_values = [float(r) for r in (cell.get("r_values") or [])]
    stability = _stability(r_values)
    base = {"context_key": key, "count": count, "stability": stability}
    if count < MIN_SAMPLE_DEMOTE:
        return {**base, "usable": False, "adjustment": 0.0, "direction": "NONE",
                "reason": "INSUFFICIENT_CONTRACT_SAMPLE"}

    expectancy = float(cell.get("expectancy_R") or 0.0)
    wilson = cell.get("wilson_lower_bound")
    base.update({"expectancy_R": expectancy, "wilson_lower_bound": wilson})

    if expectancy < 0:
        adjustment = max(NEGATIVE_CAP, round(expectancy * 13.0, 4))
        return {**base, "usable": True, "adjustment": adjustment, "direction": "DEMOTE",
                "reason": "CONTRACT_NEGATIVE_EXPECTANCY"}
    if expectancy > 0:
        if count < MIN_SAMPLE_PROMOTE:
            return {**base, "usable": False, "adjustment": 0.0, "direction": "NONE",
                    "reason": "INSUFFICIENT_SAMPLE_FOR_PROMOTION"}
        if wilson is None or float(wilson) < PROMOTE_WILSON_MIN:
            return {**base, "usable": False, "adjustment": 0.0, "direction": "NONE",
                    "reason": "WIN_RATE_LOWER_BOUND_TOO_WEAK_FOR_PROMOTION"}
        if not stability["stable"]:
            return {**base, "usable": False, "adjustment": 0.0, "direction": "NONE",
                    "reason": "CONTRACT_UNSTABLE_ACROSS_PERIODS"}
        adjustment = min(POSITIVE_CAP, round(expectancy * 6.5, 4))
        return {**base, "usable": True, "adjustment": adjustment, "direction": "PROMOTE",
                "reason": "CONTRACT_POSITIVE_EXPECTANCY_VALIDATED"}
    return {**base, "usable": True, "adjustment": 0.0, "direction": "NONE",
            "reason": "CONTRACT_FLAT"}


def classify_contract_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    """Why did this contract win or lose? Evidence-justified only.

    ``row`` is a settled FoPaperTrade.as_dict() (or equivalent). Every branch
    below reads a field the desk actually captured at entry or exit -- never
    a plausible-sounding guess. When the underlying's own direction cannot be
    established (missing spot on either side), this returns UNKNOWN rather
    than assuming the contract, not the call, was at fault.
    """
    entry_spot = _f(row.get("entry_underlying_spot"))
    exit_spot = _f(row.get("exit_underlying_spot"))
    option_type = str(row.get("option_type") or "").upper()
    net_pnl = _f(row.get("net_pnl"))
    exit_reason = str(row.get("exit_reason") or "").upper()

    if exit_reason == "IV_CRUSH":
        return {"classification": CLASSIFICATION_IV_CRUSH, "underlying_direction": "UNKNOWN"}

    if entry_spot <= 0 or exit_spot <= 0 or option_type not in ("CE", "PE"):
        return {"classification": CLASSIFICATION_UNKNOWN, "underlying_direction": "UNKNOWN"}

    moved_up = exit_spot > entry_spot
    underlying_right = moved_up if option_type == "CE" else not moved_up
    direction = "RIGHT" if underlying_right else "WRONG"

    if underlying_right and net_pnl > 0:
        return {"classification": CLASSIFICATION_UNDERLYING_RIGHT_CONTRACT_RIGHT,
                "underlying_direction": direction}
    if not underlying_right and net_pnl <= 0:
        return {"classification": CLASSIFICATION_UNDERLYING_WRONG_CONTRACT_WRONG,
                "underlying_direction": direction}
    if not underlying_right and net_pnl > 0:
        return {"classification": CLASSIFICATION_UNDERLYING_WRONG_CONTRACT_OK_MECHANICS,
                "underlying_direction": direction}

    # underlying_right and net_pnl <= 0: the call was correct, the contract
    # still lost. Rank the most specific, evidence-backed reason available.
    spread_pct = row.get("entry_spread_pct")
    if spread_pct is not None and _f(spread_pct) >= WIDE_SPREAD_PCT:
        return {"classification": CLASSIFICATION_WIDE_SPREAD_COST, "underlying_direction": direction}

    entry_dte = row.get("entry_dte")
    if entry_dte is not None and int(_f(entry_dte)) <= SHORT_DTE_DAYS and exit_reason in ("MAX_HOLD", "EOD"):
        return {"classification": CLASSIFICATION_BAD_EXPIRY_CHOICE, "underlying_direction": direction}

    entry_oi = row.get("entry_oi")
    if entry_oi is not None and int(_f(entry_oi)) > 0 and int(_f(entry_oi)) < THIN_LIQUIDITY_OI:
        return {"classification": CLASSIFICATION_POOR_LIQUIDITY, "underlying_direction": direction}

    delta = row.get("entry_delta")
    if delta is not None:
        delta_abs = abs(_f(delta))
        if delta_abs > 0 and delta_abs < LOW_DELTA_ABS:
            return {"classification": CLASSIFICATION_DELTA_TOO_LOW, "underlying_direction": direction}
        if delta_abs >= HIGH_DELTA_ABS:
            return {"classification": CLASSIFICATION_DELTA_TOO_HIGH, "underlying_direction": direction}

    theta = row.get("entry_theta_per_day")
    quantity = _f(row.get("quantity"))
    if theta is not None and quantity > 0 and exit_reason in ("MAX_HOLD", "EOD"):
        try:
            opened = str(row.get("opened_at") or "")[:10]
            settled = str(row.get("settled_at") or "")[:10]
            from datetime import date
            held_days = max(1, (date.fromisoformat(settled) - date.fromisoformat(opened)).days) if opened and settled and opened != settled else 1
        except ValueError:
            held_days = 1
        theta_decay_estimate = abs(_f(theta)) * quantity * held_days
        if theta_decay_estimate > 0 and abs(net_pnl) > 0 and theta_decay_estimate >= abs(net_pnl) * 0.5:
            return {"classification": CLASSIFICATION_THETA_DAMAGE, "underlying_direction": direction}

    return {"classification": CLASSIFICATION_UNDERLYING_RIGHT_CONTRACT_POOR, "underlying_direction": direction}
