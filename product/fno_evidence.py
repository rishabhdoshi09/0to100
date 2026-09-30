"""Wires real, settled F&O paper trades into the SAME conditional-evidence
store and ranking engine the equity desk already uses.

Before this module: FoPaperPosition.context_key was read at candidate-open
time (``contract.get("context_key")``) but nothing upstream ever populated
it, so it was always empty in production. FoPaperTrade settlement never
called ``product.conditional_evidence.record_outcome``. Together these two
gaps meant no amount of real, settled F&O paper history could ever move F&O
candidate ranking -- the exact "disconnected path" this module closes.

Design choices, deliberately mirroring the equity desk rather than inventing
a parallel F&O-only learning stack:

- The evidence cell key is built with the SAME ``product.decision_chain
  .context_key`` bucketing the equity ranking engine reads
  (``product.decision_ranking.rank`` / ``product.conditional_evidence
  .ranking_evidence``), from the setup-scoring dict ``fo_setup.py`` already
  computes. No parallel bucketing scheme, no new ranking engine. An
  ``FNO_`` prefix on the setup label keeps F&O and equity evidence in
  separate cells despite sharing one store.
- A settled trade whose costs were not fully modeled, or whose entry/exit
  path was not fully observed (``production_evidence_eligible`` is already
  computed by fo_paper_runtime.py and was already going unused for this),
  is not trustworthy PAPER_FORWARD evidence. It is skipped entirely --
  never recorded under a different, more permissive label. Silence here is
  intentional: see product.evidence_class for why mixing evidence classes
  is how a system talks itself into being ready before it is.
- This module never touches the option CONTRACT (strike/expiry/IV/OI/
  Greeks) dimension of the trade. No historical NSE option-chain data
  source exists anywhere in this repository, live or historical, and none
  is fabricated here. It grades the same thing fo_setup.py's score already
  claims to grade: the underlying directional call.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.conditional_evidence import record_outcome
from product.decision_chain import (
    EvidenceUpdate,
    Outcome,
    confidence_bucket,
    context_key as _context_key,
    extension_bucket,
    volatility_bucket,
)
from product.evidence_class import PAPER_FORWARD


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0  # filters NaN


def _market_regime_bucket(nifty_alignment: float) -> str:
    if nifty_alignment >= 2.0:
        return "RISK_ON"
    if nifty_alignment <= -2.0:
        return "RISK_OFF"
    return "NEUTRAL"


def _sector_state_bucket(sector_strength: float) -> str:
    if sector_strength >= 2.0:
        return "LEADING"
    if sector_strength <= -2.0:
        return "LAGGING"
    return "NEUTRAL"


def fno_context_key(setup: Mapping[str, Any] | None) -> str:
    """Derive an evidence-cell key for an F&O directional setup.

    ``setup`` is the dict ``product.fo_setup`` already computes for a
    candidate (score, direction, components incl. sector_strength /
    nifty_alignment, breakout_distance_pct). Returns "" when there is not
    even a direction to key on -- callers must treat that as "cannot
    record evidence for this candidate", not as a valid cell.
    """
    setup = setup or {}
    direction = str(setup.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return ""
    components = setup.get("components") if isinstance(setup.get("components"), Mapping) else {}
    score = setup.get("score")
    return _context_key(
        setup=f"FNO_{direction}",
        market_regime=_market_regime_bucket(_f(components.get("nifty_alignment"))),
        sector_state=_sector_state_bucket(_f(components.get("sector_strength"))),
        volatility=volatility_bucket(_f(setup.get("atr_pct")) or None),
        extension=extension_bucket(_f(setup.get("breakout_distance_pct")) or None),
        confidence=confidence_bucket((_f(score) / 100.0) if score is not None else None),
    )


def record_fno_settlement(
    trade_row: Mapping[str, Any],
    *,
    context_key: str,
    path: str | None = None,
) -> EvidenceUpdate | None:
    """Fold one settled F&O paper trade into evidence, or skip it honestly.

    Returns None (never raises) rather than recording untrustworthy or
    unattributable evidence:
      - no context_key: the candidate that produced this trade could not be
        keyed to an evidence cell (e.g. no direction on the setup).
      - not production_evidence_eligible: costs were not fully modeled, or
        the entry/exit path was not fully observed. fo_paper_runtime.py
        already computes this flag; it was previously unused for evidence.
      - zero or missing risk (entry == stop, or no quantity): an R-multiple
        cannot be honestly computed.
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
    price_move_unit = entry if entry else 0.0
    mfe_R = (
        (_f(mfe_pct) / 100.0 * price_move_unit) / risk_per_unit
        if mfe_pct is not None and risk_per_unit > 0 else None
    )
    mae_R = (
        (_f(mae_pct) / 100.0 * price_move_unit) / risk_per_unit
        if mae_pct is not None and risk_per_unit > 0 else None
    )
    trade_id = str(trade_row.get("trade_id") or "")
    if not trade_id:
        return None
    outcome = Outcome(
        position_id=trade_id,
        paper_order_id=trade_id,
        paper_intent_id=trade_id,
        decision_id=trade_id,
        symbol=str(trade_row.get("underlying") or trade_row.get("option_symbol") or ""),
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
    )
