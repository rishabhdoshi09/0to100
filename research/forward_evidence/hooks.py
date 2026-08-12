"""Hooks into the institutional intelligence cycle — thin glue, no parallel OMS."""
from __future__ import annotations

from typing import Any

from research.forward_evidence.decision_snapshot import freeze_decision
from research.forward_evidence.outcome_ledger import record_from_closed_trade
from research.forward_evidence.policy_allowlist import PaperPolicyAllowlist, seed_default_allowlist
from research.forward_evidence.sources import PAPER_FORWARD
from research.forward_evidence import memory_bridge as MB


def may_open_paper(policy_id: str, *, family: str = "", allowlist: PaperPolicyAllowlist | None = None) -> bool:
    al = allowlist or seed_default_allowlist()
    return al.may_paper_trade(policy_id, family=family)


def on_paper_intent_opened(
    *,
    policy_id: str,
    symbol: str,
    timestamp: str,
    entry: float,
    stop: float,
    target: float,
    qty: int,
    policy_version: str = "1",
    config_hash: str = "",
    cycle_id: str = "",
    intent_id: str = "",
    target_position_id: str = "",
    data_snapshot_id: str = "",
    evidence_state: str = "",
    mode: str = "PAPER_AUTO",
    portfolio_state: dict | None = None,
    risk_state: dict | None = None,
) -> dict:
    snap = freeze_decision(
        policy_id=policy_id,
        symbol=symbol,
        timestamp=timestamp,
        entry=entry,
        stop=stop,
        target=target,
        intended_quantity=qty,
        policy_version=policy_version,
        config_hash=config_hash,
        market_price=entry,
        cycle_id=cycle_id,
        intent_id=intent_id,
        target_position_id=target_position_id,
        data_snapshot_id=data_snapshot_id,
        evidence_state=evidence_state,
        mode=mode,
        evidence_source=PAPER_FORWARD,
        portfolio_state=portfolio_state,
        risk_state=risk_state,
    )
    return snap.as_dict()


def on_paper_trade_closed(
    trade: dict | Any,
    *,
    decision_id: str = "",
    intent_id: str = "",
    cycle_id: str = "",
    update_memory: bool = True,
) -> dict:
    td = trade.as_dict() if hasattr(trade, "as_dict") else dict(trade)
    # Resolve decision_id from snapshots if not provided
    if not decision_id:
        from research.forward_evidence.decision_snapshot import load_snapshots
        for s in reversed(load_snapshots()):
            if (s.get("policy_id") == td.get("strategy_id")
                    and s.get("symbol") == str(td.get("symbol", "")).upper()
                    and s.get("timestamp") == td.get("entry_date")):
                decision_id = s["decision_id"]
                intent_id = intent_id or s.get("intent_id", "")
                break
    out = record_from_closed_trade(
        td,
        decision_id=decision_id,
        intent_id=intent_id,
        cycle_id=cycle_id,
        evidence_source=PAPER_FORWARD,
    )
    if update_memory:
        try:
            MB.remember_forward_policy(str(td.get("strategy_id", "")))
        except Exception:
            pass
    return out.as_dict()
