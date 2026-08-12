"""Canonical forward outcome ledger with explicit evidence_source labelling."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from research.forward_evidence.sources import PAPER_FORWARD, assert_known

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "logs" / "forward_evidence" / "forward_outcomes.jsonl"

# Repository-native exit / fill vocabulary (aligned with PaperBook + OMS)
FILL_OUTCOMES = (
    "FILLED", "PARTIAL", "NO_FILL", "EXPIRED", "STOPPED", "TARGET",
    "TIME_EXIT", "CANCELLED", "DATA_UNKNOWN", "MAX_HOLD", "GAP_STOP",
)


@dataclass
class ForwardOutcome:
    evidence_source: str
    decision_id: str
    intent_id: str = ""
    order_id: str = ""
    fill_id: str = ""
    policy_id: str = ""
    symbol: str = ""
    entry_time: str = ""
    entry_price: float = 0.0
    exit_time: str = ""
    exit_price: float = 0.0
    quantity: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    r_outcome: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    mae: float | None = None
    mfe: float | None = None
    holding_period: int = 0
    exit_reason: str = ""
    market_regime_context: dict = field(default_factory=dict)
    portfolio_context: dict = field(default_factory=dict)
    data_quality: str = ""
    outcome_status: str = "FILLED"
    cycle_id: str = ""
    outcome_id: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _outcome_id(decision_id: str, exit_time: str, evidence_source: str) -> str:
    raw = f"{decision_id}|{exit_time}|{evidence_source}"
    return "out-" + hashlib.sha1(raw.encode()).hexdigest()[:16]


def map_exit_reason(reason: str) -> str:
    r = str(reason or "").upper()
    return {
        "STOP": "STOPPED",
        "TARGET": "TARGET",
        "MAX_HOLD": "TIME_EXIT",
        "GAP_STOP": "GAP_STOP",
        "EXPIRED": "EXPIRED",
        "CANCELLED": "CANCELLED",
        "NO_FILL": "NO_FILL",
        "PARTIAL": "PARTIAL",
    }.get(r, r if r in FILL_OUTCOMES else "FILLED")


def record_from_closed_trade(
    trade: dict,
    *,
    decision_id: str = "",
    intent_id: str = "",
    order_id: str = "",
    evidence_source: str = PAPER_FORWARD,
    cycle_id: str = "",
    fees: float = 0.0,
    slippage: float = 0.0,
    mae: float | None = None,
    mfe: float | None = None,
    regime: dict | None = None,
    portfolio_context: dict | None = None,
    data_quality: str = "",
    path: Path | None = None,
) -> ForwardOutcome:
    src = assert_known(evidence_source)
    entry_t = str(trade.get("entry_date", ""))
    exit_t = str(trade.get("exit_date", ""))
    did = decision_id or f"legacy-{trade.get('strategy_id','')}-{trade.get('symbol','')}-{entry_t}"
    oid = _outcome_id(did, exit_t, src)
    out = ForwardOutcome(
        evidence_source=src,
        decision_id=did,
        intent_id=intent_id,
        order_id=order_id,
        fill_id=f"fill-{oid}",
        policy_id=str(trade.get("strategy_id", "")),
        symbol=str(trade.get("symbol", "")).upper(),
        entry_time=entry_t,
        entry_price=float(trade.get("entry_price", 0) or 0),
        exit_time=exit_t,
        exit_price=float(trade.get("exit_price", 0) or 0),
        quantity=int(trade.get("qty", 0) or 0),
        gross_pnl=float(trade.get("pnl", 0) or 0) + float(fees),
        net_pnl=float(trade.get("pnl", 0) or 0),
        r_outcome=float(trade.get("realized_R", 0) or 0),
        fees=float(fees),
        slippage=float(slippage),
        mae=mae,
        mfe=mfe,
        holding_period=0,
        exit_reason=str(trade.get("exit_reason", "")),
        market_regime_context=dict(regime or {}),
        portfolio_context=dict(portfolio_context or {}),
        data_quality=data_quality,
        outcome_status=map_exit_reason(str(trade.get("exit_reason", ""))),
        cycle_id=cycle_id,
        outcome_id=oid,
    )
    if entry_t and exit_t:
        try:
            import pandas as pd
            out.holding_period = int((pd.Timestamp(exit_t) - pd.Timestamp(entry_t)).days)
        except Exception:
            out.holding_period = 0
    append_outcome(out, path=path)
    return out


def append_outcome(out: ForwardOutcome, *, path: Path | None = None) -> None:
    p = Path(path) if path else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    if _has_outcome(out.outcome_id, p):
        return
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out.as_dict(), default=str) + "\n")


def _has_outcome(outcome_id: str, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if outcome_id in line and json.loads(line).get("outcome_id") == outcome_id:
                    return True
    except Exception:
        return False
    return False


def load_outcomes(
    *,
    path: Path | None = None,
    evidence_source: str | None = None,
    policy_id: str | None = None,
) -> list[dict]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if evidence_source and row.get("evidence_source") != evidence_source:
            continue
        if policy_id and row.get("policy_id") != policy_id:
            continue
        out.append(row)
    return out
