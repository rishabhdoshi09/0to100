"""Decision-time snapshot freeze — never recompute the original decision with future data."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "logs" / "forward_evidence" / "decision_snapshots.jsonl"


@dataclass
class DecisionSnapshot:
    decision_id: str
    policy_id: str
    policy_version: str
    config_hash: str
    timestamp: str
    symbol: str
    market_price: float
    features: dict = field(default_factory=dict)
    fundamentals_events_used: dict = field(default_factory=dict)
    available_at_state: str = ""
    market_state: dict = field(default_factory=dict)
    portfolio_state: dict = field(default_factory=dict)
    risk_state: dict = field(default_factory=dict)
    signal_score: float = 0.0
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    intended_quantity: int = 0
    evidence_state: str = ""
    data_snapshot_ids: dict = field(default_factory=dict)
    code_git_provenance: dict = field(default_factory=dict)
    evidence_source: str = "PAPER_FORWARD"
    cycle_id: str = ""
    intent_id: str = ""
    target_position_id: str = ""
    mode: str = "PAPER_AUTO"

    def as_dict(self) -> dict:
        return asdict(self)


def _decision_id(policy_id: str, symbol: str, timestamp: str, cycle_id: str, intent_id: str) -> str:
    raw = f"{policy_id}|{symbol}|{timestamp}|{cycle_id}|{intent_id}"
    return "dec-" + hashlib.sha1(raw.encode()).hexdigest()[:16]


def freeze_decision(
    *,
    policy_id: str,
    symbol: str,
    timestamp: str,
    entry: float,
    stop: float,
    target: float,
    intended_quantity: int = 0,
    policy_version: str = "1",
    config_hash: str = "",
    market_price: float | None = None,
    signal_score: float = 0.0,
    evidence_state: str = "",
    cycle_id: str = "",
    intent_id: str = "",
    target_position_id: str = "",
    data_snapshot_id: str = "",
    portfolio_state: dict | None = None,
    risk_state: dict | None = None,
    market_state: dict | None = None,
    features: dict | None = None,
    fundamentals_events_used: dict | None = None,
    available_at_state: str = "",
    git_sha: str = "",
    mode: str = "PAPER_AUTO",
    evidence_source: str = "PAPER_FORWARD",
    path: Path | None = None,
) -> DecisionSnapshot:
    snap = DecisionSnapshot(
        decision_id=_decision_id(policy_id, symbol, timestamp, cycle_id, intent_id),
        policy_id=policy_id,
        policy_version=str(policy_version),
        config_hash=config_hash or "",
        timestamp=timestamp,
        symbol=str(symbol).upper(),
        market_price=float(market_price if market_price is not None else entry),
        features=dict(features or {}),
        fundamentals_events_used=dict(fundamentals_events_used or {}),
        available_at_state=available_at_state,
        market_state=dict(market_state or {}),
        portfolio_state=dict(portfolio_state or {}),
        risk_state=dict(risk_state or {}),
        signal_score=float(signal_score),
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        intended_quantity=int(intended_quantity),
        evidence_state=evidence_state,
        data_snapshot_ids={"ohlcv": data_snapshot_id} if data_snapshot_id else {},
        code_git_provenance={"git_sha": git_sha} if git_sha else {},
        evidence_source=evidence_source,
        cycle_id=cycle_id,
        intent_id=intent_id,
        target_position_id=target_position_id,
        mode=mode,
    )
    append_snapshot(snap, path=path)
    return snap


def append_snapshot(snap: DecisionSnapshot, *, path: Path | None = None) -> None:
    p = Path(path) if path else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    # Idempotent: skip if same decision_id already present (restart safety)
    if _has_decision(snap.decision_id, p):
        return
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap.as_dict(), default=str) + "\n")


def _has_decision(decision_id: str, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if decision_id in line and json.loads(line).get("decision_id") == decision_id:
                    return True
    except Exception:
        return False
    return False


def load_snapshots(path: Path | None = None) -> list[dict]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def get_snapshot(decision_id: str, path: Path | None = None) -> dict | None:
    for row in load_snapshots(path):
        if row.get("decision_id") == decision_id:
            return row
    return None
