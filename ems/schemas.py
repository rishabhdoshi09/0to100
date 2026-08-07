"""
📐 Broker-neutral execution schemas.

Immutable, fully-traceable records for the live order lifecycle. Every live record carries the
whole decision chain (cycle/snapshot/strategy/rules/card/allocation/intent) plus an idempotency
key, so no order can exist without provenance and no retry can double-submit.

Reuses the intelligence `TradeIntent` (broker-independent) as the input — this module never
imports a broker.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict

SCHEMA_VERSION = 1


def _id(kind: str, payload: dict) -> str:
    blob = json.dumps({"k": kind, **payload}, sort_keys=True, default=str).encode()
    return f"{kind}-{hashlib.sha256(blob).hexdigest()[:16]}"


# ── operating modes + live readiness (Phase 13) ──────────────────────────────────
OFF, RESEARCH_ONLY, SHADOW, PAPER_AUTO = "OFF", "RESEARCH_ONLY", "SHADOW", "PAPER_AUTO"
LIMITED_LIVE, GUARDED_LIVE, FULL_AUTO = "LIMITED_LIVE", "GUARDED_LIVE", "FULL_AUTO"
NO_NEW_ENTRIES, LIQUIDATE_ONLY, HALTED = "NO_NEW_ENTRIES", "LIQUIDATE_ONLY", "HALTED"
MODES = (OFF, RESEARCH_ONLY, SHADOW, PAPER_AUTO, LIMITED_LIVE, GUARDED_LIVE, FULL_AUTO,
         NO_NEW_ENTRIES, LIQUIDATE_ONLY, HALTED)
_LIVE_MODES = {LIMITED_LIVE, GUARDED_LIVE, FULL_AUTO}

# readiness states — never blurred
NOT_READY = "NOT_READY"
ARCHITECTURE_READY = "ARCHITECTURE_READY"
SIMULATOR_CERTIFIED = "SIMULATOR_CERTIFIED"
BROKER_CONNECTED = "BROKER_CONNECTED"
USER_ACTIVATED = "USER_ACTIVATED"

# risk decisions (Phase 9)
APPROVE, APPROVE_REDUCED, REJECT = "APPROVE", "APPROVE_REDUCED", "REJECT"
CANCEL_PENDING, REDUCE_POSITION, EXIT_POSITION = "CANCEL_PENDING", "REDUCE_POSITION", "EXIT_POSITION"
BLOCK_NEW_ENTRIES, RG_LIQUIDATE_ONLY, HALT = "BLOCK_NEW_ENTRIES", "LIQUIDATE_ONLY", "HALT"

# daily-loss / drawdown protection states (Phase 11)
NORMAL, CAUTION, REDUCE_ONLY = "NORMAL", "CAUTION", "REDUCE_ONLY"
CAP_NO_NEW_ENTRIES, CAP_LIQUIDATE_ONLY, CAP_HALTED = "NO_NEW_ENTRIES", "LIQUIDATE_ONLY", "HALTED"


def is_live_mode(mode: str) -> bool:
    return mode in _LIVE_MODES


# ── owner Operating Envelope (Phase 12) — user-owned, checksummed ────────────────

@dataclass(frozen=True)
class OperatingEnvelope:
    broker_account: str
    max_live_capital: float
    approved_families: tuple = ()
    approved_symbols: tuple = ()               # empty ⇒ any (still capped by other limits)
    approved_exchanges: tuple = ("NSE",)
    approved_product_types: tuple = ("CNC",)
    approved_order_types: tuple = ("LIMIT", "MARKET")
    max_risk_per_trade_pct: float = 0.01
    max_portfolio_risk_pct: float = 0.05
    max_positions: int = 5
    daily_loss_limit: float = 0.0
    weekly_loss_limit: float = 0.0
    drawdown_limit_pct: float = 0.10
    emergency_policy: str = "exit_at_market"
    activation_date: str = ""
    review_date: str = ""
    approved_by: str = ""                      # a USER id — required for live
    schema_version: int = SCHEMA_VERSION
    checksum: str = ""

    def _identity(self):
        d = asdict(self); d.pop("checksum", None); return d

    def compute_checksum(self) -> str:
        return _id("env", self._identity())

    def is_user_approved(self) -> bool:
        return bool(self.approved_by) and self.checksum == self.compute_checksum()

    def allows(self, *, family: str, symbol: str, capital: float, product: str = "CNC",
               order_type: str = "LIMIT") -> tuple:
        if not self.is_user_approved():
            return False, "envelope not user-approved"
        if self.approved_families and family not in self.approved_families:
            return False, f"family {family} not in approved envelope"
        if self.approved_symbols and symbol not in self.approved_symbols:
            return False, f"symbol {symbol} not in approved envelope"
        if capital > self.max_live_capital:
            return False, "capital exceeds envelope ceiling"
        if product not in self.approved_product_types:
            return False, f"product {product} not approved"
        if order_type not in self.approved_order_types:
            return False, f"order type {order_type} not approved"
        return True, "ok"

    def as_dict(self):
        return asdict(self)


def approve_envelope(env: OperatingEnvelope, *, actor: str) -> OperatingEnvelope:
    """USER-only approval: stamps approver + checksum. Refuses a non-user actor. The system can
    never infer approval from UI navigation or an env var — only this call, by a user."""
    import dataclasses
    if actor != "user":
        raise PermissionError("only a user may approve an Operating Envelope")
    e = dataclasses.replace(env, approved_by="user")
    return dataclasses.replace(e, checksum=e.compute_checksum())


# ── execution records ────────────────────────────────────────────────────────────

@dataclass
class _Traced:
    idempotency_key: str = ""
    correlation_id: str = ""
    causation_id: str = ""
    cycle_id: str = ""
    snapshot_id: str = ""
    strategy_id: str = ""
    strategy_version: int = 0
    rules_hash: str = ""
    card_id: str = ""
    allocation_id: str = ""
    intent_id: str = ""
    broker_account: str = ""
    broker_adapter: str = ""
    config_hash: str = ""
    mode: str = ""
    envelope_checksum: str = ""


@dataclass
class RiskDecision(_Traced):
    decision: str = REJECT
    approved_qty: int = 0
    requested_qty: int = 0
    reason: str = ""
    limit_code: str = ""
    limits_version: str = "v1"

    def as_dict(self): return asdict(self)


@dataclass
class ExecutionPlan(_Traced):
    symbol: str = ""
    exchange: str = "NSE"
    product: str = "CNC"
    side: str = "BUY"
    qty: int = 0
    order_type: str = "LIMIT"
    limit_price: float = 0.0
    slippage_tolerance_bps: float = 30.0
    validity: str = "DAY"
    stop_price: float = 0.0
    target_price: float = 0.0
    max_retries: int = 1
    expected_cost: float = 0.0
    expected_risk: float = 0.0
    plan_id: str = ""

    def frozen_id(self) -> str:
        return _id("plan", {"idem": self.idempotency_key, "sym": self.symbol,
                            "qty": self.qty, "type": self.order_type, "lp": self.limit_price})

    def as_dict(self): return asdict(self)


@dataclass
class OrderStateRecord(_Traced):
    plan_id: str = ""
    broker_order_id: str = ""
    state: str = "INTENT_RECEIVED"
    symbol: str = ""
    side: str = "BUY"
    requested_qty: int = 0
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    history: list = field(default_factory=list)   # append-only (state, ts, note)
    created_ts: float = field(default_factory=lambda: time.time())

    def as_dict(self): return asdict(self)


@dataclass
class FillRecord(_Traced):
    broker_order_id: str = ""
    fill_id: str = ""
    qty: int = 0
    price: float = 0.0
    fees: float = 0.0
    ts: str = ""

    def as_dict(self): return asdict(self)


@dataclass
class PositionRecord:
    strategy_id: str
    symbol: str
    qty: int
    avg_price: float
    stop_price: float = 0.0
    target_price: float = 0.0
    protected: bool = False
    protection_order_id: str = ""
    opened_ts: str = ""

    def as_dict(self): return asdict(self)


@dataclass
class ProtectionPlan(_Traced):
    symbol: str = ""
    qty: int = 0
    stop_price: float = 0.0
    target_price: float = 0.0
    mechanism: str = "STOP"                     # broker capability-honest; not "OCO" unless atomic

    def as_dict(self): return asdict(self)


@dataclass
class ProtectionStatus:
    symbol: str
    protected: bool
    broker_verified: bool
    protection_order_id: str = ""
    reason: str = ""

    def as_dict(self): return asdict(self)


@dataclass
class ReconciliationReport:
    findings: list = field(default_factory=list)  # (classification, detail)
    critical: bool = False

    def add(self, classification, detail=""):
        self.findings.append((classification, detail))
        if classification in ("QUANTITY_MISMATCH", "PROTECTION_MISMATCH", "CRITICAL_CONFLICT",
                              "FUNDS_MISMATCH"):
            self.critical = True

    def as_dict(self): return {"findings": self.findings, "critical": self.critical}


@dataclass
class ExecutionIncident:
    severity: str        # INFO/WARNING/HIGH/CRITICAL
    code: str
    detail: str
    resolved: bool = False
    ts: str = ""

    def as_dict(self): return asdict(self)
