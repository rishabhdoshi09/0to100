"""
🎛️ ems — broker-neutral Execution Management System, independent Risk Governor, deterministic
broker simulator, execution ledger, reconciliation, recovery and live preflight.

Simulator-certified this milestone; NO real broker is wired. Real live remains impossible without
a real BrokerAdapter, a user-approved OperatingEnvelope and a passing live preflight. The
intelligence package keeps zero broker imports — the EMS lives here, outside it, and only reads
the broker-independent TradeIntent.
"""
from ems import schemas
from ems import state_machine
from ems.broker import BrokerAdapter
from ems.simulator import SimBroker
from ems.risk_governor import RiskGovernor, RiskLimits, capital_protection_state
from ems.ledger import ExecutionLedger
from ems.ems import EMS, SubmissionRefused
from ems import preflight

__all__ = ["schemas", "state_machine", "BrokerAdapter", "SimBroker", "RiskGovernor",
           "RiskLimits", "capital_protection_state", "ExecutionLedger", "EMS",
           "SubmissionRefused", "preflight"]
