"""Shared test fixture: walk-forward provenance that meets the full current evidence standard.

Includes a REAL alpha-ledger reservation (in the test runtime), because the
gate verifies the budget attestation against the ledger.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone


def proven_provenance(generated_at="fresh", **over):
    import scan.signal_backtest as SB
    from core.evaluation_spec import HORIZON_SESSIONS
    from research import alpha_ledger

    stamp = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat() if generated_at == "fresh" else generated_at
    prov = {"generated_at": stamp, "report_schema": SB.REPORT_SCHEMA,
            "eval_spec": SB._eval_spec_tag(HORIZON_SESSIONS), "horizon_sessions": HORIZON_SESSIONS,
            "family_method": SB.FAMILY_METHOD, "code_sha": "test-sha",
            "alpha_accounting": alpha_ledger.reserve(SB.gate_program(), f"test-{uuid.uuid4().hex}")}
    prov.update(over)
    return {k: v for k, v in prov.items() if v is not None}
