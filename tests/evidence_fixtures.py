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
            "family_method": SB.FAMILY_METHOD, "code_sha": "test-sha", "universe_sha256": "u" * 64,
            "data_first_session": "2025-11-03", "data_last_session": "2026-11-18",
            "members_sha256": "m" * 64, "family_size": 44}
    binding = {k: prov.get(k) for k in alpha_ledger.BINDING_FIELDS}
    prov["alpha_accounting"] = alpha_ledger.reserve(SB.gate_program(), f"test-{uuid.uuid4().hex}",
                                                    family_size=44, members_sha256="m" * 64, binding=binding)
    prov.update(over)
    return {k: v for k, v in prov.items() if v is not None}
