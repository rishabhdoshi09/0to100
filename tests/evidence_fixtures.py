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
    # The data a valid report measured ends with (at most) its generation day.
    last_session = str(stamp)[:10] if str(stamp)[:4].isdigit() else "2026-11-18"
    prov = {"generated_at": stamp, "report_schema": SB.REPORT_SCHEMA,
            "eval_spec": SB._eval_spec_tag(HORIZON_SESSIONS), "horizon_sessions": HORIZON_SESSIONS,
            "family_method": SB.FAMILY_METHOD, "code_sha": "test-sha", "universe_sha256": "u" * 64,
            "data_first_session": "2025-11-03", "data_last_session": last_session,
            "exact_horizon_check": True,
            "members_sha256": "m" * 64, "family_size": 44}
    binding = {k: prov.get(k) for k in alpha_ledger.BINDING_FIELDS}
    prov["alpha_accounting"] = alpha_ledger.reserve(SB.gate_program(), f"test-{uuid.uuid4().hex}",
                                                    family_size=44, members_sha256="m" * 64, binding=binding)
    # A finalised look whose result digest the snapshot recomputed from its rows (C-2).
    digest = uuid.uuid4().hex * 2
    alpha_ledger.finalize(prov["alpha_accounting"]["look_id"], digest)
    prov["result_sha256"] = digest
    prov["result_sha256_recomputed"] = digest
    prov.update(over)
    return {k: v for k, v in prov.items() if v is not None}


def rulebook_outcome(net_r: float, outcome: str = "FLAT", return_pct: float | None = None) -> dict:
    """A counterfactual outcome as core.outcome_resolver.counterfactual_outcome returns it."""
    from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS, spec_tag
    return {"eval_spec": spec_tag(), "horizon_sessions": OUTCOME_HORIZON_SESSIONS, "not_pnl": True,
            "outcome": outcome, "gross_r": net_r + 0.05, "net_r": net_r, "cost_r": 0.05,
            "return_pct": return_pct if return_pct is not None else net_r * 6.0}
