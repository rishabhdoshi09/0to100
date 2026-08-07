"""
🛡️ Portfolio & risk gate (Phase J) — strategies REQUEST risk; this layer grants or denies it.

A TradeIntent must clear portfolio-level checks before it can reach the paper book. Brain 2
cannot override this. Every denial returns a specific reason code + the relevant limit,
current exposure and requested exposure, so the block is auditable.

Per-trade / per-name / total-open-risk / max-positions are enforced by the PaperBook itself on
open; here we add the PORTFOLIO checks the book can't see: family cap, correlation-cluster cap,
duplicate-symbol exposure, regime stand-down, and data/system health.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    ok: bool
    reason_code: str = ""
    detail: str = ""
    limit: float = 0.0
    current: float = 0.0
    requested: float = 0.0


def check(intent, *, family: str, book, family_risk: dict, cluster_risk: dict,
          cluster_of: str, cfg, regime: str = "RISK_ON", data_ok: bool = True,
          reconciled: bool = True) -> GateResult:
    """Return a GateResult. `family_risk`/`cluster_risk` are running tallies (pct) for THIS
    cycle; `cluster_of` is this intent's correlation cluster id (or '')."""
    if not data_ok:
        return GateResult(False, "NO_DATA", "no validated data — no new risk")
    if not reconciled:
        return GateResult(False, "UNRECONCILED", "state not reconciled after restart — "
                          "refusing new risk until resolved")
    if regime == "RISK_OFF":
        return GateResult(False, "REGIME_STANDDOWN", "regime is RISK_OFF — no new entries")

    req = float(intent.intended_risk_pct)
    # duplicate economic exposure: already long this symbol
    if any(p.symbol == intent.symbol for p in book.open.values()):
        return GateResult(False, "DUPLICATE_SYMBOL",
                          f"already have an open position in {intent.symbol}")
    # family cap
    fam = family or ""
    cur_fam = family_risk.get(fam, 0.0)
    if cur_fam + req > cfg.max_family_risk_pct + 1e-9:
        return GateResult(False, "FAMILY_CAP", f"family {fam} risk cap",
                          cfg.max_family_risk_pct, cur_fam, req)
    # correlation-cluster cap
    if cluster_of:
        cur_cl = cluster_risk.get(cluster_of, 0.0)
        if cur_cl + req > cfg.max_cluster_risk_pct + 1e-9:
            return GateResult(False, "CLUSTER_CAP", f"correlation cluster {cluster_of} cap",
                              cfg.max_cluster_risk_pct, cur_cl, req)
    # position-count limit (book also guards this, but block early with a clear reason)
    if len(book.open) >= book.max_positions:
        return GateResult(False, "MAX_POSITIONS", "max open positions",
                          float(book.max_positions), float(len(book.open)), 1.0)
    return GateResult(True, "OK")
