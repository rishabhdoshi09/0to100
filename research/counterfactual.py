"""
⚖️ Counterfactual gate attribution — which filters EARN vs COST money?

The single most underexploited asset QuantTerm owns. The autopilot has ~14
gates and the scanner a stack of demotes; nobody knows which of them make money
and which quietly tax it. A gate that rejects trades which would have WON is a
silent alpha leak. You can only ask "what would the rejected trade have done?"
if you logged the rejection with a reference price — which decision_journal
already does (TAKEN and REJECTED, each with entry/stop and a resolved outcome).
Competitors have no rejected-trade ledger, so they structurally cannot do this.

Method (causal-inference-lite, honest about confounding):
  • For each rejection reason (a gate), take the R-multiples of the trades it
    REJECTED (what would have happened had we taken them).
  • mean_reject_R > 0 and significant → the gate rejected WINNERS → it is
    COSTING money (leaving it on the table) → consider loosening.
  • mean_reject_R < 0 and significant → the gate rejected LOSERS → it is
    EARNING its keep → doing its job.
  • Opportunity cost vs. what we DID trade (the taken baseline) is reported too.
  • Every gate is tested (two-sided t) and the batch is FDR-corrected through
    the Research OS, so a "costly gate" must be statistically real, not one of
    fourteen coin-flips. Thin gates stay "insufficient evidence", never a claim.

Caveat kept explicit: rejected trades differ from taken trades in the very way
that got them rejected — this is association, surfaced for a human to act on,
not a randomized experiment. It flags gates worth investigating; it never auto-
removes a safety gate.

Pure functions over {gate: R-array}; the I/O layer reads decisions.db. Fail-open.
"""
from __future__ import annotations

import os as _os

import numpy as np
from scipy.stats import t as _student_t

_MIN_N = int(_os.getenv("QT_CF_MIN_N", "30") or 30)        # per-gate floor
_ALPHA = float(_os.getenv("QT_CF_ALPHA", "0.05") or 0.05)


def gate_attribution(rejected_by_gate: dict, taken_r=None,
                     min_n: int = _MIN_N, alpha: float = _ALPHA) -> list[dict]:
    """Per-gate attribution from {gate_name: [R-multiples of trades it rejected]}
    and (optionally) `taken_r` (R-multiples of trades we actually took, for the
    opportunity-cost baseline). Returns findings that survive FDR, worst leak
    first. Each: gate, n, mean_reject_r, opp_cost_r, p_value, qvalue, verdict
    (COSTING | EARNING), insight."""
    from research.harness import expectancy_stats, benjamini_hochberg

    taken = np.asarray(taken_r, dtype=float) if taken_r is not None else None
    taken = taken[~np.isnan(taken)] if taken is not None else None
    taken_mean = float(taken.mean()) if taken is not None and taken.size else 0.0

    tests = []
    for gate, rs in rejected_by_gate.items():
        x = np.asarray(rs, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < min_n:
            continue
        s = expectancy_stats(x)
        # two-sided p that the rejected-set mean differs from zero
        if s["std_r"] > 0 and s["n"] > 1:
            p_two = float(2.0 * _student_t.sf(abs(s["t_stat"]), df=s["n"] - 1))
        else:
            p_two = 1.0
        tests.append({"gate": str(gate), "n": s["n"],
                      "mean_reject_r": round(s["mean_r"], 4),
                      "opp_cost_r": round(s["mean_r"] - taken_mean, 4),
                      "p_value": p_two})
    if not tests:
        return []
    bh = benjamini_hochberg([t["p_value"] for t in tests], alpha=alpha)
    findings = []
    for t, keep, q in zip(tests, bh["rejected"], bh["qvalues"]):
        if not keep:
            continue
        t["qvalue"] = round(float(q), 4)
        if t["mean_reject_r"] > 0:
            t["verdict"] = "COSTING"
            t["insight"] = (f"Gate '{t['gate']}' rejected {t['n']} trades that "
                            f"averaged {t['mean_reject_r']:+.2f}R — it's leaving "
                            f"money on the table. Worth loosening / reviewing.")
        else:
            t["verdict"] = "EARNING"
            t["insight"] = (f"Gate '{t['gate']}' filtered {t['n']} trades "
                            f"averaging {t['mean_reject_r']:+.2f}R — it's earning "
                            f"its keep (rejecting losers).")
        findings.append(t)
    # surface the money-LEAKS first (most positive rejected-R), then earners
    return sorted(findings, key=lambda t: t["mean_reject_r"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# I/O — read decisions.db, split taken vs rejected-by-gate, in R (fail-open)
# ══════════════════════════════════════════════════════════════════════════════

def _decision_r(entry_ref: float, stop_ref: float, outcome_pct: float) -> float | None:
    """Outcome as a cost-aware R-multiple — same convention as scan/live_edge.

    Retail evidence must be net of brokerage/STT/slippage. None when geometry
    is invalid.
    """
    from core.costs import outcome_to_net_r

    return outcome_to_net_r(entry_ref, stop_ref, outcome_pct, product="CNC", clip=(-1.5, 4.0))


def _load_decisions() -> tuple[dict, list]:
    """({gate: [R rejected]}, [R taken]) from decisions.db. Fail-open → ({}, [])."""
    try:
        from core.decision_journal import _conn
        c = _conn()
        try:
            rows = c.execute(
                "SELECT decision, reason, entry_ref, stop_ref, outcome_pct "
                "FROM decisions WHERE outcome_pct IS NOT NULL "
                "AND entry_ref > 0 AND stop_ref > 0").fetchall()
        finally:
            c.close()
    except Exception:
        return {}, []
    rejected: dict[str, list] = {}
    taken: list = []
    for row in rows:
        r = _decision_r(float(row["entry_ref"]), float(row["stop_ref"]),
                        float(row["outcome_pct"]))
        if r is None:
            continue
        if str(row["decision"]).upper() == "TAKEN":
            taken.append(r)
        else:
            gate = str(row["reason"] or "—").strip() or "—"
            rejected.setdefault(gate, []).append(r)
    return rejected, taken


def gate_attribution_report() -> list[dict]:
    """Live gate attribution from decisions.db, FDR-gated. Fail-open → []."""
    rejected, taken = _load_decisions()
    if not rejected:
        return []
    return gate_attribution(rejected, taken_r=taken)


def gate_directives(max_items: int = 2) -> list[dict]:
    """Brain-ready directives: a COSTING gate warns (it's leaking money), an
    EARNING gate informs (reassurance it's pulling its weight). Fail-open → []."""
    dirs: list[dict] = []
    report = gate_attribution_report()
    costing = [f for f in report if f["verdict"] == "COSTING"]
    earning = [f for f in report if f["verdict"] == "EARNING"]
    for f in costing[:max_items]:
        dirs.append({"severity": "warn", "text": f"⚖️ {f['insight']}"})
    if not costing and earning:
        f = max(earning, key=lambda t: abs(t["mean_reject_r"]))
        dirs.append({"severity": "info", "text": f"⚖️ {f['insight']}"})
    return dirs
