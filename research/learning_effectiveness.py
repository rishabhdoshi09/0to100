"""Does learning change decisions, and does it improve them? A reproducible, read-only report.

Run against a COPY of a runtime (never the live runtime: ``/Volumes`` is
refused). It reads the forward ledger, the learning-policy store and its
observation-claims ledger (with the outbox where the build has one), the
feature store and the challenger state, and answers, with independent sample
sizes:

* how many unique verified forward observations exist, settled versus pending;
* how many settled outcomes actually reached learning (claims / outbox), how
  many were applied, quarantined or lost;
* which decisions a learned policy actually changed (binding reason or rank),
  as opposed to an effect merely recorded on a decision another gate rejected;
* which policies are written but never consulted by selection, and why each
  cannot be promoted;
* feature labels by evidence lane; champion versus challenger.

It never writes, never changes a policy and never "shows progress": when
learning has insufficient evidence to change any decision, that is the result.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

SELECTION_STATUSES = {"ACTIVE", "ELIGIBLE", "EXPERIMENTAL"}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def policy_consumption(policies: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """For each policy: can product.evidence_policy_engine ever let it move a decision, and if not, why."""
    from product.evidence_policy_engine import HARD_REASON_CODES

    out = []
    for p in policies:
        pid = str(p.get("policy_id") or "")
        status = str(p.get("production_status") or "")
        reason = ""
        if pid.startswith("HIST_SETUP::"):
            reason = "HISTORY_FIRST_GATE_INPUT"          # read by the history-first system gate, not as a policy
        elif p.get("affects_selection") is False:
            reason = "IGNORED: affects_selection is false"
        elif status not in SELECTION_STATUSES:
            reason = f"IGNORED: status {status or 'missing'} is never consulted"
        elif str(p.get("dimension")) == "reason_code" and str(p.get("bucket")) in HARD_REASON_CODES:
            reason = "OBSERVATION ONLY: a hard gate is never weakened"
        elif str(p.get("confidence")) == "INSUFFICIENT_EVIDENCE" and status != "ACTIVE":
            reason = "NEUTRAL: insufficient evidence"
        else:
            reason = "CAN_AFFECT_SELECTION"
        out.append({"policy_id": pid, "status": status, "n": p.get("sample_size"),
                    "sessions": p.get("sessions_observed"), "expectancy_R": p.get("expectancy_R"),
                    "edge_vs_baseline_R": p.get("expectancy_difference_R"),
                    "source": p.get("evidence_source"), "consumption": reason})
    return out


def learning_reach(forward_rows: Sequence[Mapping[str, Any]], claims_db: Path) -> dict[str, Any]:
    """Of the settled independent counterfactual decisions, how many reached learning (a unit claim)?"""
    from product.forward_evidence import independent_decisions, is_forward_evidence

    settled = [r for r in independent_decisions(forward_rows)
               if not r.get("entered") and r.get("later_outcome") is not None and is_forward_evidence(r)]
    keys = set()
    states: dict[str, Counter] = defaultdict(Counter)
    outbox: dict[str, Any] = {"present": False}
    if claims_db.exists():
        conn = sqlite3.connect(f"file:{claims_db}?mode=ro", uri=True)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(observations)")}
            state_col = "state" if "state" in cols else "'LEGACY'"
            for scope, key, state in conn.execute(f"SELECT scope, observation_key, {state_col} FROM observations"):
                states[scope][state] += 1
                if scope == "COUNTERFACTUAL":
                    keys.add(key)
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='outbox'").fetchone():
                outbox = {"present": True, "by_state": dict(conn.execute(
                    "SELECT state, COUNT(*) FROM outbox GROUP BY state").fetchall()),
                    "feature_links": dict(Counter(
                        (str(res or "").split("feature=")[-1].split(":")[0] if "feature=" in str(res or "") else "-")
                        for (res,) in conn.execute("SELECT result FROM outbox WHERE state='DONE'")))}
        finally:
            conn.close()
    reached = 0
    for r in settled:
        day = str(r.get("market_timestamp") or (r.get("pit_proof") or {}).get("as_of") or "")[:10]
        if f"cf:{day}|{str(r.get('symbol') or '').upper()}|{r.get('reason_code') or 'REJECTED'}" in keys:
            reached += 1
    return {"settled_independent_counterfactuals": len(settled), "reached_learning_unit_claim": reached,
            "never_reached_learning": len(settled) - reached,
            "reach_rate": round(reached / len(settled), 4) if settled else None,
            "claims_by_scope_and_state": {k: dict(v) for k, v in sorted(states.items())},
            "outbox": outbox}


def feature_labels(db: Path) -> dict[str, Any]:
    if not db.exists():
        return {"present": False}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        kinds = {k: {"observations": n, "labelled": int(lab or 0)} for k, n, lab in conn.execute(
            "SELECT kind, COUNT(*), SUM(outcome IS NOT NULL) FROM observations GROUP BY kind")}
        lanes = Counter()
        for (meta,) in conn.execute("SELECT outcome_meta FROM observations WHERE outcome IS NOT NULL"):
            try:
                lanes[str((json.loads(meta) if meta else {}).get("evidence_class") or "UNSTATED")] += 1
            except ValueError:
                lanes["UNPARSEABLE"] += 1
    finally:
        conn.close()
    return {"present": True, "by_kind": kinds, "labels_by_evidence_lane": dict(lanes)}


def build_report(root: str | Path) -> dict[str, Any]:
    from product.forward_evidence import independent_decisions
    from product.learning_scorecard import learning_influence

    root = Path(root).resolve()
    if str(root).startswith("/Volumes/"):
        raise SystemExit(f"refusing {root}: copy the runtime first (never read the live runtime)")
    logs = root / "logs" if (root / "logs").exists() else root
    product = logs / "product"
    forward = _jsonl(product / "forward_evidence.jsonl")
    policies = list((json.loads((product / "learning_policies.json").read_text()) if
                     (product / "learning_policies.json").exists() else {}).get("policies") or [])
    independent = independent_decisions(forward)
    groups = Counter(str(r.get("group") or "") for r in independent)
    settled = sum(1 for r in independent if r.get("later_outcome") is not None)
    influence = learning_influence(policies, forward)
    consumption = policy_consumption(policies)
    can_affect = [c for c in consumption if c["consumption"] == "CAN_AFFECT_SELECTION"]
    challengers = {}
    card = product / "learning_scorecard.json"
    if card.exists():
        sc = json.loads(card.read_text())
        challengers = sc.get("challengers") or {}
        deployed_answer = (sc.get("learning_influence") or {}).get("decisions_by_policy_effect")
    else:
        deployed_answer = None
    blockers = []
    if not can_affect:
        blockers.append("No learned policy is in a status the selection engine consults with sufficient evidence "
                        "(every one is REJECTED, OBSERVING, hard-gate observation, or historical-only).")
    if influence["decisions_changed_by_learning"] == 0:
        blockers.append("No independent decision was changed by a learned policy (binding reason or rank).")
    taken = sum(1 for r in independent if r.get("entered"))
    if taken < 100:
        blockers.append(f"Only {taken} taken paper trade(s): forward-paper graduation needs >= 100 trades over "
                        ">= 40 sessions, judged by the interval, not the count.")
    ml = (challengers.get("ml_challenger") or {})
    if not ml.get("real_forward_n"):
        blockers.append("The ML challenger has no real forward observations; it cannot be compared with the "
                        "champion.")
    cc = (challengers.get("champion_challenger") or {})
    if not cc.get("with_forward_evidence"):
        blockers.append("No champion/challenger policy has forward evidence.")
    return {
        "forward_observations": {
            "ledger_rows": len(forward), "independent_decisions": len(independent),
            "copies_per_decision": round(len(forward) / len(independent), 2) if independent else None,
            "by_group": dict(groups), "settled": settled, "pending": len(independent) - settled,
            "missing_decision_id": sum(1 for r in forward if not str(r.get("decision_id") or "").strip()),
            "carrying_feature_decision_id": sum(1 for r in forward if r.get("feature_decision_id")),
            "provenance": dict(Counter(str(r.get("provenance")) for r in forward)),
        },
        "learning_reach": learning_reach(forward, product / "learning_policies.observations.sqlite3"),
        "decision_impact": influence,
        "deployed_scorecard_decisions_by_policy_effect": deployed_answer,
        "policies": {"total": len(policies), "by_status": dict(Counter(str(p.get("production_status")) for p in policies)),
                     "consumption": consumption, "can_affect_selection": len(can_affect)},
        "baseline_vs_learned": (
            "IDENTICAL: no learned policy is consulted by selection, so the learned policy set equals the baseline"
            if not can_affect else "COMPARE: see per-policy edge_vs_baseline_R with n and sessions"),
        "feature_store": feature_labels(logs / "feature_store.db"),
        "challengers": challengers,
        "cannot_promote_because": blockers,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="a COPY of the runtime root (containing logs/)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = build_report(a.root)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("forward_observations", "learning_reach", "cannot_promote_because")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
