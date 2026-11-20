"""Rebuild counterfactual learning policies from independent observations.

Before ``learning_observations`` existed, every paper cycle's copy of a
rejected name was ingested as its own observation (owner's Mac 2026-11-18:
4,258 settled forward-ledger rows were 814 independent decisions;
REJECT::LOW_QUALITY_SETUP stored n=2,989 for 298 decisions). This module
recomputes the counterfactual-derived policies from the settled ledgers,
counting each (decision session, symbol, reason) once.

Dry run by default. ``apply=True`` backs the policy file up, claims every unit
in the observation ledger (so the live ingest treats them as already counted)
and then rewrites only the counterfactual-derived policies. Source ledgers are
read, never modified. Run it while no outcome_resolution job is running.

    python -m product.learning_policy_repair            # report only
    python -m product.learning_policy_repair --apply    # backup + rewrite
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from product import learning_observations as LO
from product import learning_policy_store as LPS

# Must mirror _paper_learning_loop_core.ingest_counterfactual.
_MAPPED = {
    "CORRECT_REJECTION": 0.40,
    "AVOIDED_LOSER": 0.40,
    "MISSED_WINNER": -0.40,
    "GOOD_WAIT": 0.20,
}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _comparable(cls: str, spec: Any) -> bool:
    """Mirror of the live ingest guard: a forward-outcome label counts only under the current rulebook."""
    from product.counterfactual_learning import FORWARD_OUTCOME_CLASSES, UNMEASURED, is_rulebook
    if cls == UNMEASURED:
        return False
    return cls not in FORWARD_OUTCOME_CLASSES or is_rulebook(spec)


def independent_units(forward_rows: Iterable[Mapping[str, Any]], cf_rows: Iterable[Mapping[str, Any]],
                      excluded: dict[str, int] | None = None):
    """{(day, symbol, reason): {"classification", "setup"}} — first settlement wins.

    Labels measured under another rulebook (the legacy 5-session close-to-close
    counterfactuals) are not comparable with taken trades and are skipped, as
    the live ingest skips them; ``excluded`` counts them.
    """
    units: dict[tuple[str, str, str], dict[str, str]] = {}
    excluded = excluded if excluded is not None else {}
    from product.forward_evidence import is_forward_evidence
    for row in forward_rows:
        if row.get("entered") or not row.get("later_outcome"):
            continue
        if not is_forward_evidence(row):
            excluded["simulated_provenance"] = excluded.get("simulated_provenance", 0) + 1
            continue
        cls = str(row.get("counterfactual_classification") or "")
        if not cls:
            continue
        if not _comparable(cls, (row.get("later_outcome") or {}).get("eval_spec")):
            excluded["forward_ledger"] = excluded.get("forward_ledger", 0) + 1
            continue
        day = str(row.get("market_timestamp") or (row.get("pit_proof") or {}).get("as_of") or "")[:10]
        key = (day, str(row.get("symbol") or "").upper(), str(row.get("reason_code") or "REJECTED"))
        if len(day) == 10 and key[1]:
            units.setdefault(key, {"classification": cls, "setup": str(row.get("setup") or "")})
    for row in cf_rows:
        cls = str(row.get("classification") or "")
        if not cls:
            continue
        if not _comparable(cls, row.get("eval_spec") or (row.get("outcome") or {}).get("eval_spec")):
            excluded["counterfactuals"] = excluded.get("counterfactuals", 0) + 1
            continue
        day = str(row.get("as_of") or "")[:10]
        key = (day, str(row.get("symbol") or "").upper(), str(row.get("reason_code") or "REJECTED"))
        if len(day) == 10 and key[1]:
            units.setdefault(key, {"classification": cls, "setup": str(row.get("setup") or "")})
    return units


def _observations(units) -> dict[str, list[tuple[str, str, float]]]:
    """policy_id -> [(observation_key, session, mapped_R)] exactly as the live ingest would."""
    per: dict[str, list[tuple[str, str, float]]] = {}
    for (day, symbol, reason), meta in units.items():
        key = f"cf:{day}|{symbol}|{reason}"
        cls = meta["classification"]
        per.setdefault(f"REJECT::{reason}", []).append((key, day, _MAPPED.get(cls, 0.0)))
        if cls == "MISSED_WINNER" and meta["setup"] and reason == "EVIDENCE_POLICY_BLOCK":
            per.setdefault(f"SETUP::{meta['setup']}", []).append((key, day, 0.40))
        if cls == "GOOD_WAIT":
            per.setdefault("ENTRY_QUALITY::GOOD_WAIT", []).append((key, day, 0.20))
    return per


def plan(*, forward_path: Path, cf_path: Path, policy_file: Path) -> dict[str, Any]:
    excluded: dict[str, int] = {}
    units = independent_units(_read_jsonl(forward_path), _read_jsonl(cf_path), excluded)
    per = _observations(units)
    store = LPS.load_policies(policy_file)
    existing = {str(p.get("policy_id")): dict(p) for p in store.get("policies") or []}
    # A counterfactual-owned policy with no comparable observation left (only
    # legacy-horizon evidence) is reset to n = 0 rather than kept stale.
    for policy_id, old in existing.items():
        if str(old.get("evidence_source") or "").startswith("counterfactual") and policy_id not in per:
            per[policy_id] = []
    changes = []
    for policy_id, obs in sorted(per.items()):
        old = existing.get(policy_id)
        if old is None or not str(old.get("evidence_source") or "").startswith("counterfactual"):
            continue  # only policies the counterfactual ingest owns
        values = [v for _k, _s, v in obs]
        n = len(values)
        mean = sum(values) / n if n else 0.0
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
        half = 1.96 * sd / math.sqrt(n) if sd is not None else None
        changes.append({
            "policy_id": policy_id,
            "stored_n": int(old.get("sample_size") or 0),
            "stored_expectancy_R": old.get("expectancy_R"),
            "stored_status": old.get("production_status"),
            "independent_n": n,
            "expectancy_R": round(mean, 4),
            "ci95_R": None if half is None else [round(mean - half, 4), round(mean + half, 4)],
            "sessions": len({s for _k, s, _v in obs}),
            "observations": obs,
        })
    return {
        "independent_units": len(units),
        "excluded_incompatible_eval_spec": excluded,
        "changes": changes,
        "forward_path": str(forward_path),
        "counterfactual_path": str(cf_path),
        "policy_file": str(policy_file),
    }


def apply_plan(report: Mapping[str, Any], *, backup_dir: Path | None = None) -> dict[str, Any]:
    policy_file = Path(report["policy_file"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_dir or policy_file.parent.parent / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{policy_file.stem}_{stamp}_pre_independent_rebuild.json"
    if policy_file.exists():
        shutil.copy2(policy_file, backup)
    ledger = LO.ledger_path_for(policy_file)
    for change in report["changes"]:
        for key, session, _v in change["observations"]:
            LO.claim(ledger, "COUNTERFACTUAL", key, session)
            LO.claim(ledger, change["policy_id"], key, session)
    store = LPS.load_policies(policy_file)
    existing = {str(p.get("policy_id")): dict(p) for p in store.get("policies") or []}
    written = []
    for change in report["changes"]:
        old = existing[change["policy_id"]]
        keep = {k: v for k, v in old.items() if k not in {
            "policy_id", "version", "dimension", "bucket", "sample_size", "expectancy_R",
            "baseline_R", "expectancy_difference_R", "shrunk_expectancy_R", "shrinkage_k",
            "confidence", "production_status", "evidence_source", "last_updated",
            "first_seen", "date_range", "regimes_tested", "sessions_observed", "live_locked",
        }}
        keep["affects_selection"] = old.get("affects_selection", True)
        keep["sessions_observed"] = LO.distinct_sessions(ledger, change["policy_id"])
        keep["rebuilt_from_independent_units"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "previous_sample_size": change["stored_n"],
            "previous_expectancy_R": change["stored_expectancy_R"],
            "ci95_R": change["ci95_R"],
            "backup": str(backup),
        }
        written.append(LPS.upsert_policy(
            policy_id=change["policy_id"],
            dimension=str(old.get("dimension") or ""),
            bucket=str(old.get("bucket") or ""),
            sample_size=change["independent_n"],
            expectancy_R=change["expectancy_R"],
            baseline_R=float(old.get("baseline_R") or 0.0),
            source=str(old.get("evidence_source") or "counterfactual_not_pnl"),
            path=policy_file,
            extra=keep,
        ))
    return {"backup": str(backup), "rewritten": [w["policy_id"] for w in written], "policies": written}


def main(argv: list[str] | None = None) -> int:
    from product.counterfactual_learning import ledger_path as cf_ledger
    from product.forward_evidence import ledger_path as fwd_ledger

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="back up and rewrite the policies")
    ap.add_argument("--policy-file", default=None)
    args = ap.parse_args(argv)
    report = plan(
        forward_path=fwd_ledger(),
        cf_path=cf_ledger(),
        policy_file=LPS.policy_path(args.policy_file),
    )
    print(f"independent counterfactual decisions: {report['independent_units']}")
    for c in report["changes"]:
        print(
            f"{c['policy_id']:<42} n {c['stored_n']:>5} -> {c['independent_n']:<5} "
            f"R {c['stored_expectancy_R']} -> {c['expectancy_R']} ci95={c['ci95_R']} "
            f"sessions={c['sessions']} (was {c['stored_status']})"
        )
    if not args.apply:
        print("dry run: nothing written (pass --apply to back up and rewrite)")
        return 0
    out = apply_plan(report)
    print(f"backup: {out['backup']}")
    for p in out["policies"]:
        print(f"rewrote {p['policy_id']}: n={p['sample_size']} status={p['production_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
