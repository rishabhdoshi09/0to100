"""Information-stage discovery from newly verified Gauntlet fields."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from time import monotonic

from core.runtime_paths import logs_path
from gauntlet import frontier
from product.learning_observations import writer_lock
from research import registry
from research.autonomy import discovery_programme as previous
from research.autonomy import outcome_discovery as base

PROGRAMME = "nse-frontier-information-v1"
from research.autonomy.schedules import OUTCOME_FRONTIER_PREFIX as KEY_PREFIX
BUDGET = 2
ROW_LIMIT = 170_000
# Adjust against the three related v1 diagnostics plus two information tests.
CRITERIA = {"economic_lower_bound": {">": 0}, "partial_rank_given_momentum": {">": 0}}
SPECS = (
    ("delivery_change", 3, "A rising delivery fraction may identify persistent cash participation after a short change in ownership demand.",
     "Prior absolute-turnover test could not distinguish durable ownership demand from activity."),
    ("overnight_share", 20, "Stocks whose price discovery concentrates overnight may reflect persistent information rather than transient regular-session flows.",
     "Prior next-open gap result was negligible; the contrast between overnight and intraday formation returns tests a different mechanism and horizon."),
)


def _entries():
    entries = []
    for row in registry.list_experiments():
        window = json.loads(row["data_window"] or "{}")
        if window.get("programme") == PROGRAMME:
            entries.append((row, window))
    entries.sort(key=lambda item: item[1]["slot"])
    if (len(entries) > BUDGET or [w["slot"] for _, w in entries] != list(range(1, len(entries)+1))
            or len({w["feature"] for _, w in entries}) != len(entries)
            or any(r["result"] is None for r, _ in entries[:-1])):
        raise ValueError("Conflicting frontier budget records")
    return entries


def scheduler_key():
    entries = _entries()
    slot = len(entries) + 1 - int(bool(entries and not entries[-1][0]["result"]))
    return KEY_PREFIX + str(slot) if slot <= BUDGET else None


def _measure(db, days, feature, horizon):
    ordinals = {d: i for i, d in enumerate(days)}
    grouped = defaultdict(list)
    count = 0
    for count, (symbol, day, features, outcomes) in enumerate(db.execute(
        "SELECT symbol,day,feature,outcome FROM frontier_observations ORDER BY day,symbol LIMIT ?", (ROW_LIMIT+1,),
    ), 1):
        if count > ROW_LIMIT or day >= base.END or day not in ordinals:
            raise ValueError("Frontier row escapes scope")
        f, o = json.loads(features), json.loads(outcomes)
        if ordinals[day]+horizon >= ordinals[base.END]:
            continue
        if f[feature] is not None:
            grouped[day].append((f[feature], symbol, o.get(str(horizon)), f["momentum"], None))
    metrics = previous.summarize(grouped, ordinals, count, horizon, trials=5, incremental=True)
    metrics["without_scanner_signals"] = None  # eligibility uses archives alone
    return metrics


def run(checkpoint: Path, archive: Path, slot: int):
    if not 1 <= slot <= BUDGET:
        raise ValueError("Frontier slot outside budget")
    source_path = base._isolated_checkpoint(checkpoint)
    prior_entries = previous.rows()
    if len([row for row, _ in prior_entries if row["result"]]) != previous.BUDGET:
        raise ValueError("Prior programme must be complete before advancing the frontier")
    source_identity = base._identity(source_path)
    if source_identity[-1] > base.WINDOW["budget"]["input_bytes"]:
        raise ValueError("Frontier source exceeds bounded checkpoint size")
    source_digest = base._digest(source_path, source_identity)
    if any(json.loads(row["result"])["source_sha256"] != source_digest for row, _ in prior_entries):
        raise ValueError("Frontier requires the previously registered Gauntlet archive")
    path, days, report = frontier.build(checkpoint, archive)
    if base._digest(source_path, source_identity) != source_digest:
        raise ValueError("Gauntlet base changed during frontier build")
    with writer_lock(logs_path("outcome_discovery")):
        entries = _entries()
        existing = next(((r, w) for r, w in entries if w["slot"] == slot), None)
        if existing and existing[0]["result"]:
            return {"decision": "DISCOVERY_ALREADY_COMPLETED", **json.loads(existing[0]["result"])}
        if scheduler_key() != KEY_PREFIX + str(slot):
            raise ValueError("Frontier requires the next unspent slot")
        source = base._identity(path)
        source_hash = base._digest(path, source)
        if any(json.loads(r["result"])["source_sha256"] != source_hash for r, _ in entries if r["result"]):
            raise ValueError("Frontier programmes cannot mix observation generations")
        code = hashlib.sha256(Path(__file__).read_bytes() + Path(previous.__file__).read_bytes()).hexdigest()
        known = {window["feature"] for _, window in entries}
        prior_results = {w["family"]: json.loads(r["result"]) for r, w in previous.rows()}
        # Prefer an ownership measure after weak absolute turnover, and an attribution
        # measure after weak executable continuation. No parameter or horizon search.
        priority = {
            "delivery_change": 2 + int(prior_results["liquidity"]["status"] == "REJECTED"),
            "overnight_share": 1 + int(prior_results["continuation"]["status"] == "REJECTED"),
        }
        if existing:
            row, window = existing
            hid = row["hypothesis_id"]
            feature, horizon = window["feature"], window["horizon"]
            if row["code_hash"] != code or window["source_sha256"] != source_hash:
                raise ValueError("Pending frontier input changed")
        else:
            feature, horizon, rationale, lineage = max(
                (s for s in SPECS if s[0] not in known), key=lambda s: priority[s[0]],
            )
            window = {
                "stage": "DISCOVERY", "information_stage": "A", "programme": PROGRAMME,
                "feature": feature, "slot": slot, "horizon": horizon, "motivation": lineage,
                "start": days[0], "end_exclusive": base.END, "source_sha256": source_hash,
                "max_possible_blocks": (len(days)-1)//horizon, "row_budget": ROW_LIMIT,
                "cumulative_looks": BUDGET, "related_trials_adjusted": 5, "coverage": previous.MIN_COVERAGE,
                "benchmark": "same-date eligible basket; diagnostic delivery costs 32/52 bp",
            }
            hid = registry.register_hypothesis(f"Frontier · {feature}", CRITERIA, window, rationale, code_hash=code)
        registry.start_discovery(hid)
        started = monotonic()
        failure = None
        try:
            with closing(sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
                metrics = _measure(db, days, feature, horizon)
        except (ArithmeticError, KeyError, ValueError) as exc:
            failure, metrics = str(exc), {}
        if base._digest(path, source) != source_hash:
            raise ValueError("Frontier checkpoint changed during measurement")
        info = metrics.get("excess_inference", {})
        status = ("PROMISING" if registry.meets_criteria(metrics, CRITERIA)
                  else "REJECTED" if info.get("status") == "OK" and info["mean"] <= 0
                  else "INCONCLUSIVE")
        result = {
            "hypothesis_id": hid, "status": status, "source_sha256": source_hash, "code_sha256": code,
            "evidence_class": "HISTORICAL_REPLAY", "promotion_eligible": False,
            "specification": window, "metrics": metrics, "failure": failure, "budget_used": slot, "budget_total": BUDGET,
            "elapsed_seconds": round(monotonic()-started, 3),
            "bridged_share_actions": report["bridged_share_actions"],
            "economic_stage": "PENDING_SEPARATE_PREREGISTRATION" if status == "PROMISING" else "NOT_QUALIFIED",
            "next": "SELECT_NEXT_FEASIBLE_HYPOTHESIS" if slot < BUDGET else "AWAIT_DATED_MEMBERSHIP_AND_ADDITIONAL_REGIMES",
        }
        registry.record_discovery_result(hid, result)
        return {"decision": "DISCOVERY_FAILED" if failure is not None else "DISCOVERY_COMPLETE", **result}


def configured_archive():
    path = os.environ.get("QT_FRONTIER_ARCHIVE")
    return Path(path) if path else None
