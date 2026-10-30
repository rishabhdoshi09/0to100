"""Offline, research-only comparison gate for external Harness2 workflow trials.

This script does NOT run agents, invoke Google Harness2, connect to market data,
mutate trading policies, or promote anything. Input measurements are externally
supplied and unverified. No QuantTerm production modules are imported.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MIN_PAIRED_TASKS = 30
MAX_INPUT_BYTES = 2_000_000
MAX_CANDIDATES = 16
MAX_TASKS = 10_000
MAX_P95_LATENCY_RATIO = 1.25
ALLOWED_COMPONENTS = frozenset({"system_prompt", "tool_guidance", "memory", "skills"})
ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class InvalidExperiment(ValueError):
    """A report is invalid and must not produce a decision."""


def _valid_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise InvalidExperiment(f"{label} must be 1-80 letters, numbers, _, ., or -")
    return value


def _outcomes(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not (1 <= len(value) <= MAX_TASKS):
        raise InvalidExperiment(f"{label}.outcomes must be 1..{MAX_TASKS} observations")
    observations: dict[str, dict[str, Any]] = {}
    for row in value:
        if not isinstance(row, dict):
            raise InvalidExperiment(f"{label}.outcomes must contain objects")
        task_id = _valid_id(row.get("task_id"), f"{label}.task_id")
        if task_id in observations:
            raise InvalidExperiment(f"{label} duplicates task_id {task_id}")
        if type(row.get("passed")) is not bool:
            raise InvalidExperiment(f"{label}.{task_id}.passed must be boolean")
        latency = row.get("latency_ms")
        if type(latency) is not int or latency < 0 or latency > 86_400_000:
            raise InvalidExperiment(f"{label}.{task_id}.latency_ms must be a nonnegative integer <= 86400000")
        observations[task_id] = {"passed": row["passed"], "latency_ms": latency}
    return observations


def _p95(values: list[int]) -> int:
    values = sorted(values)
    return values[math.ceil(0.95 * len(values)) - 1]


def evaluate_experiment(document: Any) -> dict[str, Any]:
    """Compare identical task sets without granting any production authority."""
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int:
        raise InvalidExperiment("schema_version must be integer 1")
    if document["schema_version"] != SCHEMA_VERSION:
        raise InvalidExperiment("unsupported schema_version")
    experiment_id = _valid_id(document.get("experiment_id"), "experiment_id")
    dataset_id = _valid_id(document.get("dataset_id"), "dataset_id")
    if not isinstance(document.get("baseline"), dict):
        raise InvalidExperiment("baseline must be an object")
    baseline = _outcomes(document["baseline"].get("outcomes"), "baseline")
    candidates = document.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise InvalidExperiment(f"candidates must have 1..{MAX_CANDIDATES} items")

    task_ids = set(baseline)
    base_passes = sum(x["passed"] for x in baseline.values())
    base_p95 = _p95([x["latency_ms"] for x in baseline.values()])
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for row in candidates:
        if not isinstance(row, dict):
            raise InvalidExperiment("each candidate must be an object")
        candidate_id = _valid_id(row.get("id"), "candidate.id")
        if candidate_id in seen:
            raise InvalidExperiment(f"duplicate candidate id {candidate_id}")
        seen.add(candidate_id)
        component = row.get("component")
        if not isinstance(component, str) or component not in ALLOWED_COMPONENTS:
            raise InvalidExperiment(
                f"{candidate_id}: component must be one of {', '.join(sorted(ALLOWED_COMPONENTS))}; "
                "guardrails, scripts, plugins, and subagents are not approved for this pilot"
            )
        observations = _outcomes(row.get("outcomes"), candidate_id)
        if set(observations) != task_ids:
            raise InvalidExperiment(f"{candidate_id}: task_ids must exactly match baseline")

        new_passes = sorted(k for k in task_ids if not baseline[k]["passed"] and observations[k]["passed"])
        regressions = sorted(k for k in task_ids if baseline[k]["passed"] and not observations[k]["passed"])
        candidate_passes = sum(x["passed"] for x in observations.values())
        p95 = _p95([x["latency_ms"] for x in observations.values()])
        reasons: list[str] = []
        if len(task_ids) < MIN_PAIRED_TASKS:
            reasons.append("INSUFFICIENT_PAIRED_TASKS")
        if regressions:
            reasons.append("CORRECTNESS_REGRESSION")
        if candidate_passes <= base_passes:
            reasons.append("NO_NET_PASS_GAIN")
        # Absolute 1ms allowance avoids a zero-latency base accepting any duration.
        if p95 > max(1, base_p95) * MAX_P95_LATENCY_RATIO:
            reasons.append("P95_LATENCY_REGRESSION")
        verdict = "REVIEW_REQUIRED" if not reasons else "REJECT_OR_INCONCLUSIVE"
        results.append({
            "candidate_id": candidate_id,
            "component": component,
            "task_count": len(task_ids),
            "baseline_passes": base_passes,
            "candidate_passes": candidate_passes,
            "pass_rate_gain_pp": round(100 * (candidate_passes - base_passes) / len(task_ids), 3),
            "new_pass_task_ids": new_passes,
            "regressed_task_ids": regressions,
            "baseline_p95_latency_ms": base_p95,
            "candidate_p95_latency_ms": p95,
            "verdict": verdict,
            "reasons": reasons,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "evidence_origin": "EXTERNALLY_SUPPLIED_NOT_VERIFIED",
        "mode": "RESEARCH_ONLY_OFFLINE",
        "automatic_promotion_allowed": False,
        "live_or_paper_execution_authority": False,
        "outcomes_are_trading_performance": False,
        "candidates": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON with paired task observations")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; stdout by default")
    args = parser.parse_args(argv)
    try:
        if args.input.stat().st_size > MAX_INPUT_BYTES:
            raise InvalidExperiment("input too large")
        document = json.loads(args.input.read_text(encoding="utf-8"))
        result = evaluate_experiment(document)
    except (OSError, UnicodeError, json.JSONDecodeError, InvalidExperiment) as exc:
        print(f"harness2_pilot_gate: {exc}", file=sys.stderr)
        return 2
    report = json.dumps(result, sort_keys=True, indent=2) + "\n"
    try:
        if args.output:
            args.output.write_text(report, encoding="utf-8")
        else:
            sys.stdout.write(report)
    except OSError as exc:
        print(f"harness2_pilot_gate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
