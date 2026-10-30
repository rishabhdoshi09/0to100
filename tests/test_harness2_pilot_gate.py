"""Tests for the isolated offline Harness2 workflow comparison gate."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "harness2_pilot_gate.py"
spec = importlib.util.spec_from_file_location("harness2_pilot_gate", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def trial(n=30, base_passes=20, candidate_passes=23, base_ms=100, candidate_ms=100):
    return {
        "schema_version": 1,
        "experiment_id": "pilot-001",
        "dataset_id": "nonproduction-qa-v1",
        "baseline": {"outcomes": [
            {"task_id": f"task-{i:02d}", "passed": i < base_passes, "latency_ms": base_ms}
            for i in range(n)
        ]},
        "candidates": [{
            "id": "candidate-prompt-1",
            "component": "system_prompt",
            "outcomes": [
                {"task_id": f"task-{i:02d}", "passed": i < candidate_passes, "latency_ms": candidate_ms}
                for i in range(n)
            ],
        }],
    }


def test_improved_is_review_only_never_promotion():
    r = module.evaluate_experiment(trial())
    assert r["automatic_promotion_allowed"] is False
    assert r["live_or_paper_execution_authority"] is False
    assert r["evidence_origin"] == "EXTERNALLY_SUPPLIED_NOT_VERIFIED"
    c = r["candidates"][0]
    assert c["new_pass_task_ids"] == ["task-20", "task-21", "task-22"]
    assert c["regressed_task_ids"] == []
    assert c["verdict"] == "REVIEW_REQUIRED"


def test_regression_veto_even_if_net_gain():
    doc = trial()
    doc["candidates"][0]["outcomes"][0]["passed"] = False
    r = module.evaluate_experiment(doc)["candidates"][0]
    assert r["pass_rate_gain_pp"] > 0
    assert r["regressed_task_ids"] == ["task-00"]
    assert "CORRECTNESS_REGRESSION" in r["reasons"]


def test_insufficient_tasks_cannot_pass():
    r = module.evaluate_experiment(trial(n=12, base_passes=5, candidate_passes=7))
    assert "INSUFFICIENT_PAIRED_TASKS" in r["candidates"][0]["reasons"]


def test_p95_latency_gate():
    r = module.evaluate_experiment(trial(candidate_ms=150))
    assert "P95_LATENCY_REGRESSION" in r["candidates"][0]["reasons"]


def test_same_task_set_required():
    doc = trial()
    doc["candidates"][0]["outcomes"][0]["task_id"] = "different-task"
    with pytest.raises(module.InvalidExperiment, match="exactly match"):
        module.evaluate_experiment(doc)


def test_duplicate_tasks_and_candidate_ids_fail():
    doc = trial()
    doc["baseline"]["outcomes"][1]["task_id"] = "task-00"
    with pytest.raises(module.InvalidExperiment, match="duplicates"):
        module.evaluate_experiment(doc)
    doc = trial()
    doc["candidates"].append(dict(doc["candidates"][0]))
    with pytest.raises(module.InvalidExperiment, match="duplicate candidate"):
        module.evaluate_experiment(doc)


@pytest.mark.parametrize("component", ["guardrails", "scripts", "plugins", "subagents", None, [], {"tool": "shell"}])
def test_disallowed_components_rejected(component):
    doc = trial()
    doc["candidates"][0]["component"] = component
    with pytest.raises(module.InvalidExperiment, match="component"):
        module.evaluate_experiment(doc)


@pytest.mark.parametrize("invalid", [None, 12, "true"])
def test_nonboolean_results_fail(invalid):
    doc = trial()
    doc["baseline"]["outcomes"][0]["passed"] = invalid
    with pytest.raises(module.InvalidExperiment, match="boolean"):
        module.evaluate_experiment(doc)


def test_no_net_gain_rejected():
    r = module.evaluate_experiment(trial(candidate_passes=20))
    assert "NO_NET_PASS_GAIN" in r["candidates"][0]["reasons"]


def test_json_cli_offline(tmp_path):
    data = tmp_path / "inputs.json"
    data.write_text(json.dumps(trial()), encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(data)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["candidates"][0]["verdict"] == "REVIEW_REQUIRED"


def test_json_cli_rejects_invalid(tmp_path):
    data = tmp_path / "invalid.json"
    data.write_text("not json", encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(data)], capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout


def test_boolean_latency_rejected():
    doc = trial()
    doc["baseline"]["outcomes"][0]["latency_ms"] = True
    with pytest.raises(module.InvalidExperiment, match="latency_ms"):
        module.evaluate_experiment(doc)


def test_schema_bool_rejected():
    doc = trial()
    doc["schema_version"] = True
    with pytest.raises(module.InvalidExperiment, match="schema_version"):
        module.evaluate_experiment(doc)
