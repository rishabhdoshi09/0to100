# Harness2 research pilot: offline evaluation gate

**Status: experiment only.** This is an *evaluation gate inspired by* Google Research's Harness², not a port or execution of Google's framework. It does not generate or execute prompts, install dependencies, invoke models, connect to Kite, or change QuantTerm's Evolution Champion.

Source: https://github.com/google-research/harness2 (Apache 2.0). Google's published runner presently targets Linux/Python 3.12 and documents unrestricted executor network access and disabled Codex sandboxing. **Never run the upstream agent executor on the production Mac, with broker keys, mounted runtime volumes, or unrestricted access to a QuantTerm checkout.** Use a disposable isolated Linux runner with no live credentials and externally enforced network/filesystem restrictions.

## Workflow

1. On a separate machine, prepare at least **30 non-sensitive, fixed, paired agent QA tasks** (e.g., isolated debugging or research-plan tasks). No Kite credentials, customer data, trading orders, broker secrets, or private production logs.
2. Run the baseline workflow and a single bounded prompt, tool-guidance, memory, or skills candidate using an external evaluator (Harness² or another test runner). Hold task inputs and scoring rubrics constant; evaluate objectively. The current script **does not authenticate** the results; preserve original run IDs and artifacts separately for independent audit.
3. Export baseline and candidate observations in the JSON format below; observations must be real measurements from the runner, not imagined improvements. Keep one observation per task per variant; failed/timeouts should be recorded as passed: false with measured timeout latency.
4. Run this from the repository root:

       python scripts/harness2_pilot_gate.py --input observations.json --output report.json

5. A **REVIEW_REQUIRED** result means only that the offline comparison is eligible for **human inspection of the original run evidence**. There is **no automatic promotion or integration**; rigorous independent validation, security review, and production architecture review are required before any future change.

## Input schema (illustrative shape, not example trading evidence)

    {
      "schema_version": 1,
      "experiment_id": "agent-qa-2026-10",
      "dataset_id": "fixed-qa-fixture-v1",
      "baseline": {"outcomes": [
        {"task_id": "task-001", "passed": true, "latency_ms": 1250}
      ]},
      "candidates": [{
        "id": "prompt-candidate-01",
        "component": "system_prompt",
        "outcomes": [
          {"task_id": "task-001", "passed": true, "latency_ms": 1240}
        ]
      }]
    }

The short example intentionally **does not meet** the 30-task review threshold. Run observations may be sensitive; do not commit them or report outputs. The repository .gitignore ignores *.json by default.

## Deterministic vetoes and boundaries

- Both variants must contain exactly the same task IDs; duplicates, invalid types, and missing observations fail closed.
- At least 30 paired tasks, **zero correctness regressions**, positive net pass improvement and candidate p95 latency at most 125% of baseline p95 are required for REVIEW_REQUIRED.
- Changing guardrails, scripts, plugins, subagents or other executable components is out of scope. This gate accepts only system_prompt, tool_guidance, memory and skills experiment labels, **not** their contents; it executes nothing.
- This is a small pilot gate, not a statistical demonstration of generalization. Passing it does not establish the validity of the grading data, eliminate overfitting, prove statistically significant improvements, or demonstrate improved trading performance.
- The script is disconnected from scheduler, risk, portfolio, paper execution, deployment, and API. No imports from production services; no side-effecting execution or network code.

## Local verification

    python -m pytest -q tests/test_harness2_pilot_gate.py

This unit test suite uses generated synthetic **test fixtures** only; it contains no market truth, financial returns, or claimed agent-performance results.
