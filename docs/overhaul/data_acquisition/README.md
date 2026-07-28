# NSE Historical Data Acquisition — decision & contracts

This directory is the milestone deliverable: *how* QuantTerm acquires the minimum
reproducible NSE dataset needed to run EXP-006 on real observations. It is about **data
availability, provenance and correctness — not strategy results**.

- `discovery_report.json` / `DISCOVERY_REPORT.md` — existing-data discovery (answer: no
  usable dataset exists / reachable here).
- `SOURCE_DECISION.md` — source comparison, decision (**NSE official archives**),
  licensing, reproducibility.
- `MINIMUM_DATASET_CONTRACT.md` — the 10 minimum requirements + the strong-sector
  decision rule.
- `CORPORATE_ACTION_POLICY.md` — CA/price-integrity handling + limitation-direction
  classification + the verdict rule under incomplete data.
- `UNIVERSE_HISTORY_POLICY.md` — survivorship & sector-membership policy.
- `STORAGE_AND_SNAPSHOT_DESIGN.md` — raw-vs-derived store + snapshot identity + immutable
  run records.

## Canonical network-free test command

```
python -m pytest                 # complete network-free unit suite (excludes tests/integration by classification)
```

Integration tests (environment-dependent, may be slow / need network) are separate:

```
QT_INTEGRATION=1 python -m pytest tests/integration
```

## EXP-006 readiness gate (must ALL pass before the economic run)

Non-empty real price history · benchmark available · acceptable OHLC integrity ·
acceptable adjustment status · deterministic snapshot · sufficient chronological
coverage · no proven future leakage · limitations compatible with a defensible verdict.
When satisfied, run the **unchanged** frozen runner:
`python -m research.momentum_breakout.runner --out logs/experiments/EXP-006`.
If not satisfied, the runner reports the precise blocker and stops (fail closed).

**Current status in this environment: BLOCKED** — no NSE network, no bhav data, no
universe/CA history. The economic hypothesis remains UNEVALUATED.
