# EXP-006 Historical Evidence Run — committed artifact record

This directory is the **permanent, version-controlled** artifact set produced by
executing the **frozen** EXP-006 runner (`research/momentum_breakout/runner.py`) on
the best available point-in-time NSE dataset in this environment. (The runner's
default output path is `logs/experiments/EXP-006/`, which is git-ignored — so the run
is persisted here, under `docs/`, to make it auditable from the repository.)

## Verdict

**INCONCLUSIVE — DATA_UNAVAILABLE.**

No point-in-time NSE dataset exists in this environment and none can be built:

- `logs/bhav/` contains **0** files; `bhavcopy_store.is_ready()` is **False**.
- `bhavcopy_store.build_store()` requires the NSE network, which is unavailable here
  (`https://www1.nseindia.com` → **HTTP 000**; the build blocks and cannot complete —
  verified by a bounded 45s attempt that timed out).
- No `logs/universe_history.json` (survivorship), no `logs/ca_events.json` (corporate
  actions), no point-in-time fundamentals, no dated sector-membership history.

The data-quality gate therefore **failed closed** and the runner emitted
INCONCLUSIVE rather than fabricating a PASS/FAIL. **This is not strategy evidence.**
It is the honest result: the registered hypothesis cannot be judged without the data.

## Reproducibility identities

| Field | Value |
|---|---|
| Experiment | EXP-006 · Institutional Momentum Breakout v1 |
| Primary exit | `structural_trend` (structural stop + close-below-50-DMA) |
| Snapshot ID | `ad652107580ddae1` |
| EXP-006 config hash | `4f638f99e13bf939` (unchanged from the frozen framework) |
| Code commit | `a634be3` (frozen runner) |
| Cost model | modelled aggregate 0.22% round-trip + 0.10% slippage (NOT a broker contract-note replication) |
| Universe policy | survivorship_complete = **false** |
| Adjustment policy | corporate actions unavailable (RAW) |
| Benchmark / sector / fundamental identity | unavailable / current-membership-not-dated / none (no PIT fundamentals) |

## Files (this fail-closed set)

- `data_quality.json` — machine-readable gate result (fatal: DATA_UNAVAILABLE).
- `snapshot_manifest.json` — dataset snapshot identity (reproducibility stamp).
- `experiment_spec.json` — the frozen EXP-006 pre-registration (hypothesis, primary +
  secondary exits, entry convention, comparisons, decision metrics).
- `config_snapshot.json` — the frozen thresholds/config that produced the config hash.
- `limitations.json` — recorded limitations.
- `verdict.json` — the final evidence verdict.
- `artifact_index.json` — index of the above.

When real data is present, the same run additionally emits: `observations.jsonl`,
`rejected_candidates.jsonl`, `trade_ledger.jsonl`, `no_fills.jsonl`,
`primary_metrics.json`, `exit_variants.json`, `ablations.json`,
`benchmark_comparisons.json`, `regime.json`, `sector_breakdown.json`,
`valuation_breakdown.json`, and `multiple_testing.json`.

## How to reproduce on a data host (e.g. a Mac with a built bhav store)

```bash
# 1. Ensure the NSE bhavcopy + index stores are built (needs network), and ideally
#    supply logs/universe_history.json and logs/ca_events.json for research-grade data.
# 2. Run the frozen EXP-006 runner:
python -m research.momentum_breakout.runner --out logs/experiments/EXP-006
```

The runner freezes a snapshot, generates candidates chronologically (one event per
breakout), simulates the pre-registered primary + secondary exits (gap-aware, next-bar
entry), runs the six ablations + benchmark comparisons, applies the existing harness +
BH-FDR multiple-testing control, writes the full artifact set, and prints the verdict.

## What a real-data verdict can and cannot be

The runner's **research-grade gate** is enforced: a would-be PASS on
survivorship-incomplete or CA-unadjusted data is **downgraded to INCONCLUSIVE** (a
biased PASS is not defensible — the EXP-005 lesson). A **FAIL is retained** (meaningful
even on optimistically-biased data). So on the current data policy, a PASS is not
attainable until at least survivorship + corporate actions reach research grade.

## Guarantees (unchanged; re-verified by the test suite)

The evidence run imports no execution, broker, GTT or order path. PAPER autopilot
remains operational, Telegram remains paper-only, and LIVE remains migration-locked.
Verified by `tests/test_momentum_breakout_run.py` (execution isolation, DATA_UNAVAILABLE
fail-closed, artifact reproducibility, no same-bar entry, gap handling, exit-variant
separation, ablation isolation).
