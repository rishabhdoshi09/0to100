# QT-08: is the walk-forward's sample stable when data is appended? (21 Nov 2026)

**Research only.** Nothing was deployed. No report was written and no alpha look was reserved.

## The disagreement

GPT's verification pass on the overnight fixes said the walk-forward's sampling grid changes when bars are added. Claude answered that outcome-independent subsampling is not a selection bias. This note settles the question with an experiment.

## The experiment (append-only, production walk-forward)

`research/sampling_stability.py` runs the **production** walk-forward (`scan.signal_backtest.run_backtest` through `research.edge_diagnosis.collect`) on the offline SSD copy:

- **Data:** 1,764 sessions to 2026-11-19, 3,469 symbols.
- **Configuration:** the production lookback (250) and stride (5), 200 symbols.

The copy is cut k = 0, 1, 2, 3 and 4 sessions before its last session. Every earlier bar is identical; the cut only removes the future, which makes the history **append-only**. On the decision window that every cut can measure, the module compares four things:

- which decisions each cut selects;
- whether a decision measured by two cuts has the same outcome (the point-in-time check);
- how far each signal's mean net R moves from the grid phase alone, against its own standard error;
- for "realistic", the symbol sample, when the cut drops names that listed later, as production sees them.

Reproduce:

```
nice -n 19 venv/bin/python -m research.sampling_stability --root <SSD copy root> --symbols 200 --cuts 5 \
    --grid {rolling|session_anchored} [--realistic-symbols] --accept-unbound-coverage --out <json>
```

The results are under `docs/research/qt08/`.

## Results

| Configuration | Symbol-sample Jaccard (k = 1–4 vs k = 0) | k = 0 decisions still selected | Phase-only shift, median (max) SD/SE per signal | Max outcome difference on shared decisions |
|---|---|---|---|---|
| rolling grid, fixed symbols | 1.00 | **16.9%** | **0.38 (0.68)** | 0.0 |
| session-anchored grid, fixed symbols | 1.00 | 100% | 0.00 (0.05) | 0.0 |
| rolling grid + even-stride symbols, realistic listings **(production before this fix)** | **0.26** | **11.1%** | **0.84 (1.97)** | 0.0 |
| session-anchored grid + hash-rank symbols, realistic listings **(production after)** | 1.00 | 99.9% | 0.00 (0.13) | 0.0 |

## What it means for each inference claim

| Claim | Rolling grid + even-stride symbols | Verdict |
|---|---|---|
| **Point-in-time correctness** | Every decision two cuts share has an identical outcome (difference 0.0). Each decision sees only bars up to its session. | **Holds.** |
| **Unbiasedness** | Which decisions are sampled depends on bar counts and the store's symbol list, not on outcomes. | **Holds.** Claude's narrow point stands. |
| **Reproducibility** | One appended session replaced about 83–89% of the measured decisions in an unchanged historical window. A report cannot be reproduced by a later run on the same history. | **Fails.** |
| **Statistical confidence** | The interval of each single report is valid. But the phase adds a hidden random effect of up to about 2 SE between reports on almost the same data, so a sequence of reports does not converge with the data. | **Weakened.** |
| **Hypothesis-family accounting** | The alpha ledger charges every saved look (52 per year, Holm within a look), so family-wise error stays bounded. Each look, however, is a near-independent redraw of the same history, not an update with new data. | **Bounded, but each look buys little information.** |
| **Edge-certification stability** | A row near its threshold can be certified or withdrawn by the calendar, through a holiday week, a few new listings or one extra bar, rather than by evidence. | **Fails.** This was GPT's point, and it is the decisive one. |

**Resolution.** GPT was right that the sampling grid makes evidence unstable. Claude was right that it is not a point-in-time or selection bias. The behaviour is not acceptable for an admission gate whose BUY decision must depend on the data, not on the sampling phase.

## The correction (implemented)

1. **Decision grid** (`scan.signal_backtest.DECISION_GRID = "session_anchored"`). A decision day is sampled when its official-session ordinal is a multiple of the stride. Appending data can only **add** decisions.
2. **Symbol sample** (`_spread_sample`). Each symbol has a fixed SHA-256 rank and the `limit` lowest are taken. A new listing displaces at most one name. The sample is still spread across the alphabet and independent of names and outcomes.
3. **Versioned.** `SAMPLING = "hash_rank_session_anchored_v1"` is part of `gate_program()`. Every earlier attestation therefore belongs to another program and **fails closed**, and the supervisor reruns the walk-forward because `sampling` changed (`Supervisor._legacy_calibration`).
   - The rerun spends one look from the year's alpha budget.
   - No BUY was certified anyway, so nothing is lost.

**Regressions.**

- `tests/test_cycle2_20261121.py::test_qt08_*`:
  - an anchored grid stays nested with identical outcomes under append, while the rolling grid moves;
  - the production program is the new one;
  - a new listing displaces at most one sampled symbol.
- `tests/test_weekly_backtest.py`: the sample still spans the alphabet.

**Assumptions stated.**

- The stride is fixed at 5. A different stride is a different program.
- The official-session index is the realised archive plus the verified calendar (`core.session_window`). A session missing from the archive changes later ordinals, so a repaired archive is a new data identity, as it already is.
- Hash-rank sampling is uniform only in expectation. It is a fixed sample, and an unlucky draw stays unlucky. Its benefit is reproducibility, not a better draw.
