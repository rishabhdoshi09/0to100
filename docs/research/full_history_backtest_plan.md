# Full-history walk-forward: decision-ready plan

20 Nov 2026. This is a plan, not a run, and it is **not executed**. A sustained full-history backtest on the production Mac needs the owner's separate, explicit authorization.

## 1. Why consider it

Under the current evidence standard (overlap-robust bound, ≥ 20 blocks, Holm at α/52 per weekly report), the production walk-forward cannot certify any BUY. Its 250-session lookback is about 25 independent 10-session blocks. `docs/research/power_study_2026-11-20.json` puts power under the annual budget at 0.00 for a 0.5 SD edge at 20 blocks. With about 160–320 blocks, under plain overlap:

- a 0.5 SD edge reaches 1.00;
- a 0.3 SD edge reaches 0.91.

The bhavcopy store holds about 1,763 official sessions. After the 60-bar warm-up and the 10-session horizon, about 1,690 sessions (~169 blocks) are usable.

Read in R units: from the production report, a large signal's per-session mean R has an SD of about 0.5R. A 0.3 SD edge is therefore about **+0.15R per trade-day**.

## 2. What the current evidence already says

The production report (16 Nov, schema 2, 800 symbols, 250 sessions, 609.5 s) measures **every signal with ≥ 30 trades as negative**:

| Measure | Range |
|---|---|
| Mean R per trade | −0.02R (GOLDEN_CROSS) to −0.88R (VCP) |
| Day-clustered lower bounds | all below zero |
| Only positive point estimate | HIGH_TIGHT_FLAG, +0.26R on 19 trades (below the floor) |

A longer history buys **precision, not an edge**. It would mostly confirm or refute these negatives more tightly. More sessions cannot make a negative edge positive, and treating a longer run as proof by itself is excluded.

## 3. Resource estimate (production Mac: i5-5250U, 2 cores / 4 threads, 8 GB RAM, USB HDD runtime)

| Item | Current (250 sessions) | Full history (~1,690 sessions) | Basis |
|---|---|---|---|
| Evaluations | 800 × 50 = 40,000 | 800 × 338 ≈ 270,000 (×6.8) | `sample_step = 5` |
| Cost per evaluation | ~15 ms all-in | ~15 ms | flat in history length (synthetic benchmark: 1.6 ms for the scanner alone at 300 and 1,790 bars) |
| Wall-clock, 1 core | 609.5 s (measured) | **~70 min** | linear in evaluations |
| Extra RAM | small | ~50–100 MB (trade and confirmation rows ×6.8) | per-evaluation peak ~0.1 MB |
| USB disk I/O | the store is already in memory in the running process | the same | the bhavcopy store is loaded once; the run reads no further archives |
| CPU contention | one core busy for ~10 min | one core busy for ~70 min | competes with scans and paper cycles |

Risks:

- A run inside market hours would slow the 15-minute scan and paper slots.
- A restart mid-run loses all progress, because `run_backtest` is monolithic today.
- Lid-close sleep or a power loss mid-run (see the host-power notes) aborts it.

## 4. Data validity before the result could be trusted

1. **Point-in-time universe and survivorship.** The bhavcopy archives list every symbol traded on each date, including later-delisted ones, but only for symbols still in the store. Check `nse_universe.point_in_time_universe` survivorship completeness for the older years first.
2. **Corporate actions.** The store is back-adjusted on read from `logs/ca_events.json`. Older splits and bonuses missing from that table would create phantom gaps and corrupt older outcomes. Run `core.data_integrity.verify_ca_adjustment` on an **SSD copy** and require PASS for every year used.
3. **Exact horizons.** `window_is_exact` now drops trades whose 10 bars are not 10 consecutive official sessions. Report the skip count per year; a high skip rate in a year signals data gaps.
4. **Exchange calendar.** The walk-forward indexes sessions by the official bars present, not by the holiday list (only 2026 is verified), so older years are usable. Weekend special sessions are dropped.
5. **Kite.** Not involved: the history is official bhavcopy only.
6. **Reproducibility.** The report records its code SHA, universe hash, first and last session, eval spec, family method and alpha attestation. A rerun on the same store and code must reproduce it.

## 5. Options compared

| Option | Host load | What it buys | Validity notes |
|---|---|---|---|
| A. Full history, one off-hours run (weekend night, AC power, `caffeinate`) | ~70 min of one core, once; weekly if kept | ~169 blocks: real power for edges of +0.15R or more | Needs a resumable driver (option C) so a sleep or restart does not waste the run. Spends one look of the annual budget per run. |
| B. Incremental recomputation | Weekly cost ≈ only the new sessions (~5 × 800 evaluations) plus a merge | The same evidence as A at a small fraction of the weekly cost | Cached per-trade outcomes are keyed by data version. Any corporate-action change to a symbol invalidates its cache. Moderate engineering; the safest long-term path. |
| C. Partitioned research-only run | 8 × ~9 min chunks (100 symbols each), resumable | Same as A; a restart resumes | Stats are additive: clusters, sums and counts merge exactly. Write a partial aggregate per chunk. |
| D. Symbol sampling (e.g. 400 symbols, full history) | ~35 min | Same number of blocks; somewhat noisier per block | Power is driven mainly by blocks and common-factor noise, so the loss is modest. Even-stride sampling must stay. |
| E. Preregistered single hypothesis (e.g. one combination, chosen BEFORE seeing new data) | Same as the run used | No family penalty (m = 1) | Must be declared in the repo before the run. It cannot be chosen from these results (that would be post-hoc). |
| F. Prospective paper evidence only | None | Clean forward evidence | ~40–80 independent blocks (1.6–3 years) for a 0.5–0.3 SD edge. Slow, but cannot be overfit. |

## 6. Recommendation (for the owner's decision)

1. **Do nothing on the host until authorized.** The current gate correctly reports NO_ELIGIBLE_BUY.
2. **If authorized:** build C (a resumable, partitioned driver) first, then run A once on a weekend night after:
   - a PASS of the corporate-action integrity check on an SSD copy;
   - a survivorship check of the older universe.

   Report the per-signal blocks, means and robust bounds. Expect the negatives to be confirmed rather than an edge to appear.
3. **Longer term:** B (incremental) keeps the weekly cost small.
4. **For any future promotion claim:** pair it with E (one preregistered hypothesis) and F (forward confirmation). The walk-forward alone should remain a screening device.

Thresholds are not changed by any of this, and a longer history is not proof. It only makes the existing test able to see an edge if one exists.
