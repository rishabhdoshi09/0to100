"""
🏛️ The Historical Gauntlet — QuantTerm's one scientifically-valid experiment.

This package is the LAST engineering before the project turns from building
software to generating evidence. It does exactly one thing: take clean historical
data and return ONE verdict per strategy — PASS / FAIL / INCONCLUSIVE — with a
committee-grade report behind it. Nothing here is a feature; every module exists
only because the statistics cannot be trusted without it.

  ledger    (E1) — immutable per-trade record; every field later statistics need
  validator (E4) — abort-on-fail dataset gate that runs BEFORE anything
  runner    (E2) — one command: data → ledger → harness → verdict
  report    (E3) — the committee report (all stats, assumptions, limitations)
  registry  (E5) — experiment id + git/dataset/config hashes + seed (reproducible)
  freeze    (E6) — lock parameters/weights/signals for the run's duration

Run:  python -m gauntlet
"""
