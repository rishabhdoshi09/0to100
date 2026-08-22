# FEATURE-001 — SEPA deprecation map

Core SEPA remains `RETIRED_RESEARCH_BENCHMARK`. This milestone does **not** delete research code.

## Keep

- Canonical Trend Template arithmetic (`research/sepa/trend.py`) and `trend_features_v1`
- `rs_cs_v1` (`research/sepa/rs.py`) and `rs_features_v1`
- SEPA-001 / 001R / 001R2.1 / 003 experiment documents, configs, and result files
- Generic structural helpers (PIT universe, FastRS, frames) used outside SEPA
- Ideas 7-rule point scorer as **Trend Quality context** (keys `sepa_*` retained)

## Research-only (do not call from production money paths)

- Core F eligibility (`research/sepa/engine.py`)
- VCP hard-gate and causal VCP state machines used as SEPA gates
- Pivot / buy-zone experiments
- Any future autopilot plan that requires Core F

## Deprecate from production semantics

- Headlines that read as a trade licence: `SEPA Ready`, `SEPA BUY`, `MEETS SEPA`
- Desk copy that says Ready Stage-2 **is** Minervini SEPA approved for money
- Treating `sepa_score >= 40` as SEPA eligibility (it never was Core F)
- VCP page language that implies a validated Minervini system

## Do not delete in this milestone

`research/sepa/**`, `docs/overhaul/experiments/SEPA-*/**`, `scan/setup_engine.py` VCP archetype, `screener/vcp_scanner.py`. Dead-code deletion needs its own tested cleanup.
