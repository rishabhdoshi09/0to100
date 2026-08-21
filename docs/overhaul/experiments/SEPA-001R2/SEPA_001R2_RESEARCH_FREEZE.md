# SEPA-001R2 research freeze

**Status:** Frozen **before** the canonical long-history A–G run.  
**Revision:** SEPA-001R2.1  
**No further threshold modification** is permitted for this experiment.

SEPA-001 / SEPA-001R / prior R2 audit notes remain immutable. This freeze
locks the **research code and data identity** used to produce
`SEPA_001R2_RESULTS.md` and `SEPA_001R2_DECISION.md`.

Paper integration is **not** part of this freeze. Live orders, GTT, broker
execution, and autopilot remain untouched.

---

## 1. Tests

Complete SEPA suites were green immediately before this file was written:

```
tests/test_sepa_001r21.py
tests/test_sepa_001r2.py
tests/test_sepa_001r.py
tests/test_sepa_001_eligibility.py
tests/test_sepa_001_ablation.py
tests/test_sepa_001_setup.py
```

`pytest` exit code 0 (94 passed).

R2.1 required tests included: causal CA vs 2024 universe, pre-CA signal
usable, horizon CA-censor, post-event lookback isolation, post-event
re-entry, future CA ↛ historical RS, FastInvestable ≡ screen membership,
session embargo (weekend / holiday / 1-session / 20-session / overlap),
G without placeholder R, daily E vs 5-day miss, scanner wrapper ≡
`UnifiedScanner._analyze`.

---

## 2. Frozen strategy (unchanged)

| Knob | Frozen value |
|---|---|
| Eligibility | `sepa-001r2.v1` |
| VCP | `vcp_causal_v2` |
| Pivot | `pivot_last_contraction_v1` |
| RS threshold | 70 |
| RS formula | `rs_cs_v1` 0.40×r63 + 0.20×r126 + 0.20×r189 + 0.20×r252 |
| Buy-zone | [−0.25%, +1.50%] |
| Structural stop cap | 8% |
| VCP lookback | 120 sessions |
| Contractions | min 2 / max 6 |
| Fill | next-open vs buy-zone; no `entry = last price` |
| Config hash | `76acdb2bb188a5f4` |

Do **not** retune RS, VCP, Stage-2, stop, or the buy-zone from validation
or confirmation numbers.

---

## 3. Code identity

| Field | Value |
|---|---|
| Branch | `cursor/sepa-001r2-validity-942f` |
| Code SHA (parent of this freeze commit) | `f516e5836e708f21ccad440f0d784e0efd481928` |
| Freeze commit | *this file’s git SHA after commit* |
| Universe methodology | `bhav_inferred_asof_v2` (candidates = official bars ≤ as_of; causal CA segments; datetime64 ns as-of) |
| Scanner research function | `UnifiedScanner._analyze` via `research.sepa.scanner_research.research_scanner_analyze` |
| Scanner equivalence | `tests/test_sepa_001r21.py::test_scanner_research_wrapper_matches_unified_analyze` PASS |
| Observation | `date_step=1`, `scanner_step=1` |
| Embargo | actual exit session, not `as_of + Timedelta(days=hold)` |
| G | pure signal study (`research.sepa.signal_study`) — not SEPA R |

The long-history payload records `git rev-parse HEAD` at run start.

---

## 4. Data identity

| Field | Value |
|---|---|
| Source | official NSE `sec_bhavdata_full` (no fabricated bars) |
| First session | 2019-08-23 |
| Last session | 2026-08-21 |
| Store sessions | 1787 |
| Research frames (`≥80` bars) | 3054 |
| CSV+cache manifest SHA-256 | `137e00a020aed647dd4963eba9c590f302dea693f41bc9d1b9e7b42d69f2d133` |
| Symbols by year (frames with ≥1 bar) | 2019:1585 2020:1719 2021:1888 2022:2034 2023:2168 2024:2300 2025:2546 2026:2623 |

2018 files are not served on this endpoint (404) and were not invented.

---

## 5. Corporate-action identity

| Field | Value |
|---|---|
| Policy | `ca_sharecount_v1` — never infer a factor |
| Ledger path | `logs/ca_events.json` |
| Ledger SHA-256 | `900f11a9fc5a1852a3aef294e1567c68aa2481510cc3a8bea05e011f104c1725` |
| Short hash | `900f11a9fc5a1852` |
| Events / symbols | 607 / 462 |
| Coverage | 2019-01-15 → 2026-08-25 |
| Global `ca_complete` | **false** (verifier threshold unchanged) |
| Research treatment | `CATimeline` causal segments + `CA_CENSORED_OUTCOME` |

---

## 6. Walk-forward (predeclared)

See `SEPA_001R2_VALIDATION_PROTOCOL.md` (committed before this freeze and
before results):

- Warm-up: 2019-08-23 through 252 sessions
- Development: first eligible date → 2023-12-31
- Validation: 2024-01-01 → 2024-12-31
- Confirmation: 2025-01-01 → 2026-08-21

Deployment `has_unseen_block` uses the confirmation block. It is not
hardcoded `False`.

---

## 7. After this freeze

1. Run canonical daily A–G on the official book already on disk.
2. Persist `ablation_001r2.json`, setup/opportunity ledgers, funnel,
   results, walk-forward, and a single decision.
3. Stop. Do not implement paper integration in this milestone.
