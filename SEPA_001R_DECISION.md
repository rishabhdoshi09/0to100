# SEPA-001R final decision

**Is core SEPA now evidence-supported for this NSE system?**

## KEEP RESEARCH-ONLY

Not PROMOTE TO PAPER — CA verify failed, PIT class is `PIT_DEGRADED`, core F has 4 trades with a **fully negative** block-bootstrap CI.  
Not REJECT — Stage-2 and RS still move the scanner baseline in the expected direction; timing/lifecycle now produces real (rare) SEPA fills instead of zero.  
Not MODIFY AND RETEST as the *primary* label: the 001 deficiencies (late pivot, step-10, missing CA table, n=80 snapshots) were addressed. The remaining gap is **evidence**, not a missing code object.

Promotion to paper is **not** recommended. No autopilot, broker, GTT, or production BUY wiring was added.

### Promotion gate (this run)

| # | Condition | Result |
|---|---|---|
| 1 | CA integrity acceptable | **No** — verify FAIL, `ca_complete=false` |
| 2 | No known look-ahead in VCP/pivot | **Yes** — tests in `tests/test_sepa_001r.py` |
| 3 | Universe limitations bounded | **Yes** — `PIT_DEGRADED`, documented |
| 4 | Core E or F meaningful sample | **No** — E=2, F=4 |
| 5 | Expectancy ≥ 0 after costs | **No** — F −0.62R, E −1.09R |
| 6 | CI not materially negative | **No** — F CI [−1.09, −0.15] |
| 7 | Not one year / one sector | **No** — F is 2026-only, n=4 |
| 8 | Buy-zone reasonably stable | **Not estimable** — do not widen |
| 9 | RS not selected from a tiny sample | **Yes** — threshold stays 70; buckets monotone |
| 10 | Live-equivalent eligibility = backtest | Research object only; **not** wired live |

## Ten follow-up answers

1. **Did corrected VCP timing solve the near-zero trade problem?** Partially. Fills went from ~0 to **4** valid F fills, with gap-through/missed/extended classified instead of chased. It did **not** produce a powered trade stream. Median first-snapshot distance is still ~+10% because many bases are first seen already through last-contraction resistance; the lifecycle then waits or refuses.

2. **Did daily evaluation materially improve specific-entry capture?** **Yes.** `sample_step=1` is why CHENNPETRO / LAURUSLABS / MOTHERSON / SBIN `ENTRY_READY` dates exist. SEPA-001 step-10 could not see a 1.5% band. Dedup kept 944 unique setups rather than one row per day.

3. **Did corporate-action corrections materially change results?** A real NSE share-count ledger was ingested (**290** events, hash `e260673881d9e5c3`). Verification **did not pass**. 38 gap symbols were excluded. Results are **degraded**, not fully PIT-safe. Do not claim CA is “done.”

4. **Is Stage-2 still additive?** **Yes, as a scanner filter.** A −0.25R (n=1463, REJECT) → B +0.11R (n=483). Block CI still includes a small negative. Not a standalone system.

5. **Is RS still additive?** **Mildly on the scanner path** (B +0.11 → C +0.13) and **clearly on ungated 20d forward returns** (50–69 +1.7% → 95–99 +9.8%). Keep RS≥70. Do not jump to 90.

6. **Does structural VCP still add value?** As a scanner add-on (D n=62, +0.13R) the point estimate is similar to C with fewer trades — **UNDERPOWERED**, and D still uses scanner fills. As **core SEPA** (E/F) it is a refusal engine, not yet an edge.

7. **Does specific-entry discipline improve expectancy or primarily reduce frequency?** **Primarily reduce frequency.** D 62 → E 2 / F 4. Expectancy on the filled remnant is **worse**, not better. Discipline is doing its job; do not convert misses into `entry = last price`.

8. **Is the full core-SEPA stack superior to the baseline?** **INCONCLUSIVE / no.** F cannot be compared to A with n=4. Pieces of SEPA (Stage-2, RS) still look like filters. The specific entry does not yet show a positive executable edge.

9. **What is the statistical confidence?** F: n=4, mean −0.62R, CI [−1.09, −0.15], harness UNDERPOWERED. B/C harness PROMOTE is deflated Sharpe on scanner fills with CI crossing zero and PIT_DEGRADED — **not** paper-grade. `n≥30` is not sufficient by itself (A has 1463 and is a loser).

10. **What exact evidence is still required before paper trading?**
    - `verify_ca_adjustment` PASS and `ca_complete=true`
    - Official listing/delisting archive **or** an explicit decision to accept `PIT_DEGRADED`
    - Core E or F with a **powered** sample (not 4 trades) and CI not materially negative
    - At least one unseen calendar block with fills (F had none in 2025)
    - Sector/regime coverage that is honest about PIT
    - Live eligibility function **identical** to `evaluate_sepa_eligibility` / `sepa-001r.v1`  
    Then re-ask promotion. Do not skip to autopilot.
