# QuantTerm release readiness (26 Nov 2026)

**Verdict: NOT READY. Production certification is BLOCKED.** The software candidate is coherent and
verified in development. The operational gates cannot be exercised while the host's runtime storage is
unavailable, and several gates need the owner's words. Nothing here is production-verified.

- **Candidate:** the head of `claude/cycle9-20261126` (cycle 9 + cycle 10 so far). Last fully certified SHA:
  `a55451fc` (CI 5/5 on both `claude/cycle9-20261126` and `claude/settlement-batch-20261118`; the Mac full
  guarded suite ran at `b9e2842b`, 537 files; the one failing test there is updated by the test-only `a55451fc`).
- **Production:** `125880a6` on `host/mac-release-20261110`. It is untouched.
- **Live broker:** HARD-LOCKED. No code path in this candidate unlocks it.

## 1. Gates

PASS: measured, with evidence. FAIL: measured and not met. BLOCKED: cannot be exercised now (the reason
is given). OPEN: work remains.

| # | Gate | Criterion (from existing requirements) | Status | Evidence / reason |
|---|---|---|---|---|
| G1 | No open release-blocking defect | Every P0/P1 in §2 closed or explicitly accepted | **OPEN** | §2: A1 (P1, accepted residual pending a design), B08 (P1, live path only, dormant) |
| G2 | Exact-SHA CI | All 5 workflows green on the release SHA | **PASS** at `a55451fc`; re-run per new head | GitHub runs 45798607943 (CI) et al.; candidate branch 45800115001 |
| G3 | Full guarded Mac suite | Every test file passes on the frozen release SHA (one interpreter per file, launchctl/hdiutil/kill guarded) | **PASS** at `b9e2842b` + the test-only delta | 537 files; the delta file and 9 repair-tool files green at `a55451fc` |
| G4 | Crash, restart and idempotency | Crash-injection tests at every persistence boundary of the money and learning paths | **PASS (development)** | Cycles 2–10. Cycle 9: GOOD_WAIT (15 tests), manifest lock (17), freeze index (11), stop boundaries (5), review (15) |
| G5 | Mutation coverage of the changed risk areas | Every targeted mutant killed, or documented as equivalent | **PASS** | Cycle 9: 57/57 (one equivalent documented). Cycle 10: each fix's base failure recorded |
| G6 | Independent review reconciled | Each substantial cycle cross-reviewed; every claim tested; disagreements recorded | **PASS** for cycle 9 | `docs/audits/2026-11-26-cycle9-checkpoint.md`: 13/13 reproduced and fixed; A1/A5 reasoned |
| G7 | Point-in-time research | No input after its information time; one input generation per cross-section | **PASS (development)** for feature002 and the walk-forward gate | Cycle 9 item 4 plus review. Historical runs are not re-derived |
| G8 | Evidence honesty | No BUY without dated, Holm-surviving, provenance-bound evidence; NO TRADE explained | **PASS (design)** | The evidence gate since 7e2df305; Home shows `NO_ELIGIBLE_BUY`. No edge is claimed: setups showed no selection alpha (−0.17% vs EW) |
| G9 | Host preflight | Runbook §0 passes on the actual host | **BLOCKED** | The runtime volume has been unmounted since 24 Nov 20:01 IST (§3) |
| G10 | Backups | A verified image plus manifest; restore tested on a disposable copy | **PARTIAL** | Stage 1 image verified 22–23 Nov (0 mismatches). Stage 3 re-verification and final pass need the owner's words |
| G11 | Rollback | Deploy rollback documented and rehearsed | **PARTIAL** | `git reset --hard <prev>` + `host_install install` (used five times in November). F1 and store rollbacks tested on copies only |
| G12 | Performance | API, scan, settlement and startup latency within the operating requirement on the real host | **BLOCKED** | The desk is down. Development benchmark: freeze p50 845→85 ms (SSD copy). P2 open: `/api/product-readiness` 12.7 s, `/api/autonomy-desk` 8.4 s (23 Nov, host) |
| G13 | Paper and accounting state | The paper book, F&O book and ledgers reconcile at the installed SHA | **BLOCKED** | Their files are on the unmounted runtime |
| G14 | Operational soak | A documented paper soak at the release SHA (the previous soak ran 72 h) | **BLOCKED** | Needs the deploy and a mounted runtime |
| G15 | Monitoring and alerts | Storage, power and battery alerts reach Telegram | **PARTIAL** | They fire in code. The 24–26 Nov alerts failed on DNS (NameResolutionError) while the host slept or woke |
| G16 | Security and interlocks | Live lock holds under adverse conditions; secrets never in logs or reports; dependency advisories clean | **PASS (development)** | Dependency Vulnerability Advisory green; the live lock was not changed this cycle; reports carry no secrets |

## 2. Engineering backlog (prioritized)

| Pri | Item | State |
|---|---|---|
| P1 | **A1** A crash between the paper fill's book save and its GOOD_WAIT mark loses that credit | **Accepted residual.** A safe reconciliation needs a durable entry-origin marker on each book position. Several engines open `PaperBook` positions, so the book alone cannot tell which entries the live path credits. It is an under-count, consistent on both settlers |
| P1 | **B08** The legacy Zerodha adapter has no idempotency after an accepted timeout | Open, **live path only**. Dormant: the real wrapper blocks every submission. A live-readiness blocker, not a paper one |
| P2 | The 23 Nov Nifty disagreement (+1.12% market panel vs −1.41% F&O context) | **BLOCKED:** its evidence is on the unmounted runtime. The F&O formula (`last / ohlc.close − 1`) is correct in-session |
| P2 | `/api/product-readiness` 12.7 s, `/api/autonomy-desk` 8.4 s | **BLOCKED:** needs a runtime copy to profile |
| P2 | A5: an unlocked external ledger edit in a microsecond window | Accepted. It fails closed on the next interrupted append |
| Done (cycle 9) | GOOD_WAIT crash recovery; F1 manifest writers; F8 freeze cost; feature002 PIT; stop boundaries; 13 review findings | `a55451fc` |
| Done (cycle 10) | Shutdown join deadline (C8.7-03); durable taken-evidence appends | after `a55451fc` |

## 3. Host, storage and power assessment

**Observed (read-only, 26 Nov):**
- The desk has been paused since 24 Nov 20:01 IST (`Errno 5`).
- The `Expansion` USB HDD (exFAT) is mounted.
- The WORKING APFS sparsebundle is **attached but not mounted**: disk3, disk4s1. Its diskimages-helper dates from 24 Nov 09:44, before the outage.
- Clamshell sleep on battery 25 Nov 14:26 → 26 Nov 13:26; the battery reached 4%. It is now on AC at 100%.

**Recorded history** (`docs/audits/2026-11-23-cycle8-checkpoint.md`, memory):
- in November, at least eight outages from lid close or battery drain;
- the USB drive leaving the bus on an AC→battery switch;
- a kernel panic in `com.apple.driver.DiskImages` on 23 Nov;
- battery health "Poor", capacity 65%;
- two F1 data losses seconds before a sleep.

**Assessment.** These are hardware and host-configuration risks. Software can only fail closed, which it does: no fallback directory, identity-checked storage, barriers. Retries cannot make them safe. In order of effect:
1. **Power (physical).** Keep the Mac on AC with the lid open; this removes most events. Consider replacing the degraded battery: it cannot carry the bus-powered HDD's transient, which is plausible, not proven.
2. **Sleep policy (owner, needs a password).** `sudo pmset -a disablesleep 1` is the only setting that blocks clamshell sleep; `caffeinate -i` does not.
3. **Storage hardware (purchase).** Replace the bus-powered USB HDD with an externally powered SSD, APFS-formatted. This removes the exFAT metadata window and the about-90-IOPS limit behind the November API outage. Alternatively, move the 16 GB runtime to the internal SSD; that needs disk space, as the SSD had 11 GB free on 12 Nov.
4. **Stale-attach recovery (owner decision).** The preflight's opt-in stale-detach (25ee30cf) is OFF by design. Enabling it changes a fail-closed gate.

## 4. Approvals needed (each a bounded decision)

| Words | What happens | Why |
|---|---|---|
| "detach and remount" | `hdiutil detach disk3`, then the supervisor re-attaches `QuantTermStorage_WORKING_20261107.sparsebundle` itself; no manual attach | Unblocks G9, G12, G13, G14 |
| "retarget relay" | `config.json` session_id → this session, with a backup; a TEST delivery | Review instructions reach a live session |
| "re-run stage 2" | Read-only re-plan of the store and F1 repairs with the release SHA (manifest v3) | Stage 3 prerequisite |
| Stages 3–6 words | Runbook `docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md` | Backup verify, stop, repairs, deploy |
| "deploy `<sha>`" | Runbook stage 6 (`host_install install`) | Activates the candidate |

## 5. Release procedure (after the approvals; outside 09:15–15:30 IST, between paper slots)

1. "detach and remount". Wait for "Runtime recovered" and health READY; the owner's Kite login.
2. Runbook §0 preflight at the release SHA. Any failure stops here.
3. "re-run stage 2" (read-only). Both plans must be READY with every gate true.
4. Stage 3: verify the Stage 1 image against its manifest, stop the desk, final pass, SSD tier.
5. Optional 3b/4/5 on their own words. Then stage 6: deploy the release SHA. Verify the installed SHA, services, live lock, paper books, job schedule, freshness and alerts.
6. Soak (G14), with the report scheduled. Any failed operational criterion means no release success.

Rollback: `git reset --hard 125880a6` on `host/mac-release-20261110`, then `host_install install`; and the runbook's per-stage rollbacks.
