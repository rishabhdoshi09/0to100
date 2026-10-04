# QuantTerm whole-system review — 4 October 2026

Reviewed baseline: `d8ae2f018ccc68b3ebe06f424a407abf56da2849`.

## Assessment

QuantTerm has substantial research, paper execution, risk and recovery machinery.
The most consequential defects in this review were inconsistencies between
persisted evidence, displayed decisions and learned ranking authority. Fixing
those boundaries is more valuable than displaying more unverified trade cards.

This is a review of the critical production paths and their regression and
acceptance contracts, not a claim that every source line, upstream feed or
possible failure has been independently certified. It does not establish a
profitable strategy, Bloomberg equivalence or readiness of the owner's Mac.
The release remains paper-only with live execution locked.

## Findings addressed in this change

| Priority | Finding and consequence | Change | Evidence |
| --- | --- | --- | --- |
| P1 | Committee truth cache keyed only by main SQLite database mtime. A WAL commit could change BUY to WAIT while presentation retained BUY. Presentation also opened the migrating writer connection. | Cache by resolved database path and database/WAL revisions; read with a bounded, read-only SQLite connection. Missing stores are never created. | Real WAL writer commit reproduces stale BUY on the baseline. Held `BEGIN IMMEDIATE` writer and legacy schema verify that presentation reads do not migrate or need a writer lock. |
| P1 | Telegram recommendations used research tier/badge instead of the current frozen committee judgment. The Decision adapter could also convert a decorated WAIT back to BUY because its research tier took precedence. | One current-scan committee snapshot for recommendation delivery; exact lineage checks; committee state takes precedence for decorated cards. Pending/unavailable truth grants no Buy wording. | Regression sends a raw high-conviction Buy card with a persisted WAIT; Telegram and adapter return Wait. An older scan cannot inherit the verdict. |
| P1 | Separate notifier instances could overwrite the shared delivery state. A cash save could erase an already delivered F&O event, enabling duplicates after restart. | Short cross-process persistence lock and union of today's sent keys. Existing F&O send lock retained. | Baseline loses the F&O key when a stale cash notifier saves; updated store retains both keys across restart. This is durable deduplication, not a claim of exactly-once delivery across a crash during network send. |
| P1 | US scanner reported ready/scanned for requested symbols even when all downloads or analyses failed. An outer failure could leave a previous successful artifact active. | Persist per-symbol coverage, actual evaluated count, missing history, analysis and batch failures. Distinguish error, partial and complete no-signal. Failed scans replace stale artifacts. Operation blocks without a new usable scan. Home exposes coverage and reason. | Zero/partial/complete history, batch timeout and analyzer failure regressions. Baseline incorrectly reports ready with no data. |
| P1 | US taken-trade learning used gross returns and nominated entry instead of the journal entry/quantity and modeled net costs. Rejected-name counterfactuals could provide promotion sample/expectancy in the same cells. Duplicate/nonfinite/legacy outcomes were not excluded. | Ranking authority uses unique, finite, settled US PAPER_FORWARD TAKE outcomes with explicit net-cost basis. Counterfactual and uncertified gross history remain stored and counted, but grant no ranking authority. Legacy model generation invalidated. | A baseline positive outcome becomes negative after modeled costs and actual journal basis. Counterfactual-only, wrong-class, gross, NaN, infinity, duplicate and conflicting-identity regressions cannot promote. Positive/negative certified forward evidence still adjusts paper rank within existing caps. |
| P2 | F&O underlying ranking reloaded conditional evidence for each candidate. Concurrent settlement could change the evidence generation within a batch; mounted storage incurred repeated reads. | Load one conditional-evidence snapshot per ranking batch. Existing evidence hierarchy, promotion floors, caps and live lock preserved. | Batch regression checks one load and the same generation across candidates. Existing underlying/contract evidence tests remain in the gate. |
| P2 | Why page presented a research BUY headline without the current committee judgment or execution state beside it. | Explicit current committee verdict/reason/execution state beside a labeled research assessment. Immutable research ranking and decision identity are preserved. | Backend lineage tests and rendered frontend WAIT-versus-research-BUY regression. |
| P1 | Home's production discovery cache was independent of committee verdict updates; a previous eligible candidate could remain after the committee changed to WAIT or its entry became unready. | Recheck the exact-scan committee snapshot at presentation. Only frozen BUY with no explicit entry wait remains in the primary list; pending and unavailable truth cannot inherit cached eligibility. Expose the current execution state. | Cached discovery plus real WAL updates regression; Home exclusions/strict-list tests. |

## System review matrix

| Domain | Reviewed boundary / verification | Assessment and remaining limit |
| --- | --- | --- |
| NSE data and freshness | Kite authoritative covered fields; cache credential rotation; verified active history snapshot and whole-market coverage accounting; prior provider/cache regressions. | Provider rejection must remain visible and fail closed. A fresh full-universe completion on the owner's mounted runtime still needs host evidence. |
| Scan, selection and Home | Actual coverage versus shortlist counts; production selection/risk gate; rejected/wait/reserve exclusions; current recommendation lineage and F&O projection. | No qualifying trade is valid no-action. A data failure or committee-pending state must not be disguised as no eligible trade. Research scores are not calibrated probabilities. |
| Decision intelligence | Research ranking, frozen committee state, conditional evidence, evidence class, confidence/sample floors, Why payload and source lineage. | This release closes stale/contradictory authority boundaries. Additional indicators cannot substitute for forward statistical evidence. |
| Portfolio and execution | Portfolio selection authority, target book, per-name/sector/cluster and aggregate risk; real paper money path; broker mutation boundary and live interlock. | Focused safety suites pass. Live authority remains locked; no order submission is enabled by this audit. |
| OMS, reconciliation, protection | Complete broker snapshots, ambiguous-state handling, protection state and restart recovery. | Existing reconciliation/protection tests pass. Genuine broker/exchange reconciliation and fills are not certified by hermetic tests. |
| Evolution and promotion | Frozen market twin, champion control of real paper selection, challenger isolation, source integrity and promotion governance. | Existing gates remain enforced. A promoted policy is not evidence of future profitability. |
| F&O | Directional and contract gates; expiry/quote/premium/lot completeness; current Home/Telegram candidate versus open/closed paper events; read-only ledger; evidence fusion. | Complete candidates are distinguished from opened trades. Closed sessions, unavailable quotes and inadequate evidence must remain explicit blockers. Real market-path evidence is still required. |
| US | Dedicated scan/operation/paper lane, durable coverage, costed forward learning, rejected-name counterfactual separation and present capability labels. | Forward paper capability only. Trustworthy historical PIT membership including delistings is still unavailable; survivorship-biased historical replay cannot be promoted. |
| Runtime and storage | API bootstrap responsiveness, host ownership, orphan/restart recovery, APFS preflight, toolchain and shutdown/retry scheduling. | CI host migration/restart checks supplement, but cannot replace, observation on the 2015 Mac and external disk. The old native API sample contents were not available for independent diagnosis. |
| Frontend and communication | Frontend unit/build gates; mounted-view browser acceptance; dashboard resilience; recommendation/F&O delivery and reason visibility. | Browser acceptance covers operator workflows. Large frontend bundle and real-device latency remain performance concerns to measure; no latency SLA is certified here. |
| Scientific readiness | Separate institutional-readiness domains; evidence classification and failure injection. | No aggregate readiness percentage or alpha claim. The previously failing strategy experiments are not converted into PASS by operational fixes. |

## Validation and release boundary

Local validation: 184 changed-path Python checks; 178 additional risk, execution,
OMS, promotion, readiness, recovery and F&O-delivery checks; 126 frontend checks;
production TypeScript/Vite build; whitespace check. Regression cases were also
executed against the baseline to confirm causal failures, rather than only
checking the new implementation. Tests use isolated runtime stores and do not
send actual Telegram messages or submit broker orders.

Required release gates on the exact change SHA: CI (12 hermetic shards plus
canonical audit/safety jobs), Terminal UI, Host Migration Gate (Linux and Intel
Mac), Product Acceptance (complete stack, every mounted frontend view with
accelerated operator-hours, restart/recovery and host smoke). Their final
outcomes are recorded on the release PR. Accelerated operator-hours are workflow
coverage, not elapsed market trading or evidence of alpha.

## What must still be demonstrated in operation

1. Install the exact merged build on the canonical Mac through
   `bash ~/0to100/scripts/run_quantterm_complete.sh --update`. Confirm its build
   SHA, a responsive market API and current complete-stack health.
2. Observe a completed current-data NSE scan with genuine full-universe coverage
   and explicit per-name exclusions; distinguish this from a narrow shortlist.
3. During an eligible market session, verify the same contract/decision lineage
   on Home, the F&O page, paper ledger and Telegram. A setup is not an executed
   position; closed-session no-action is not a failure.
4. Accumulate independent costed forward outcomes and evaluate sample size,
   calibration, drawdown, regime stability and liquidity. No release checklist
   can certify profitable trading before that evidence exists.

The development/review work can be released after its gates pass. Host readiness,
economic edge, feed licensing/completeness and institutional latency/reliability
remain separate claims requiring their own evidence.
