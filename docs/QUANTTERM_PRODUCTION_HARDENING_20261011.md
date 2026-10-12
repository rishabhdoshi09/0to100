# QuantTerm production-hardening acceptance — 11 October 2026

## Evidence-based incident baseline

This is a release gate, not a claim of launch readiness. Logs from the canonical 8 GB Intel Mac running on external APFS storage showed:

- `MARKET_SCAN` completed SUCCESSFULLY after **928.4 s** including **638.1 s** after technical screening; approximately 2,179 / 2,384 approved NSE symbols reached the scanner ledger and 419 technical setups qualified.
- Dashboard GET degraded from ~30–60 s to **347.6 s**, with data readiness alone consuming **217.1 s**.
- Decision Simulation status GET reached **230 s+**; scan freshness stage reached **274.8 s**.
- The original UI polled operation details **every second**, and created new `OperationStore` instances that ran schema DDL and opened SQLite handles on every status request.
- Repeated Scr​​eener zero-row snapshots had been treated as successful cache refreshes.
- Full result blobs were printed into the market operation logs.
- No evidence of nine actual paper BUY executions was provided; the nine `SCANNER_SETUP` BUY labels are NOT proof of approved orders.

## Release candidate architecture

**Separation of authority is mandatory.**

- **Scan worker:** owns Kite/current-market inputs, official NSE historical sessions, detector evaluations, coverage ledger, and atomic scan publishing. It may do slow computation but must not block the desk's read threads.
- **Read-only HTTP snapshots:** bounded, single-flight, timestamped status projections. Expired data must be labelled and fail-closed for paper/autonomy/F&O/approval eligibility. HTTP cached statuses MUST NOT be consulted by trade execution or approval mutations.
- **Market operations:** UI polls only the metadata status endpoint; audit details remain accessible from the full result endpoint on explicit request. SQLite schema set-up and read connections are reused safely per DB inode/thread.
- **Research overlay:** stage timings explicitly identify where time went in SEPA ranking, fundamentals, recommendations, decision discovery and reports; do not discard data/edge evidence simply to improve speed.
- **Data integrity:** reject zero-row fundamentals as proof of financial coverage; preserve last-good numeric evidence and official warehouse data; no synthetic data.

## Go / hold gates

1. **CI:** all canonical Python shards, TypeScript UI build/tests, product acceptance, host migration and safety tests PASS on the **exact consolidated commit SHA**. Any failed or cancelled gate = HOLD.
2. **Live-money safety:** live money remains positively verified LOCKED, broker order authorization FALSE, public mutation requires operator token. Do not use a green UI badge as evidence of this lock.
3. **HTTP response time:** cold dashboard/decision-gate GET responds with a truthful warming state in <=3 s. After a snapshot completes, at least 20 sampled reads per endpoint meet p95 <=3 s and no unhandled HTTP 500. Expired snapshots show stale warnings rather than fabricated eligibility.
4. **Host resource soak:** on the real Mac, after 30 minutes of ordinary polling plus one scan, no runaway memory/swap growth, worker exit, FD exhaustion, repeated SQLite busy storms, or hung UI. Record swap delta, memory pressure, CPU, disk I/O, and p95 request latency.
5. **Market data:** correct NSE market-session provenance (e.g. 2026-10-09 on Sunday 2026-10-11), no fake current-session prices, positive official/broker source labels. Target >=95% approved universe successfully evaluated, zero silent symbol skips, explicit reasons for remaining gaps. No pass at the observed ~91.4% coverage.
6. **Scan timings:** instrumented wall-clock budget for history adoption, technical scan, SEPA, long-term, recommendation build, decision board, reports and publish. Stop guessing which phase consumes 638 seconds. Tune the measured hot phase and repeat a full real scan before a runtime claim. The 15-min incremental scanner and the whole-universe daily scan have different latency SLOs and must be measured separately.
7. **Paper lifecycle:** reproduce a permitted `PAPER_FORWARD` entry, entry ticket, book mutation, exchange-side-independent virtual exit supervision, settlement, journal, and Telegram delivery on a safe paper scenario. Valid NO_ACTION counts as a correct decision. Neither scanner BUY labels nor Telegram signal alerts qualify as paper executions.
8. **F&O:** prove Kite instrument mapping + session freshness + contract liquidity and risk evidence for the actual CE/PE, plus UI/Telegram visibility. Mere `fno_available=True` on an underlying does not qualify an options contract.
9. **External disk recovery:** test unavailable/unmounted/replaced `QuantTermStorage` under controlled maintenance. Workers must fail closed and preserve last-good artifacts, not create fresh production state under an unmounted path. Reattach and restart only after preflight.
10. **Public launch:** independent TLS/reverse-proxy auth, CORS/CSRF, rate limits, secret scanning, dependency vulnerability checks, publicly exposed route review, data licensing, and access logs. Local API and GitHub CI alone cannot certify internet exposure.

## Commands: on host, only after exact SHA passes all GitHub gates

The following are read-only and do not restart services or place orders:

```bash
cd ~/0to100
# Set this from the GitHub-reviewed release candidate, NOT from the Mac's
# current HEAD; otherwise an outdated checkout can attest to itself.
export EXPECTED_QT_SHA='<paste-exact-40-character-green-PR-head-SHA>'
# For a private local operator desk (the ordinary Mac deployment):
python3 scripts/quantterm_readiness_gate.py --expected-sha "$EXPECTED_QT_SHA"
# A public spectator deployment additionally needs --public:
# python3 scripts/quantterm_readiness_gate.py --public --expected-sha "$EXPECTED_QT_SHA"
curl -sS -m 5 -w '\nHTTP %{http_code} %{time_total}s\n' http://127.0.0.1:8765/api/health
sysctl vm.swapusage
memory_pressure -Q
df -h /Volumes/QuantTermStorage
grep -E '\[SCAN STAGE\]|\[SCAN OVERLAY\]|\[API SLOW\]' /Volumes/QuantTermStorage/QuantTerm/runtime/logs/service/{market_ops,market_api}.log | tail -n 50
```

Readiness gate verdicts are `HOLD` or `LOCAL_READ_GATE_PASS`; the latter is strictly a *local read-path* proof. It does not certify public internet safety, live order execution, research expectancy, or a completed paper trade.

## Read-only API soak after deploying a verified SHA

The local read acceptance probe above checks functional gate truth once. A
separate **20-cycle read soak** checks request latency and freshness stability
without making mutations, placing paper orders, or restarting services:

```bash
cd ~/0to100
python3 scripts/quantterm_read_soak.py --samples 20 --p95-budget-seconds 3 --expected-sha "$EXPECTED_QT_SHA"
```

With `--expected-sha`, all read samples must identify the full 40-character
reviewed build advertised by `/api/health`. Missing identity, truncated SHA,
or a mid-soak build change is HOLD even if latency looks good. The installed
launchd service and canonical manual launcher set `QT_BUILD_SHA`; the API
health handler does not run git or expose environment secrets.

It samples four local GET endpoints sequentially (health, access, dashboard,
decision-simulation-gate). HTTP 200 is not sufficient: /api/health must identify
the canonical QuantTerm API, report lifecycle READY, operational_ready=true,
evidence_ready=true, and positively verify that live money remains locked.
A DEGRADED/RECOVERING API or stale evidence fails the release soak even if
every response is fast. It fails on any unsafe operational read or when
the endpoint's nearest-rank p95 exceeds three seconds. The default one-second interval and mandatory 20 cycles cross the 15-second
Decision Simulation TTL; reducing the interval below one second is rejected
so a warm-cache burst cannot masquerade as a durability soak. Single-flight cache
TTL rollover may legitimately return a short **labelled, fail-closed STALE**
snapshot while the asynchronous refresh runs: the soak allows at most two
non-FRESH samples per 20-cycle window and no more than two consecutive
non-FRESH samples. It rejects DEGRADED or unknown status outright, and
requires at least 90% FRESH responses per cached endpoint. A successful
soak demonstrates **only local read-path stability** at the sampled cadence,
not external concurrency capacity, actual scan throughput, correct broker
fills, current F&O option-chain data, or public deployment security.

CSV session filenames alone are never proof of a completed NSE archive: when
the API price store is unloaded, the lightweight inventory validates the
newest file's required parser columns, equity price/volume samples and
embedded trading date where available. A corrupt or renamed prior-session
file fails closed instead of promoting its filename; scans still rely on the
authoritative full OHLCV store, not the metadata-only read.

## Rollout

- Keep the canonical branch + running Mac unchanged while the consolidated PR is draft or has pending/failed checks.
- Merge only after the full exact-head CI pass; never force auto-merge of partial tests.
- Use the existing canonical update/restart script after a known-green release, with a runtime backup and host-mounted volume verified first.
- Verify the local read gate and capture performance at rest; perform production paper + Telegram soak; roll back if the live lock, source lineage, latency or runtime health deviates.
- Keep real-money trading disabled unless there is a **separate** explicitly authorized live-money safety review; this engineering work does not authorize it.
