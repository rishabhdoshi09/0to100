# Autonomy Closure — engineered operating contract

> **Historical record** (baseline `cae4811` on `overhaul/evidence-lab`).
> Not the current product path. Canonical launcher:
> `bash scripts/run_quantterm_complete.sh` (Vite/React desk). Streamlit is not started.

Baseline package: `cae4811` (`overhaul/evidence-lab`). This correction makes the dedicated
`python main.py autonomy` process the single scheduler and mutation owner. Streamlit is a read-only
control room that writes durable owner controls; it starts no brain, news loop, scanner, or worker.

## Closed safety/wiring gaps

- **Pre-mutation entry gate:** `CycleContext` carries entry permission, block reason, capability
  failures and fresh live symbols. The authoritative intelligence runtime manages existing positions
  but cannot create a TradeIntent or call `PaperBook.open_position` while entries are blocked.
- **Headless scanning:** `scan/market_scan_service.py` is the one Streamlit-free whole-market scan
  service used by the supervisor and retail adapter.
- **Dependency-aware jobs:** SQLite jobs migrate in place with `blocked_on` metadata. Auth/data/EOD
  recovery requeues the same logical job instead of creating duplicates or leaving it blocked.
- **Real auth health:** token missing, expired session, provider outage and valid session are separate.
  Credential text is redacted. `main.py login` persists the daily token atomically and queues refresh.
- **One scheduler:** UI pages queue controls only. The supervisor owns auth, instruments, Kite data,
  official bhavcopy, data-quality tasks, index warm-up, news, scans, paper cycles, EOD outcomes,
  learning and bounded research.
- **EOD ordering:** post-close refresh has a separate identity. Outcome resolution waits until the
  active snapshot actually contains the completed session; learning waits for outcomes.
- **Governed evolution:** canonical PaperBook outcomes produce measured diagnostics, evidence gaps,
  constrained versioned hypotheses, preregistration, canonical evaluation, adversarial challenge,
  durable failed-memory and legal paper nomination. Missing institutional evidence cannot promote.
- **Live observation:** the supervisor owns the existing data-only Kite ticker/overlay. Fresh ticks are
  required only by strategies that explicitly request live confirmation; EOD strategies remain valid.
- **Deployment:** Linux and macOS setup scripts install separate UI and autonomy services on
  `overhaul/evidence-lab` and remove the obsolete combined service.

## Failure contract

A failure reduces only dependent capabilities. Auth/stale data/risk/reconciliation blocks new paper
risk; existing positions remain manageable when trustworthy prices and durable persistence exist.
Corporate-action or point-in-time-universe gaps remain explicit blocked data tasks and do not get
fabricated. News failure does not become “no trade.” Live execution remains structurally locked.

## Verification performed in the supplied package

- Focused autonomy/runtime safety tests.
- Product, retail, Kite activation/snapshot, paper-autonomy and news ownership tests.
- Compilation of every changed Python file.
- `git diff --check`.
- `bash -n` on deployment scripts and XML parsing of launchd plists.

The complete suite is intentionally delegated to repository CI to avoid repeated local full-suite
cycles. Genuine Zerodha/live-feed and elapsed paper-forward evidence remain external validation, so
this milestone is **AUTONOMY OPERATIONALLY CLOSED**, not PAPER PROVEN.
