# QuantTerm Terminal Operations

## Recommended local start

```bash
cd ~/0to100
source venv/bin/activate
chmod +x scripts/run_quantterm.sh
bash scripts/run_quantterm.sh
```

The stack launcher starts or reuses the single autonomy supervisor, starts the local FastAPI bridge on
`127.0.0.1:8765`, and starts the dedicated React terminal on `127.0.0.1:5173`.

## Visible autonomy contract

`python main.py autonomy` must never look silently idle. The console prints:

- startup PID, root, interval and PAPER/LIVE safety mode;
- a heartbeat every 30 seconds with state, session phase, durable job counts, next pending job and
  active failures;
- `JOB START` before a handler begins;
- `JOB DONE` with final status, elapsed time, attempt and summary;
- state transitions;
- a traceback and retry message for unexpected tick exceptions.

One unexpected tick exception is contained and retried. Repeated exceptions degrade capability rather
than silently terminating the process. A second supervisor invocation exits deliberately and prints
the PID and last heartbeat of the existing lock owner.

A separate ephemeral `logs/autonomy/runtime.json` pulse remains fresh while a long job is running.
Durable state and job truth remain in `status.json` and `jobs.db`; the runtime pulse is process
liveness only.

## Data-plane contract

The terminal never owns market data. It projects these authoritative stores:

- verified Kite snapshot: `logs/snapshots/<active-id>`;
- official NSE daily history: `logs/bhav/*.csv` plus the canonical persisted
  `logs/bhav/store_cache.pkl`;
- whole-market product scan: `logs/product/latest_scan.json`;
- long-term shortlist: `logs/product/latest_long_term_scan.json`;
- paper book: `logs/intelligence/intel_book.json`.

The autonomy supervisor and FastAPI bridge are separate processes. The bhavcopy in-memory map is
therefore loaded lazily from the same persisted cache in every reader process. It is not a second data
store.

The terminal exposes the full readiness chain:

```text
verified snapshot
→ official bhavcopy history and depth
→ saved whole-market scan
→ long-term technical shortlist
→ current fundamentals coverage
```

A missing stage is displayed as a blocker, never translated into zero opportunities. `Prepare history
& run scan` first loads the persisted cache, rebuilds from official CSV files already on disk, and only
then uses the canonical NSE downloader/builder when history is still absent.

Current fundamentals are a present-day Screener.in/cache snapshot. They are not publication-dated and
must never be inserted into historical backtests.

## Critical-overdue semantics

`CRITICAL_OVERDUE` represents current control-plane work, not historical queue debris:

- only the newest pending recurring intent of each job type is considered;
- a job type already running is not simultaneously reported overdue;
- old intraday paper cycles remain visible in the ledger but do not degrade the whole organisation as
  a critical infrastructure outage;
- a genuinely current overdue authentication/data/risk dependency still raises an incident.

## Terminal Automation workspace

The Automation view refreshes every five seconds and projects the authoritative autonomy files and
SQLite ledger. It shows:

- process state, scheduler-owner PID and heartbeat;
- active worker job and elapsed time;
- exact `allowed`, `limited`, `read_only` or `blocked` capabilities;
- active failures and live-feed health;
- recent durable jobs and their result or blocker;
- data-pipeline readiness and blockers;
- recent supervisor dialogue;
- whitelisted owner controls.

The web frontend has no broker-order endpoint. Controls are durable requests handled by the existing
single supervisor.

## Ports

- Dedicated terminal: `http://127.0.0.1:5173`
- Local read-only/control API: `http://127.0.0.1:8765`
- Optional Streamlit fallback: `http://127.0.0.1:8501`

The React terminal is the primary professional interface. The Streamlit app remains a recovery and
compatibility surface; its plain-language Home page stays the fallback default.

## Shutdown

Press `Ctrl-C` in the stack-launcher terminal. Child API/frontend processes and a launcher-owned
autonomy process are stopped cleanly. An autonomy supervisor that was already running before the
launcher is left running.
