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

## Terminal Automation workspace

The Automation view refreshes every five seconds and projects the authoritative autonomy files and
SQLite ledger. It shows:

- process state, scheduler-owner PID and heartbeat;
- exact `allowed`, `limited`, `read_only` or `blocked` capabilities;
- active failures and live-feed health;
- recent durable jobs and their result or blocker;
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
