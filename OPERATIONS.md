# QuantTerm Operations

This is the short operator runbook. Detailed host procedures remain in `docs/MACBOOK_RUNBOOK.md`.

## Start and stop the complete stack

Start:

```bash
cd ~/0to100
bash scripts/run_quantterm_complete.sh
```

Stop from any terminal:

```bash
cd ~/0to100
bash scripts/stop_quantterm.sh
```

The canonical desk is served at `http://127.0.0.1:5173`.

The stop command signals only the recorded complete-stack supervisor and lets that owner perform its bounded child cleanup. It deliberately refuses to kill arbitrary processes merely because they occupy QuantTerm's ports, and it fails closed if the recorded PID has been reused by a non-QuantTerm process.

Do not start separate copies of the API, desk, supervisor, or market-ops worker in extra terminals. The complete-stack launcher owns the local process tree.

## Persistent MacBook install

```bash
cd ~/0to100
git fetch origin
git checkout claude/build-ai-trading-system-miHHd
git pull --ff-only origin claude/build-ai-trading-system-miHHd
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

The installed service must be pinned to the exact validated source revision. After updating production code, reinstall the host service so runtime and source do not silently diverge.

## What “healthy” means

Do not rely on one green badge. Check independent lanes:

- official market history/readiness;
- canonical market scan;
- operations worker;
- autonomy/scheduler;
- paper execution;
- learning/outcome jobs;
- broker/live-data lane when configured.

A broker-login problem is not the same thing as an official-history failure.

## Recovery rule

Recover the canonical owner instead of starting a second copy.

- If a job is stuck, inspect **System** and the durable operation state.
- If the complete stack is down, restart the complete-stack launcher.
- If persistent host state is inconsistent, follow `docs/MACBOOK_RUNBOOK.md`.
- Do not delete durable runtime state merely to make a health badge green.

## Safety

PAPER is the proving ground. Live-money execution is not implied by a healthy app, green CI, a profitable backtest, or a learning-model promotion. Live authorization is a separate certification and owner-approval process.
