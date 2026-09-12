# QuantTerm MacBook runbook

This is the canonical operator path for the production branch on macOS Monterey.

## 1. Pull the verified production branch

```bash
cd ~/0to100
git fetch origin
git checkout claude/build-ai-trading-system-miHHd
git pull --ff-only origin claude/build-ai-trading-system-miHHd
git rev-parse HEAD
```

Keep the printed SHA. QuantTerm's host service is pinned to the exact checkout SHA so code and service state cannot silently drift.

## 2. Configure secrets once

```bash
cp -n .env.example .env
chmod 600 .env
```

Edit `.env` and add the required Kite credentials. Never commit `.env`.

## 3. Install the persistent host service

```bash
bash scripts/install_quantterm_host.sh
```

The installer performs preflight checks, prepares the persistent runtime root, validates the exact checkout SHA, and installs the macOS launchd service. It must fail closed if the runtime root, permissions, live-execution interlock, or host prerequisites are unsafe.

The durable runtime lives outside the mutable Git checkout under:

```text
~/Library/Application Support/QuantTerm/runtime
```

Do not delete that directory during upgrades. It contains durable operating state and evidence.

## 4. Check status

```bash
bash scripts/quantterm_status.sh
```

Healthy operation is not the same as "all market data exists". During startup or when NSE/Zerodha is unreachable, QuantTerm may truthfully show degraded, waiting, stale, or missing lanes. That is expected; do not treat a degraded status as a reason to bypass safety gates.

The desk is served at:

```text
http://127.0.0.1:5173
```

## 5. Interactive foreground run

For troubleshooting or a one-off foreground session, use the canonical complete-stack launcher:

```bash
bash scripts/run_quantterm_complete.sh
```

Do not start Streamlit or a second copy of the stack in another terminal.

## 6. After pulling a new production SHA

Because deployment uses a mutable checkout, rollback is **not atomic**. The installed service is pinned to the SHA that was validated at install time. After pulling a new production commit, reinstall/validate it:

```bash
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

If an upgrade must be reverted, explicitly restore the old checkout first, then reinstall the service:

```bash
git checkout <previous-known-good-sha>
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

Do not force the service to run when its pinned SHA differs from the checkout.

## 7. MacBook operating rules

- Keep the Mac connected to power during market sessions and long first-history bootstrap work.
- The service uses `caffeinate -i` to resist idle sleep, but closing the lid can still suspend the machine.
- Leave the lid open during unattended market operation unless you have independently verified your clamshell/power setup.
- Do not run memory-heavy parallel research jobs on the 8 GB machine while the live desk is operating.
- Prefer one QuantTerm stack, one persistent runtime root, and one production checkout.

## 8. Safety contract

The normal host remains PAPER/SHADOW only. The verified live-execution interlock must remain locked; broker mutations are not part of this deployment path. `MIN_SAMPLE=30` and the forward-evidence rules remain authoritative: no market evidence means no profitability claim.

## Morning smoke check

After install or upgrade, these are the operator checks that matter:

```bash
git rev-parse HEAD
bash scripts/quantterm_status.sh
```

Then open `http://127.0.0.1:5173` and confirm that the UI reports the same backend truth as the status command. Empty or unavailable market data must appear empty/degraded rather than be replaced with synthetic results.
