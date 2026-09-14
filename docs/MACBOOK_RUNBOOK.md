# QuantTerm MacBook runbook

This is the canonical operator path for the production branch on macOS.

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

## 3. Install/update the canonical Mac host

Use the Mac deployment entrypoint, not the generic host installer:

```bash
bash deploy/setup_mac.sh
```

`deploy/setup_mac.sh` verifies or safely reattaches the configured external APFS sparsebundle, proves the existing durable runtime and canonical symlink, retires historical launchd owners, resolves npm for launchd, stops the old canonical owner before environment mutation, and then delegates to the strict existing-runtime host installer.

Production macOS must never create a replacement runtime when the external storage is missing. A missing/wrong external runtime is a blocker, not a reason to fall back to the internal disk.

Default storage contract:

```text
external volume:  /Volumes/Expansion
sparsebundle:      /Volumes/Expansion/QuantTermStorage.sparsebundle
mounted APFS:      /Volumes/QuantTermStorage
durable runtime:   /Volumes/QuantTermStorage/QuantTerm/runtime
canonical link:    ~/Library/Application Support/QuantTerm/runtime
```

Do not delete the durable runtime during upgrades. It contains operating state and evidence.

## 4. Check status

```bash
scripts/quantterm_status.sh \
  --runtime-root "/Volumes/QuantTermStorage/QuantTerm/runtime" \
  --manager launchd
```

Healthy operation is not the same as "all market data exists". During startup or when an upstream source is unavailable, QuantTerm may truthfully show degraded, waiting, stale, or missing lanes. That is expected; do not bypass safety gates to make the screen green.

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

Because deployment uses a mutable checkout, rollback is not atomic. After pulling a new known-good production commit, re-run the canonical Mac installer so storage preflight, exact SHA, service definition and health are revalidated:

```bash
bash deploy/setup_mac.sh
scripts/quantterm_status.sh \
  --runtime-root "/Volumes/QuantTermStorage/QuantTerm/runtime" \
  --manager launchd
```

If an upgrade must be reverted, explicitly restore the old checkout first, then reinstall:

```bash
git checkout <previous-known-good-sha>
bash deploy/setup_mac.sh
scripts/quantterm_status.sh \
  --runtime-root "/Volumes/QuantTermStorage/QuantTerm/runtime" \
  --manager launchd
```

Do not force the service to run when its pinned SHA differs from the checkout.

## 7. MacBook operating rules

- Keep the Mac connected to power during market sessions and long first-history bootstrap work.
- The service uses `caffeinate -i` to resist idle sleep, but closing the lid can still suspend the machine.
- Leave the lid open during unattended market operation unless you have independently verified your clamshell/power setup.
- Avoid memory-heavy parallel research jobs on constrained hardware while the live desk is operating.
- Keep one QuantTerm stack, one canonical launchd owner, one durable runtime root and one production checkout.

## 8. Safety contract

The installed host remains PAPER/SHADOW only. The verified live-execution interlock must remain locked and unauthorized; broker mutations are not part of this deployment path. Forward-evidence rules remain authoritative: missing evidence means no profitability claim.

## Morning smoke check

```bash
git rev-parse HEAD
scripts/quantterm_status.sh \
  --runtime-root "/Volumes/QuantTermStorage/QuantTerm/runtime" \
  --manager launchd
```

Then open `http://127.0.0.1:5173` and confirm the UI reports the same backend truth as the status command. Empty/unavailable market data must appear empty, stale, blocked or degraded rather than be replaced with synthetic success.
