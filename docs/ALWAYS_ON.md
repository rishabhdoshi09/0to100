# QuantTerm — Always-On Operation

Canonical product: the **Vite/React desk**. The ordinary foreground launcher is:

```bash
bash scripts/run_quantterm_complete.sh
```

For unattended operation on the supported host path, install the persistent
host supervisor instead of keeping a terminal open.

## MacBook (canonical path)

From the production/default branch:

```bash
cd ~/0to100
git pull --ff-only
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

`deploy/setup_mac.sh` is retained only for backward compatibility and now routes
to the same canonical installer. Do **not** maintain a second custom launchd
agent for the same checkout.

The canonical host deployment provides:

- a persistent runtime root outside the mutable Git checkout;
- copy/verify migration that preserves the source and blocks divergent roots;
- exact checkout-SHA pinning so mixed-code operation fails closed;
- one host supervisor with bounded child and host restart protection;
- bootstrap heartbeat/progress during long first-history setup;
- truthful health/degraded/FAILED state and operational failure alerts;
- memory and macOS power-management preflight;
- post-session operating-report scheduling;
- a verified-locked live-execution interlock.

### Mac power behaviour

The launchd service is wrapped with `/usr/bin/caffeinate -i`, which prevents
ordinary idle system sleep while the service is active. It does **not** make a
closed laptop lid safe for unattended execution: lid-close can still suspend a
MacBook. For the 2015 MacBook Air, keep the charger connected and the lid open
for unattended market operation unless you have separately validated a
supported closed-display setup.

The installer reports power-management warnings rather than pretending they are
fixed. Avoid relying on permanent global `pmset sleep 0` changes as the primary
safety mechanism.

### Operator controls

```bash
bash scripts/quantterm_status.sh
bash scripts/quantterm_restart.sh
bash scripts/quantterm_stop.sh
```

Status is authoritative only when it matches the installed exact SHA, heartbeat
is fresh, required children are healthy, and the live interlock remains verified
locked. A running process alone is not sufficient evidence of readiness.

## Daily broker session

QuantTerm can run non-broker research/official-data/learning lanes without a Kite
session. Broker-dependent lanes require real Zerodha credentials and a valid
session. When login is required on an interactive checkout:

```bash
cd ~/0to100
source venv/bin/activate
python main.py login
```

Paste the complete redirect URL when prompted. Never insert placeholder values
just to make readiness look configured.

## Updating the Mac checkout

The persistent service is intentionally pinned to the exact Git SHA that was
validated at install time. After pulling a new production release, reinstall the
service so the deployed SHA and checkout agree:

```bash
cd ~/0to100
git pull --ff-only
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

Do not force a mismatched checkout to run under an older service definition.

### Rollback limitation

Rollback is **not atomic** today because the service runs from a mutable Git
checkout rather than immutable release directories. If a new release must be
reverted, restore the prior Git SHA and reinstall/validate that exact checkout.
The installer/service definition alone cannot roll code back for you.

## Linux / VPS

`deploy/setup_server.sh` remains the existing server-oriented compatibility path.
Its fresh-clone path is still explicitly pinned to the accepted server branch so
it cannot accidentally land on the repository's historical default:

```bash
git clone --branch cursor/live-terminal-contract-858e https://github.com/rishabhdoshi09/0to100.git
```

Before unattended use on a new Linux host, apply the same evidence standard as
on the Mac: exact code identity, durable runtime state, bounded restart/recovery,
truthful data-source health, and PAPER/SHADOW-only safety until separately
accepted.

## What “healthy” means

A healthy QuantTerm host is not merely a green web page. It means the complete
operating loop is coherent:

1. intended data sources are attempted and provenance/freshness is visible;
2. stale, missing, partial or blocked sources degrade honestly;
3. scheduled acquisition/scanning/decision/outcome/report jobs execute without
   manual triggering when they are due;
4. durable state survives process restart and checkout refresh;
5. scanner, decisions, journal, forward evidence, reports and UI all project the
   same backend truth;
6. portfolio/risk gates stay enforced;
7. live execution remains verified locked;
8. forward performance claims remain forbidden until the evidence contract is
   actually satisfied (`MIN_SAMPLE=30`).

Real-market PAPER_FORWARD outcomes still have to be earned on a host that can
reach the required upstreams. Synthetic or replay evidence is useful for proving
machinery, never for pretending the market has validated the strategy.
