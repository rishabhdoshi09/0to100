# ⚡ QuantTerm

**An evidence-driven, self-learning trading terminal for NSE and US equities** —
whole-market scanning, expected-value ranking, portfolio/risk controls, a gated
paper autopilot, and a Brain that reads every subsystem and produces a unified
posture.

> The edge is not a better breakout pattern. The edge is a system that
> **measures, allocates, and retires** its own strategies faster and more
> honestly than its operator's emotions would.

---

## What it does (one loop)

```
data → signals → EV ranking → risk gates → paper decisions → outcomes → learning
  ▲                                                                    │
  └────────────────── every resolved outcome recalibrates ─────────────┘
```

- **Scan** the NSE universe on schedule (and US indices on demand): confirmed
  breakouts, VCP/patterns, pre-breakout, pullback and momentum families.
- **Rank by Expected Value**, not points: forward outcomes feed calibrated edge
  estimates, with evidence floors so small lucky samples cannot masquerade as
  proof.
- **Gate** decisions through signal quality, market/sector context and portfolio
  risk controls, including position concentration and aggregate risk limits.
- **Track** taken and rejected decisions, then resolve outcomes and calibration
  so the system can measure whether its own probabilities deserve trust.
- **Degrade honestly**: missing, stale, blocked or incomplete upstream data is
  surfaced as such rather than replaced by fabricated rows or fake success.
- **Operate PAPER/SHADOW only by default.** The canonical live-execution
  interlock must remain verified locked until a separate, explicit deployment
  decision is made; release validation never authorizes live money.

## Canonical product path (one command)

The product UI is the **Vite/React desk**. Streamlit is not the product path.
One foreground command owns the local stack (desk, terminal API, report API,
autonomy and market operations):

```bash
cd ~/0to100
bash scripts/run_quantterm_complete.sh
```

Home is `http://127.0.0.1:5173`. On an interactive local machine the launcher
opens Home after the desk is reachable. `./quantterm.sh` is a small exec wrapper
of the same command.

A clean checkout can run without Zerodha credentials. Broker-dependent lanes
remain unavailable until real credentials/session are present; official-data,
research, replay, settlement and learning lanes must report their actual state.
If Kite login is required, paste the full redirect URL when prompted — you do
not need to extract `request_token` manually.

`scripts/run_desk.sh` is only a compatibility wrapper. Do not start Streamlit,
and do not start a second copy of the stack.

Fresh clone:

```bash
git clone https://github.com/rishabhdoshi09/0to100.git
cd 0to100
cp .env.example .env          # optional: add KITE_API_KEY / KITE_API_SECRET
bash scripts/run_quantterm_complete.sh
```

## MacBook persistent host mode

For the intended install-once / leave-running Mac workflow, use the **canonical
host installer**, not a hand-written launchd plist:

```bash
cd ~/0to100
git pull --ff-only
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

The installer performs real-market/preflight checks, establishes the persistent
runtime root, pins the installed service to the exact checkout SHA, installs the
bounded host supervisor and keeps live execution locked. On macOS the service is
wrapped with `caffeinate -i`; that prevents ordinary idle system sleep while it
runs, but **closing the MacBook lid can still suspend the machine**. Keep the lid
open for unattended operation unless the hardware is being used in a supported
closed-display setup.

Existing users of `bash deploy/setup_mac.sh` are safe: that historical command
is now only a compatibility wrapper around the canonical host installer.

Operator controls:

```bash
bash scripts/quantterm_status.sh
bash scripts/quantterm_restart.sh
bash scripts/quantterm_stop.sh
```

Persistent deployment is intentionally fail-closed. If the checkout SHA no
longer matches the installed service after an update, reinstall/validate the new
exact SHA instead of silently running mixed code. Old-code rollback is **not
atomic** because deployment still runs from a mutable Git checkout; reverting an
upgrade requires checking out the prior SHA and reinstalling the service.

Details: `docs/ALWAYS_ON.md`.

## Invariants (the non-negotiables)

1. **No fake data, ever** — no data → skip/degrade, never fabricate.
2. **Stale must look stale** — freshness is labeled everywhere.
3. **Live money stays locked unless separately authorized and verified.**
4. **Telegram/automation remains paper-safe.**
5. **1% risk/trade · 10% per name · 5% total open risk.**
6. **Evidence over vibes** — `MIN_SAMPLE=30`; small samples earn no claim.
7. **Every decision is outcome-tracked** — including rejections.

## Testing

```bash
python -m pytest tests/
```

Canonical CI is hermetic/network-free for the blocking money-critical gate,
then validates launcher/supervisor scripts, compiles modules and runs controlled
integration checks. Product Acceptance separately boots the desk, proves the
live interlock locked and validates durable restart/recovery on the exact SHA.

Issue #92 live Definition of Done (stack must already be running):

```bash
python scripts/verify_issue92_dod.py
```

That writes `docs/issue92_live_dod_proof.md` and
`docs/issue92_live_dod_proof.json` against the real local API, including the
tested git SHA. It does not mock handlers.

## Architecture

The full map lives in [`CLAUDE.md`](CLAUDE.md) — data layer, signal/ranking
layer, risk layer, execution boundary, decision journal/learning system,
research/evidence infrastructure, and the **Vite/React desk** (Home · Market
Scanner · Recommendations · Market Reports · Stock Intelligence). Archived
Streamlit pages under `ui/` are not started.

Historical research branches such as `overhaul/evidence-lab` are not the current product path.

*Canonical launcher: `bash scripts/run_quantterm_complete.sh`. Canonical
persistent host installer: `bash scripts/install_quantterm_host.sh`.*
