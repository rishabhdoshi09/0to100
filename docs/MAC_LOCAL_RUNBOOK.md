# QuantTerm on the MacBook — operator runbook

This is the supported local workflow for the production branch on macOS. It is intentionally boring: one checkout, one production branch, one durable runtime, one supervisor.

## 1. Pull the exact production branch

```bash
cd ~/0to100
git fetch origin
git checkout claude/build-ai-trading-system-miHHd
git pull --ff-only origin claude/build-ai-trading-system-miHHd
git status --short
git rev-parse HEAD
```

`git status --short` should be empty before installation. Keep the printed SHA; the host installer pins the service to the checkout it validates.

## 2. Configure Zerodha only if you want broker-dependent lanes

On first foreground start QuantTerm can create `.env` from `.env.example`. For the always-on host service, configure real values before installing if you want Kite data:

```bash
cp -n .env.example .env
chmod 600 .env
```

Edit `.env` and set `KITE_API_KEY` and `KITE_API_SECRET`. Never commit `.env`. Missing broker credentials are allowed: QuantTerm must continue in PAPER/SHADOW research mode and report broker-dependent lanes as unavailable rather than invent data.

## 3A. Recommended: install once and leave it running

```bash
bash scripts/install_quantterm_host.sh
```

The installer performs host preflight, verifies the live-execution interlock is locked, prepares the persistent runtime root, pins the exact checkout SHA, and installs the macOS `launchd` service. If preflight fails, fix the reported condition instead of bypassing it.

Check it with:

```bash
bash scripts/quantterm_status.sh
```

Open the desk at:

```text
http://127.0.0.1:5173
```

Service controls:

```bash
bash scripts/quantterm_restart.sh
bash scripts/quantterm_stop.sh
```

After a later `git pull`, reinstall from the new checkout so the service is deliberately repinned and revalidated:

```bash
bash scripts/install_quantterm_host.sh
```

Do not leave an installed service pinned to an older SHA while running a newer mutable checkout.

## 3B. Foreground mode for a quick local session

If you do not want the always-on service yet:

```bash
bash scripts/run_quantterm_complete.sh
```

Leave that terminal open. The desk is `http://127.0.0.1:5173`. Use `Ctrl-C` to stop the complete foreground stack. Do not start a second copy in another checkout or terminal.

## 4. What healthy means

A healthy process is not the same thing as healthy market evidence. The UI may truthfully show `DEGRADED`, `HISTORY_NOT_READY`, `NO_MARKET_EVIDENCE`, stale lanes, or unavailable broker capability while the application itself is operating correctly.

Do not treat those states as installation failure unless `quantterm_status.sh` says the service/process is unhealthy. In particular:

- no fabricated rows should appear when NSE/Kite is unreachable;
- stale data must remain labelled stale;
- live execution must remain locked and unauthorized;
- PAPER_FORWARD evidence is earned only from real settled paper outcomes;
- fewer than 30 settled outcomes is not evidence of an edge.

## 5. MacBook operating rules

For the Early-2015 MacBook Air / 8 GB RAM target:

- keep the machine on AC power for long bootstrap or market-session runs;
- keep the lid open during sessions that must continue uninterrupted — `caffeinate -i` prevents idle sleep but cannot guarantee work continues through lid-close sleep;
- do not run another QuantTerm checkout simultaneously;
- avoid deleting `~/Library/Application Support/QuantTerm/runtime`; it is durable operating evidence, not a build cache;
- use the status script before assuming a quiet UI means the backend stopped.

## 6. Upgrade and rollback truth

Deployment currently runs from a mutable Git checkout. Rollback is therefore **not atomic**. If an upgrade fails after the checkout has moved, restoring an older launchd definition alone is insufficient because SHA verification will reject the newer checkout.

To return to a known prior production SHA:

```bash
git checkout <KNOWN_GOOD_SHA>
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

Do not use `git reset --hard` when you are unsure whether the working tree contains something you need. Preserve runtime/evidence data.

## 7. First real-market proof

Once the service is running on the Mac, the remaining proof is environmental rather than synthetic:

1. NSE/official-data egress succeeds from the Mac network.
2. Zerodha authentication succeeds if broker lanes are enabled.
3. Official history/bootstrap completes and survives restart.
4. Automated market jobs run without manual triggering.
5. The post-session operating report is generated automatically.
6. Reboot/login recovery is proven on the actual Mac.
7. PAPER_FORWARD settles real paper outcomes until the evidence floor is reached.

Until those facts exist, the system may be operationally ready but must not claim a profitable market edge.
