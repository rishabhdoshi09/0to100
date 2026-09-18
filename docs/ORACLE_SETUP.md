# Oracle/VPS par QuantTerm 24/7

This guide covers the QuantTerm side of the deployment. Cloud-provider account,
pricing and free-tier rules can change, so verify those directly with the
provider before creating infrastructure.

The QuantTerm deployment invariant is fixed: one production checkout, one
durable runtime, and one canonical installed-host supervisor.

## Step 1 — VM

Use a supported Linux VM with enough disk/RAM for the full market data and
research workload. Create a normal login user with sudo access; do **not** run the
QuantTerm installer as root.

## Step 2 — Production checkout

```bash
ssh ubuntu@YOUR_PUBLIC_IP

# If the repository requires credentials, configure Git access first.
git clone --branch claude/build-ai-trading-system-miHHd \
  https://github.com/rishabhdoshi09/0to100.git ~/0to100
cd ~/0to100
```

`claude/build-ai-trading-system-miHHd` is the production branch. Historical
integration/research branches are not deployment targets.

## Step 3 — Canonical server installer

```bash
bash deploy/setup_server.sh
```

`setup_server.sh` installs OS/runtime dependencies, prepares the venv, retires
historical split QuantTerm units, and delegates to
`scripts/install_quantterm_host.sh`.

The resulting systemd user unit is **`quantterm.service`**. It starts
`product.host_entrypoint`, which owns exactly one `product.host_supervisor`; the
supervisor owns the required autonomy, market-ops, market API, report API and
frontend children.

No separate UI/autonomy service should be installed.

## Step 4 — Credentials and exact host re-install

Edit the secure env file:

```bash
nano ~/0to100/.env
chmod 600 ~/0to100/.env
```

Then rerun the installer so preflight, exact SHA, service definition and startup
health are all revalidated:

```bash
cd ~/0to100
bash deploy/setup_server.sh
```

## Step 5 — Status

```bash
cd ~/0to100
scripts/quantterm_status.sh \
  --runtime-root "$HOME/.local/state/quantterm/runtime" \
  --manager systemd
```

Healthy means the service is active, supervisor heartbeat is fresh, all required
children are alive+healthy, and the canonical live-execution interlock is
verified locked and unauthorized.

## Access

Desk default: `http://<server-ip>:5173`. Prefer a private VPN/network path rather
than exposing the desk directly to the public internet.

## Broker login when required

```bash
cd ~/0to100
./venv/bin/python main.py login
```

Paper-only paths must remain independent of live broker authorization where the
product contract says so; broker-bound functionality still reports login/capability
blockers truthfully.

## Updates

```bash
cd ~/0to100
git checkout claude/build-ai-trading-system-miHHd
git pull --ff-only origin claude/build-ai-trading-system-miHHd
bash deploy/setup_server.sh
```

For a pure service restart without changing the deployment:

```bash
scripts/quantterm_restart.sh --manager systemd
scripts/quantterm_status.sh \
  --runtime-root "$HOME/.local/state/quantterm/runtime" \
  --manager systemd
```

## Logs

```bash
journalctl --user -u quantterm.service -f
```

## Product verification

```bash
python scripts/verify_quantterm_stack.py
python scripts/verify_quantterm_actions.py
python scripts/run_product_acceptance.py
```

The safe-action verifier can trigger real non-money research/data operations but
never unlocks live capital or submits broker orders. Product acceptance includes
the canonical durable paper-cycle decision path and accepts an evidence-backed
no-trade when the rules legitimately produce one.
