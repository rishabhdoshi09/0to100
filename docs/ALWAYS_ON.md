# QuantTerm ko 24/7 Chalana (Always-On Setup)

Canonical product: the **Vite/React desk**. Local interactive use is still one
command: `bash scripts/run_quantterm_complete.sh` → http://127.0.0.1:5173.

Installed/always-on operation is different by design: both Linux and macOS now
use one canonical lifecycle:

`product.host_install → product.host_entrypoint → product.host_supervisor`

That supervisor owns exactly the required QuantTerm children. Historical split
UI/autonomy services are retired and must not be installed again.

---

## Option A: Linux VPS / always-on server

Use a normal login user with sudo access. `deploy/setup_server.sh` deliberately
refuses to run as root because the canonical service is a user-level systemd
service with reboot persistence verified through user lingering.

### 1. Server par login karo

```bash
ssh ubuntu@YOUR_SERVER_IP
```

### 2. Production checkout clone karo

```bash
git clone --branch claude/build-ai-trading-system-miHHd https://github.com/rishabhdoshi09/0to100.git
cd 0to100
```

`claude/build-ai-trading-system-miHHd` is the production branch. Do not deploy a
historical research/integration branch from old notes or screenshots.

### 3. Canonical installed host

```bash
bash deploy/setup_server.sh
```

The installer prepares Python/Node, retires historical system-level QuantTerm
units, then delegates to `scripts/install_quantterm_host.sh`. The resulting
**single** user service is `quantterm.service`, whose process is
`product.host_entrypoint`; it owns the supervisor and the desk/API/report/
market-ops/autonomy children.

Status:

```bash
scripts/quantterm_status.sh \
  --runtime-root "$HOME/.local/state/quantterm/runtime" \
  --manager systemd
```

### 4. Phone/laptop se kholo

Desk default: `http://YOUR_SERVER_IP:5173`. For remote private access, use your
preferred VPN/private-network setup instead of exposing the desk publicly.

---

## Option B: Apna Mac always-on

This repository's Mac deployment is intentionally tied to the configured
external APFS runtime. `deploy/setup_mac.sh` verifies/attaches the existing
sparsebundle, refuses to create a missing runtime, retires historical launchd
owners, resolves npm for launchd, and installs exactly one canonical launchd
service (`com.quantterm.desk`).

```bash
cd ~/0to100
bash deploy/setup_mac.sh
```

The default storage contract is:

- external volume: `/Volumes/Expansion`
- sparsebundle: `/Volumes/Expansion/QuantTermStorage.sparsebundle`
- mounted APFS volume: `/Volumes/QuantTermStorage`
- durable runtime: `/Volumes/QuantTermStorage/QuantTerm/runtime`
- canonical symlink: `~/Library/Application Support/QuantTerm/runtime`

Override only through the documented `QT_STORAGE_*` variables when the physical
layout is intentionally different. Missing/wrong storage is a blocker, not a
reason to fall back to the internal disk.

Status:

```bash
scripts/quantterm_status.sh \
  --runtime-root "/Volumes/QuantTermStorage/QuantTerm/runtime" \
  --manager launchd
```

For unattended operation keep the Mac powered and configured so lid/sleep does
not suspend the machine. The host preflight reports power-management risk rather
than silently claiming readiness.

---

## Option C: Raspberry Pi / old Linux laptop

Use the same Linux flow and `deploy/setup_server.sh`. The deployment invariant is
unchanged: one installed-host supervisor, one durable runtime, no split service
owners.

---

## Roz ka broker login (jab required ho)

```bash
cd ~/0to100
./venv/bin/python main.py login
```

Paper operation must remain broker-independent where designed; broker login
state is still surfaced truthfully and remains mandatory for broker-bound data or
future live capabilities.

## Updates

Production branch update ke baad canonical lifecycle ko use karo:

```bash
cd ~/0to100
git checkout claude/build-ai-trading-system-miHHd
git pull --ff-only origin claude/build-ai-trading-system-miHHd
scripts/quantterm_restart.sh --manager systemd   # Linux
# or: scripts/quantterm_restart.sh --manager launchd   # Mac
```

For a release installation, prefer rerunning the platform installer so exact-SHA
preflight, service definition, runtime identity, and startup health are all
revalidated.

## Health / verification

Read-only product surface verification:

```bash
python scripts/verify_quantterm_stack.py
```

Real safe-action verification (no broker order, no live unlock):

```bash
python scripts/verify_quantterm_actions.py
```

Canonical product acceptance, including the durable paper-cycle path:

```bash
python scripts/run_product_acceptance.py
```

Linux service logs:

```bash
journalctl --user -u quantterm.service -f
```

A green UI alone is not acceptance. The installed host is healthy only when the
service/supervisor heartbeat, all required children, persistent runtime and
canonical verified live-money lock agree.
