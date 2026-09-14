# QuantTerm autonomy deployment

Installed QuantTerm uses **one canonical service owner**, not separate UI and
autonomy services.

Lifecycle:

`product.host_install → product.host_entrypoint → product.host_supervisor`

The host supervisor owns the required children (`autonomy`, `market_ops`,
`market_api`, `report_api`, `frontend`) and writes one truthful host status over
the single durable runtime. Historical split systemd/launchd units are retired
and the platform installers explicitly remove them.

Local interactive launcher remains:

```bash
bash scripts/run_quantterm_complete.sh
```

That is a developer/operator foreground stack. It is **not** the installed
always-on service entrypoint.

## Linux (systemd user service)

Production checkout:

```bash
git clone --branch claude/build-ai-trading-system-miHHd \
  https://github.com/rishabhdoshi09/0to100.git
cd 0to100
bash deploy/setup_server.sh
```

The resulting user service is `quantterm.service` and its executable path is the
canonical host entrypoint. Status/restart:

```bash
scripts/quantterm_status.sh \
  --runtime-root "$HOME/.local/state/quantterm/runtime" \
  --manager systemd
scripts/quantterm_restart.sh --manager systemd
```

Do not copy or install historical `quantterm-ui` / `quantterm-autonomy` units.

## macOS (launchd + external APFS runtime)

```bash
cd ~/0to100
bash deploy/setup_mac.sh
```

The installer first verifies the configured APFS sparsebundle/runtime, boots out
historical QuantTerm launchd owners, resolves npm for non-interactive launchd,
and installs exactly one `com.quantterm.desk` agent through the canonical host
installer.

The strict Mac installer is adoption-only: the durable runtime must already
exist and carry its host runtime manifest. It pins the verified external runtime
throughout installation and fails closed if storage disappears; it never creates
a replacement `/Volumes/...` runtime on the internal disk.

Status/restart:

```bash
scripts/quantterm_status.sh \
  --runtime-root "/Volumes/QuantTermStorage/QuantTerm/runtime" \
  --manager launchd
scripts/quantterm_restart.sh --manager launchd
```

## Autonomy ownership

The autonomy child owns the durable research/paper job loop under the host
supervisor. Retail/operator controls persist requests; they do not create a
second scheduler or mutation owner.

Canonical paper-entry authority remains:

`research.autonomy.jobs.Deps.run_paper_cycle → run_reco_paper_cycle → brain.intel_book`

A committee BUY/intention is not an executed trade. `TRADED` is valid only when
that canonical PaperBook path persists a real paper fill. An evidence-backed
`NO_ELIGIBLE_TRADE` or legitimate closed entry window is valid truth and must not
be converted into a fake trade.

## Health and safety

Installed-host acceptance requires all of the following to agree:

- exact deployed Git SHA;
- active service manager entry;
- fresh host-supervisor heartbeat;
- all required children alive and healthy;
- one persistent runtime root;
- truthful freshness/data/operation projections;
- canonical live-execution state verified **locked** and **unauthorized**.

Live broker mutations remain blocked at the canonical broker boundary. Broker
credentials or environment feature flags are not authorization.

## Verification

```bash
python scripts/verify_quantterm_stack.py
python scripts/verify_quantterm_actions.py
python scripts/run_product_acceptance.py
```

The safe-action verifier performs real non-money operations and reversible
product writes; it never submits a broker order or unlocks live money. Product
acceptance exercises the durable paper-cycle decision path and restart recovery.
