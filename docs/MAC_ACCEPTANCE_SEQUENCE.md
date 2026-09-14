# Mac / external-APFS acceptance sequence

Cloud Linux runners cannot prove this release. These are the smallest safe
commands that establish the fifteen host invariants on the owner's MacBook
against the real external APFS runtime.

Run them in order. Stop at the first failure and capture the output — a failure
here is the point of the exercise, not an obstacle to it.

Nothing below unlocks live execution, places a broker order, forces a trade, or
unmounts a volume by force.

---

## 0. Exact deployed SHA (invariant 1)

```bash
cd ~/0to100
git fetch origin
git checkout --detach 6a72475e7da85706d463e9693928cc851b67c6bc
git rev-parse HEAD          # must print 6a72475e7da85706d463e9693928cc851b67c6bc
git status --porcelain      # must be empty: a dirty tree is not a deployed build
```

## 1. Storage identity BEFORE anything starts (invariant 2)

```bash
bash scripts/quantterm_storage_preflight.sh
diskutil info /Volumes/QuantTermStorage | grep -E "Mount Point|File System Personality|Volume UUID"
ls -la ~/Library/Application\ Support/QuantTerm/runtime   # canonical link
du -sh /Volumes/QuantTermStorage/QuantTerm/runtime        # expect ~23 GB, not a fresh empty tree
```

A runtime that is suddenly small is the split-brain failure. Do not continue.

## 2. Install through the canonical macOS path (invariants 2, 3)

```bash
bash deploy/setup_mac.sh
```

Use this, not the generic wrapper. `setup_mac.sh` runs the external-APFS
storage preflight first, retires historical launchd owners, resolves npm for
launchd, stops the canonical owner **before** mutating the Python environment,
and only then delegates to the strict existing-runtime installer.

A direct `scripts/install_quantterm_host.sh` on Darwin also defaults to strict
existing-runtime mode, but it skips the storage preflight and the legacy-owner
retirement, so it is not the production path.

Do **not** set `QT_RUNTIME_ROOT_REQUIRE_EXISTING=0` on this machine. That
opt-out exists for genuine Linux first-install bootstrap and permits creating a
new empty runtime.

## 3. One canonical owner, all children healthy (invariants 3, 4)

```bash
launchctl list | grep -i quantterm     # expect ONLY com.quantterm.desk
bash scripts/quantterm_status.sh \
  --runtime-root "/Volumes/QuantTermStorage/QuantTerm/runtime" \
  --manager launchd
```

Any `com.quantterm.ui` or `com.quantterm.autonomy` still loaded is a retired
split owner and must be booted out before this release is accepted.

## 4. Where the chain actually stands (invariants 5, 6, 7)

```bash
PYTHONPATH=. python3 -m product.chain_status
```

This names the first stopped link and what clears it. Capture it verbatim —
it is the single most informative artifact for the whole run.

## 5. Data → scan → decision (invariants 5, 6)

```bash
curl -s localhost:8765/api/health        | python3 -m json.tool | head -30
curl -s localhost:8765/api/chain-status  | python3 -m json.tool
curl -s localhost:8765/api/decisions     | python3 -m json.tool | head -40
```

Confirm `market_session_date` and the scan-executed timestamp are **different
fields with different values**, and that the scan's record count is non-zero.

## 6. Paper outcome — trade OR evidence-backed no-trade (invariants 7, 8)

```bash
curl -s localhost:8765/api/paper-autopilot | python3 -m json.tool | head -60
```

`NO_ELIGIBLE_TRADE` with populated rejection reasons is a PASS.
`PAPER_EXECUTION_FAILED` or `EXECUTION_INCONSISTENT` is a FAIL — capture it.
Do not force a trade.

## 7. Restart recovery (invariant 11)

```bash
bash scripts/quantterm_restart.sh
sleep 60
bash scripts/quantterm_status.sh
PYTHONPATH=. python3 -m product.chain_status
```

Open positions, paper ledger and settled evidence must be identical to step 4.

## 8. External-storage loss fails closed (invariants 12, 13)

Safe equivalent — no forced unmount:

```bash
# Eject cleanly through the normal path
diskutil unmount /Volumes/QuantTermStorage
sleep 30
bash scripts/quantterm_status.sh     # expect BLOCKED/FAILED, never HEALTHY
ls /Volumes/QuantTermStorage         # must NOT have been recreated as an empty dir
```

Then reconnect and confirm identity is re-verified before children restart:

```bash
bash scripts/quantterm_storage_preflight.sh
bash scripts/quantterm_restart.sh
sleep 90
bash scripts/quantterm_status.sh
du -sh /Volumes/QuantTermStorage/QuantTerm/runtime   # same size as step 1
```

## 9. Broker mutation boundary, on the real host (invariant 14)

```bash
PYTHONPATH=. python3 -m pytest tests/test_broker_mutation_boundary_exhaustive.py -q
```

Read-only. Places no order. Proves every mutation route refuses, the raw SDK
handle is unreachable, and the interlock holds even with the unsafe legacy
override enabled.

## 10. UI agrees with backend (invariant 15)

Open http://127.0.0.1:5173 and confirm against the JSON captured above:

- the PIPELINE card's first stop matches `/api/chain-status`
- Paper Trading state matches `/api/paper-autopilot`
- `Scan ran` and `Market session` show **different** dates
- no LIVE / BUY / SELL / UNLOCK control appears anywhere

---

## What to send back

The output of steps 0, 3, 4, 6, 8 and 9. Those six establish deployed SHA,
single ownership, chain truth, paper outcome, storage fail-closed behaviour and
the broker boundary — the invariants that cannot be proven from a Linux runner.


---

## 11. Verifiers (run after the desk is up)

```bash
python3 scripts/verify_quantterm_stack.py      # read-only; no order creation
python3 scripts/verify_quantterm_actions.py    # real safe actions; no money
python3 scripts/run_product_acceptance.py      # exercises RUN_CYCLE_NOW
```

`verify_quantterm_actions.py` was confirmed to fail closed on a cloud runner
with `NO_SYMBOL_EVALUATED`, because the market-operations worker had no real
NSE data. On your Mac it has data, so it should proceed. If it still reports
`NO_SYMBOL_EVALUATED`, that is a genuine data problem — not a verifier bug —
and the previous scan artifact will have been preserved rather than replaced.

## Synthetic replay fixtures — do not use on this host

`scripts/seed_replay_store.py` writes synthetic bars. It requires
`QT_ALLOW_SYNTHETIC_REPLAY_FIXTURE=1` and refuses any runtime that is non-empty
without a fixture marker. **Never point it at
`/Volumes/QuantTermStorage/QuantTerm/runtime`.** Your Mac has real market data
and needs none of it.

## What could not be proven from a Linux runner

Every invariant in steps 1-3 and 7-8 is macOS/launchd/APFS specific and has no
cloud equivalent. The following were proven on Linux and should still be
re-confirmed here because the host differs: single ownership, child health,
scan truth, paper outcome, and the broker boundary.
