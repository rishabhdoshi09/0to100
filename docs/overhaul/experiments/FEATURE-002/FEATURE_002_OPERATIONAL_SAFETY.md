# FEATURE-002 — Operational safety

## Fail open

`observe_production_scan` is wrapped in `try/except` in `auto_scan._scan_once_locked`. Inner compute is also fail-open. A logging exception cannot:

- prevent `_results` from being stored
- prevent `_save_state`
- prevent autopilot (hook runs **after** `on_setups`)
- raise into the scan worker

Default observe path uses a **daemon thread** after a deepcopy of the production cards so RS table construction cannot add scan latency.

## Isolation

- `scan/unified_scanner.py` does not import FEATURE-002
- `execution/trade_executor.py` does not import FEATURE-002
- `execution/autopilot.py` does not import FEATURE-002
- Toggle `set_enabled(False)` skips all writes; cards stay identical

## What may differ when logging is on

- Files under `logs/feature002/`
- debug log line `feature002_shadow_skip` / thread name `feature002-shadow`

## What must not differ

BUY/WATCH, side, qty, stop, target, GTT, autopilot queue, Telegram payload, Ready inputs, production sort order.
