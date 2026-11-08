# QuantTerm read-only host evidence probe

This check does **not** release QuantTerm. It produces repeatable latency evidence
on the actual Mac while preserving the existing live-order lock and trading state.

Run it **only after** the intended PR stack has been integrated and the target
commit SHA is checked out on the owner’s Mac. First verify the host's external
runtime volume and paper recovery status using the existing Mac runbook.
Do not run during a state recovery incident.

From the checked-out repository:

```bash
cd ~/0to100
git rev-parse HEAD
./venv/bin/python scripts/quantterm_release_readonly_probe.py \
  --api http://127.0.0.1:8765 --samples 30 --timeout 20 --pause 1 \
  > "$HOME/Desktop/quantterm-host-readonly-probe.json"
```

The script sends **GET only** to the loopback API, with proxies and redirects
disabled. It verifies `live_lock_verified=true`,
`live_locked=true`, and `live_execution_authorized=false` before
measuring `/api/health` and `/api/dashboard`. A missing or failed lock
check aborts further probing. It never reads API credentials or contacts Kite.

It reports finite response times and interpolated p50/p95/p99 for each route,
along with bounded failure counts and a single macOS swap snapshot. The
number of samples is printed so reviewers do not confuse a small-sample
estimate with a production service-level metric.

### Evidence reviewers must request separately

- The exact Git commit SHA, branch, working-tree cleanliness, Python version,
  host architecture, and external runtime-volume identity.
- Cold-start and warm-start measurements. These must be distinguished.
- A representative sustained soak across real NSE sessions, including process
  memory, swap growth, file descriptors, CPU, and scan concurrency.
- An explicitly authorized external-volume detach/reattach drill with the
  host safety guard and unchanged restored paper book.
- A read-only recovery backup restore into a disposable destination; never
  alter the source backup.
- Verified Kite login, current-session snapshot and paper-only scan path.
- Real-browser dashboard and Telegram integration.
- Rollback demonstration and independent review.

The script exits `0` for successfully collected measurements, `1` if
one or more sampled API calls fail, and `2` if the preflight live-execution
lock cannot be verified. **Exit 0 is NOT release certification.**

Do not assume a healthy endpoint sample proves backup recoverability, storage
durability, risk accounting, source freshness, browser behavior, or soak
reliability. These remain separate acceptance gates.
