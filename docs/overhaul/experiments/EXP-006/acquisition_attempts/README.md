# EXP-006 — data acquisition attempts (append-only)

Auditable record of each attempt to materialise the real NSE dataset required by the
EXP-006 readiness gate. Append-only; nothing overwritten. No synthetic or display-only
data is ever substituted for the missing dataset.

| Attempt | Host data-capable? | Source connectivity | Ingestion | Readiness | Experiment run? |
|---|---|---|---|---|---|
| `acq-0001` | **No** | NSE archives DENIED (HTTP 000; proxy `403 Forbidden`) | `build_store` → 0 sessions / 0 files | **RED — DATA_UNAVAILABLE** | **No** (blocked) |

## acq-0001 — precise acquisition blocker

This environment is **not a data-capable host**: the outbound proxy denies NSE archive
hosts (`nsearchives.nseindia.com` / `www.nseindia.com`) with
`Tunnel connection failed: 403 Forbidden` — the host is not in the proxy allowlist
(which permits only package registries + Anthropic). The canonical ingestion
(`data.bhavcopy_store.build_store` + `data.index_store`) therefore retrieves **0
sessions, 0 files**; `is_ready()` is False; there is no benchmark and no reproducible
snapshot. The EXP-006 readiness gate is RED, so no economic PASS/FAIL was issued and the
frozen runner was NOT executed. Machine-readable detail: `acq-0001.json`.

**Unblock (on a data-capable host with outbound NSE access):**
1. `git pull` `overhaul/evidence-lab`.
2. `python -c "from data.bhavcopy_store import build_store; build_store()"` and
   `python -c "from data.index_store import build_index_store; build_index_store()"`.
3. Supply `logs/ca_events.json` (corporate actions) and `logs/universe_history.json`
   (listing/delisting) for RESEARCH_GRADE (see `docs/overhaul/data_acquisition/`).
4. Confirm the readiness gate is green, then run the UNCHANGED frozen runner:
   `python -m research.momentum_breakout.runner --out logs/experiments/EXP-006`,
   copy the artifacts into a NEW `runs/<next-id>/` (do not touch `0001-blocked`).
