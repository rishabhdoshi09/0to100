# Operator capability inventory (Phase 13)

Source of truth: `product/runtime_capabilities.py`.

Modes:

- **AUTOMATIC** — normal day, no click
- **HOME_ACTION** — occasional safe Home button
- **ADVANCED_ACTION** — kept off the default Home card
- **DEVELOPER_ONLY** — never in normal UI or automatic runtime

Normal product usage requires a terminal only for the single launcher:

`bash scripts/run_quantterm_complete.sh` (or `./quantterm.sh`).

| Capability | Old invocation | New mode | Automatic trigger | Home action | Canonical owner | Persisted state | Failure behavior | Still requires terminal? |
|---|---|---|---|---|---|---|---|---|
| Start QuantTerm | `bash scripts/run_quantterm_complete.sh` | AUTOMATIC | launcher + supervise | — | complete → inner stack | runtime json | restart dead child, one owner | yes (launcher only) |
| Check settings | edit `.env` | AUTOMATIC | launcher start | — | complete script | `.env` | stop with instruction | no |
| Zerodha login | `python main.py login` | HOME_ACTION | auth probe | Login to Zerodha | kite + autonomy auth | access token | WAITING FOR ZERODHA LOGIN | no |
| Data freshness | data page | AUTOMATIC | desk + DATA_REFRESH | — | desk pipeline | dashboard.data | Preparing / Retry | no |
| Get today's prices | `REFRESH_DATA_NOW` | AUTOMATIC | desk first step | Retry | market_ops DATA_PREPARE | ops store | dedupe | no |
| Whole-market scan | `RUN_SCAN_NOW` / local_stack scan | AUTOMATIC | kick + scan slots | Scan now | market_ops → scan service | latest_momentum_scan.json | no duplicate scan | no |
| Research list | scan write | AUTOMATIC | after scan | — | recommendations_store | latest_recommendations.json | empty is valid | no |
| Paper decision | `RUN_CYCLE_NOW` | AUTOMATIC | PAPER_CYCLE | Decide now | paper_autopilot | journal + intel_book | no-trade is valid | no |
| Watch open paper | intelligence cycle | AUTOMATIC | intraday/EOD | — | PaperBook.mark | intel_book.json | missing bar skips | no |
| Finish the day | manual settle | AUTOMATIC | resolve_outcomes | — | settle_and_report | ledger + daily report | pending stays pending | no |
| Grade skipped names | settle_pending_counterfactuals | AUTOMATIC | later official bars | — | settle_pending_from_market | forward_evidence.jsonl | not P&L | no |
| Remember what happened | ingest_closed_book | AUTOMATIC | EOD then next cycle | — | paper_learning_loop | policies + ingested | idempotent; hard gates stay hard | no |
| Write decisions down | implicit journal | AUTOMATIC | record_cycle_evidence | — | forward_evidence | forward_evidence.jsonl | no PIT overwrite | no |
| Daily one-page summary | none | AUTOMATIC | settle_and_report | — | write_daily_report | forward_daily/ | missing stays missing | no |
| Check evidence trail | `python scripts/verify_forward_soak.py` | AUTOMATIC | startup + EOD + throttled cycle | Verify (advanced) | verify_persisted_soak | forward_soak_verify.json | valid no-trade ≠ fail | no |
| Health lanes | system-health-contract | AUTOMATIC | dashboard poll | — | system_health_contract | computed | no fake green | no |
| Broker observation | observation scripts | AUTOMATIC | zerodha_observer if logged in | — | reconciliation | logs/reconciliation | waiting, never mutate | no |
| Scan now | Home button | HOME_ACTION | — | RUN_SCAN_NOW | OperationStore | ops store | dedupe | no |
| Retry market data | control POST | HOME_ACTION | — | REFRESH_DATA_NOW | OperationStore | ops store | dedupe | no |
| Refresh company facts | control POST | HOME_ACTION | desk after scan | REFRESH_LONG_TERM_NOW | market_ops | long-term overlay | last-good labelled | no |
| Refresh daily note | control POST | HOME_ACTION | desk kick | REFRESH_NEWS_NOW | market_ops | news/reports | empty stays empty | no |
| Decide now | control POST | HOME_ACTION | — | RUN_CYCLE_NOW | autonomy controls | controls.db | stale reco cannot enter | no |
| Pause / resume paper | control POST | HOME_ACTION | — | PAUSE/RESUME | autonomy controls | owner_state | cannot unlock live | no |
| F&O / pulse force | control POST | ADVANCED_ACTION | — | advanced | market_ops | stores | optional | no |
| pytest / DoD / fixtures | engineering CLIs | DEVELOPER_ONLY | — | none | tests/scripts | tmp/docs | never Home | yes |

Live money is not a capability. It stays fail-closed.
