# Independent adversarial code review: QuantTerm (NSE India paper-trading research terminal)

You are an independent, critical reviewer, not an approver. Another engineer (Claude) wrote most of this code, and your job is to find what is wrong.

Do NOT assume that claims in commit messages, docstrings, comments or `docs/` are true. Verify each one against the code.

## Target

- The working directory is a detached checkout of `01dec27c7890a8d4d7906c1588aa9f3647bdb171`.
- Review the change set `9636b03055961e5f53b5dfb7cfe5cfede03ba13f..01dec27c7890a8d4d7906c1588aa9f3647bdb171`.
- The bundle is read-only: `<review-bundle>/commits.txt`, `<review-bundle>/stat.txt` and `<review-bundle>/diff.patch`.
- You may run read-only commands (`git log/show/diff/blame/grep`, `rg`, `sed -n`, `nl`, `cat`) on files inside this checkout or the bundle.
- The sandbox is read-only, so pytest probably cannot run. Write out the exact pytest test you would run instead; the coordinator runs it.

## Hard rules

- **Off limits:** `.env`, `~/.codex`, `~/.claude`, `/Volumes`, the production checkout, and any runtime or log directory. Never print a credential.
- **No side effects:** no writes, network calls, broker calls or process control.
- **Evidence:** every CONFIRMED finding needs a concrete trace through real code. Do not soften findings, and do not invent them.

## Invariants the code claims to enforce

- No fake data, and stale data must look stale.
- Paper only: live broker trading is hard-locked, and Telegram taps are paper.
- One rulebook (`core/evaluation_spec.py`): exactly 10 official sessions, gap-aware fills and exits. Every settler and learner must use it, and outcome windows are checked by `core/session_window.py`.
- **BUY admission.** A BUY needs a walk-forward with ≥ 30 trades and a dependence-aware lower bound > 0 over ≥ 20 disjoint windows. It must also pass:
  - Holm at α/52;
  - the global 52-look annual budget (`research/alpha_ledger.py`, production configuration only);
  - provenance checks (schema, rulebook, horizon, exact windows, data ≤ 21 days);
  - re-verification in `product/scan_store.py::load_scan` against the active snapshot, by id.
- **Risk limits:** 1% per trade, 10% per name, 5% total open risk, 5 positions, through `risk.portfolio_risk.account_exposure_gate` under `account_mutation_lock`.
- **Exploration lane:** at most 1 trade a day at ≤ 0.5% risk, always labelled EXPLORATION and never a qualified BUY.
- **Learning:** count independent units, never copies. Simulated rows never count as forward evidence.

## Scope

Look for any change in `9636b03055961e5f53b5dfb7cfe5cfede03ba13f..01dec27c7890a8d4d7906c1588aa9f3647bdb171` that:

- breaks those invariants;
- leaks future information;
- double-counts or drops evidence;
- lets an unproven setup be shown, notified, learned from or traded as a qualified BUY;
- bypasses a risk control;
- makes a test pass without testing the claim it names.

Trace the changed functions' callers.

## Output (Markdown)

1. `## Findings`, ordered P0 → P2. Each finding gives:
   - an ID, a severity and a status (CONFIRMED with a traced counterexample, or HYPOTHESIS);
   - its `path:line` and function;
   - the counterexample;
   - a minimal pytest sketch that fails today;
   - a brief fix.

   Severity levels:
   - **P0:** an unproven BUY, a risk bypass, corrupted evidence or learning, or future leakage.
   - **P1:** wrong statistics, learning or labels.
   - **P2:** robustness.
2. `## Verified correct`: short, with `file:line` evidence.
3. `## Not examined / limits`.

A few real, traced defects beat many vague concerns.

## This run: verify the FIXES for your own earlier findings (one pass, no loop)

This change set is the developer's response to an earlier GPT review of the base commit. The findings and the claimed resolutions are in `docs/audits/2026-11-20-gpt-crossreview.md` (in this checkout). Treat each resolution as a claim.

For every finding marked **Fixed**:
- check that the fix closes the counterexample;
- check that it does not open a new path: a regression in a caller, a weakened gate, or a test that passes without testing the claim.

For **Open** items, say only whether the stated reason is accurate.

For the two disagreements (C-1 common shock, C-5 intersection), give your view in at most three sentences each.

Pay particular attention to these changed functions:

| Function | Concern |
|---|---|
| `core/outcome_resolver.py::resolve_trade` | exact-session prefix, overlay drop |
| `core/session_window.py` | |
| `core/evaluation_spec.py::simulate_trade` | |
| `product/scan_store.py::bind_evidence` | id binding, outage note |
| `product/paper_autopilot.py` | `_current_scan_certificate`, `_withdraw_uncertified`, the exploration reservation under the lock |
| `alerts/telegram_actions.py::_do_paper_trade` | |
| `product/forward_soak.py::attach_closed_trades` | |
| `scan/signal_backtest.py` | `gate_program`, `apply_family_correction(untested=)`, the save threshold, `_alpha_ledger_lost` |
| `research/overlap_inference.py` | greedy block count |
| `research/alpha_ledger.py` | `override_active` |
| `product/historical_replay.py` | `evaluate_outcomes`, `NEUTRAL_REPLAY_CALIBRATION` |
| `data/corporate_actions.py::raw_price_multiplier` | its use in `scan/unified_scanner.py::_analyze` |
