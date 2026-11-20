# Independent adversarial code audit: QuantTerm (NSE India paper-trading research terminal)

You are an independent, critical reviewer, not an approver. Another engineer (Claude) wrote most of this code. Your job is to find what is wrong, so do NOT assume claims in commit messages, docstrings, comments or `docs/` are true. Treat every claim as a hypothesis and verify it against the code.

## Target
- Repository: `rishabhdoshi09/0to100`. The working directory is a git checkout at the exact commit `9636b03055961e5f53b5dfb7cfe5cfede03ba13f` (branch `claude/settlement-batch-20261118`).
- Production currently runs `125880a6`. The candidate change set is `125880a6..9636b030` (52 commits). The most recent cycle, not yet reviewed by anyone, is `dd4e9c34..9636b030` (8 commits).
- Review bundle (read-only): `<review-bundle>/commits_since_production.txt`, `<review-bundle>/stat_since_production.txt`, `<review-bundle>/diff_since_last_review.patch`, `<review-bundle>/diff_since_production.patch`.
- You may run read-only commands: `git log/show/diff/blame/grep`, `rg`, `grep`, `sed -n`, `nl`, `cat` on files INSIDE this checkout or BUNDLE. The sandbox is read-only, so pytest probably cannot run (tmp writes fail). If you need a deterministic check, write out the exact pytest test you would run; the coordinator will run it and report back.

## Hard rules
- Do not read `.env` or any file outside this checkout and BUNDLE. In particular, never read `~/.codex`, `~/.claude`, `/Volumes`, `/Users/apple/0to100` (the production checkout) or any runtime/log directory. Never print a token, key or credential.
- Do not attempt any write, network call, broker call or process control.
- Do not soften findings. Equally, do not invent them: every CONFIRMED finding must be backed by a concrete trace through real code.

## Context you need (product invariants the code claims to enforce)
- No fake data. Stale data must look stale. Live broker trading is hard-locked; this system is paper-only. Telegram taps are paper-only.
- One rulebook: `core/evaluation_spec.py`, a 10-official-session horizon with gap-aware fills and exits. Backtest, forward tracker, decision journal, paper book and counterfactuals must all use it.
- A BUY must be admitted only by evidence: a walk-forward with ≥30 trades, a dependence-aware lower bound > 0, Holm family-wise correction at α/52 per weekly report, a GLOBAL 52-look annual alpha budget (`research/alpha_ledger.py`), provenance (report schema, eval spec, horizon, family method, age ≤ 21 days), and re-verification at the scan-store boundary (`product/scan_store.py::load_scan`) against the active calibration snapshot (`scan/calibration_snapshot.py`).
- Risk: 1% risk per trade, 10% per name, 5% total open risk, max 5 positions; a paper EXPLORATION lane (1 trade/day at 0.5% risk) may probe gated setups but must never be labelled as a qualified BUY.
- Learning must count independent units (day, symbol, reason), never copies (`product/learning_observations.py`).

## Required output format (Markdown)
1. `## Findings`: one entry per finding, ordered P0 → P2. Each entry has:
   - **ID** (e.g. A-1), **Severity** P0/P1/P2:
     - P0 = can admit an unproven BUY, bypass a risk control, corrupt evidence or learning, or leak future information.
     - P1 = materially wrong statistics, learning or labelling, without a direct money path.
     - P2 = robustness or clarity.
   - **Status**: CONFIRMED (you traced a concrete counterexample through the code) or HYPOTHESIS (plausible, but not fully traced, or it depends on runtime state).
   - **Location**: `path:line` and the function name(s). Cite real line numbers from this checkout.
   - **Counterexample**: concrete inputs or state → the wrong output.
   - **Deterministic test**: a minimal pytest sketch that would fail today if the finding is real. Name existing fixtures or helpers only when you have verified they exist.
   - **Suggested fix** (brief).
2. `## Verified correct`: claims you checked and found to hold, with file:line evidence. Keep it short.
3. `## Not examined / limits`: what you did not cover.

Be concrete and economical. Quality over quantity: a few real, traced defects beat many vague concerns.

## Your scope (Run A): pipeline integrity, counterfactual horizon, learning influence

Use finding IDs A-1, A-2, …

1. **The end-to-end pipeline:** scan → rank → recommendation → paper entry → settlement → learning. Trace the real call chain in the canonical stack (not the legacy `scan/auto_scan.py` or `ui/`). Start points:
   - `scan/market_scan_service.py::run_whole_market_scan`, `scan/unified_scanner.py`, `product/scan_store.py`;
   - `product/decision_ranking.py`, `product/recommendations_workspace.py`;
   - `product/paper_autopilot.py`, `research/auto_research/paper_book.py`, `product/_paper_learning_loop_core.py`, `product/autonomous_loop.py`;
   - `research/autonomy/jobs.py` (including `HANDLERS` and any installed overrides such as `install_outcome_liveness` / `_run_outcome_resolution_light`), `research/autonomy/supervisor.py`;
   - settlement: `core/outcome_resolver.py`, `core/signal_outcome_tracker.py`, `product/forward_evidence.py`, `product/forward_soak.py`, `product/counterfactual_learning.py`;
   - learning: `product/learning_observations.py`, `product/learning_policy_repair.py`, `product/learning_scorecard.py`, `product/challenger_shadow.py`, `product/paper_learning*.py`.

   Look for broken links: a stage whose output no later stage reads, a job that reports SUCCEEDED but writes nothing, a handler that is replaced at install time so the edited function is dead code, double counting, or a crash or restart that duplicates or drops rows.

2. **The 5-session vs 10-session counterfactual-horizon discrepancy.** Previously, rejected (counterfactual) trades were judged by a 5-session close-to-close move that ignored the stop, while taken trades used the 10-session rulebook (`core/evaluation_spec.py`). The outcome job could also stamp a counterfactual label on a TAKEN trade, which blocked its real paper result. Commit `19a98b2f` claims to fix this, and `research/counterfactual_reconcile.py` plus `docs/audits/2026-11-20-counterfactual-horizon-migration.md` describe a migration. Verify independently that:
   - every path that judges a rejection now uses the same horizon, entry fill, stop/target and trail as taken trades;
   - no path still uses 5 sessions, or any other horizon constant (grep for 5, `horizon`, `HORIZON`, `days=`, `sessions`, `bars_held`);
   - taken trades can never be settled as counterfactuals;
   - legacy labels cannot reach a learning score;
   - the reconcile/migration tool cannot double-count, and is idempotent and reversible as claimed.

3. **Does feedback really steer future decisions?** Do rejected trades, missed winners, false signals and paper outcomes genuinely influence later decisions, and through which exact code path (file:line), with what gating? Or is learning recorded but never read by the decision path? Distinguish:
   - (a) data that changes a future verdict, rank, size or gate;
   - (b) data shown on a dashboard only;
   - (c) data written and never read.

   For (a), check it cannot inflate a setup into a BUY without passing the evidence gate. Check that copies or duplicates and simulated outcomes cannot count as forward evidence.
