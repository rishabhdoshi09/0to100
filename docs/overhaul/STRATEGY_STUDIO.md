# Strategy Studio — architecture, lifecycle & approval

Autonomous strategy discovery + human-in-the-loop approval. **Research + review only.**
The system may generate and REJECT strategies autonomously; it must never silently
implement, paper-deploy, promote or live-deploy one. No module can place a broker order.

## Modules (all pure, tested; `research/strategy_studio/`)
- `spec.py` — canonical versioned `StrategySpec` (config hash from RESULT-determining
  fields only) + the lifecycle state machine (user-only approval transition).
- `grammar.py` — the ONLY approved, point-in-time-safe building blocks discovery may use
  (fundamentals only where genuinely PIT).
- `discovery.py` — constrained generation (seeded, family-diverse, budgeted), append-only
  attempt registry, validity stages, overfitting controls (multiple-testing burden,
  untouched-test isolation, simpler-baseline, complexity), and the data-readiness gate.
- `review.py` — five SEPARATE confidences, the "Convince Me" case (defence + prosecution,
  evidence/plausible/speculation labels, careful recommendation language), comparison.
- `tweak.py` — guided controls + natural-language→explicit-diff (no code execution) +
  impact preview + material-change versioning.
- `approval.py` — user-only, immutable PAPER-only approval record + separate paper
  activation. No live path.
- `wizard.py` — guided manual creation (same evidence standards).
- `ui/strategy_studio_page.py` — thin page (Discover · Review · Tweak · Compare · Approve).

## Lifecycle
`GENERATED → INVALID | REJECTED | UNDER_REVIEW → PROMISING → AWAITING_USER_APPROVAL
→ APPROVED_FOR_PAPER → PAPER_EVALUATION → (ELIGIBLE_FOR_LIVE_REVIEW) | RETIRED | DECAYED`.
Only a strategy that clears the research gate reaches `AWAITING_USER_APPROVAL`. Only a
**USER** may move `AWAITING_USER_APPROVAL → APPROVED_FOR_PAPER` — research code raising a
`LifecycleError` if it tries. LIVE eligibility is **out of scope**.

## The five stages the milestone distinguishes
1. **Autonomous idea generation** — the system proposes ideas (no evidence yet).
2. **Historical evidence** — real data + the existing harness decide PASS/FAIL/INCONCLUSIVE
   (synthetic fixtures are NEVER evidence).
3. **User approval** — a human, not the system, approves a frozen version for PAPER.
4. **Paper activation** — a separate explicit confirmation; PAPER only; Telegram stays
   paper-only.
5. **Live eligibility** — not in this milestone; the live migration lock stays on.

## Safety guarantees (enforced + tested)
No research strategy places an order; no generated strategy enters PAPER without explicit
user approval; no tweak inherits old evidence (new version, new config hash, retest); no
active strategy changes itself; no promotion by highest backtest return; missing data is
never turned favourable; synthetic results are labelled non-evidence; every attempt count
is disclosed; costs and portfolio-risk controls cannot be bypassed; no LIVE deployment;
the emergency disarm is independent of strategy logic.

## Honest limitation
Real strategy EVIDENCE requires a research-grade dataset, which is not present in this
environment (see `data_acquisition/`). Until then the Studio runs on labelled demonstration
fixtures for software behaviour only — it claims no market evidence.
