# QuantTerm Professional Product Plan

## Product thesis

QuantTerm should feel like a dedicated retail quant operating system: simple enough to use in seconds,
but backed by the same evidence, risk, data lineage and operational discipline expected from a serious
quant desk. Sophistication belongs in the engine; clarity belongs in the interface.

## Permanent product principles

1. **One source of truth.** The UI reads persisted scanner, research, portfolio and supervisor state. It
   never becomes a second trading engine.
2. **No decorative numbers.** Every displayed metric must be sourced, timestamped or explicitly marked
   unavailable.
3. **One decision surface.** Market posture, opportunities, evidence, risk and next action should be
   visible without navigating through a maze of pages.
4. **Retail language first.** Scientific terms remain available behind deeper views, but the primary
   interface explains what is happening, why it matters and what could invalidate it.
5. **Evidence before action.** Every opportunity has a reason, risk, entry discipline and invalidation.
6. **Safety is structural.** Streamlit queues controls; the autonomy supervisor owns scheduling and
   mutation. Live broker execution remains locked until separately proven and enabled.

## Information architecture

### Workspace

- **Command Center** — market regime, best setups, long-term intelligence, paper portfolio and system health.
- **Scanner** — Momentum, Conviction, Breakouts, Pre-Breakout, Long-Term, F&O and Avoid as modes of one workspace.
- **Portfolio** — exposure, open risk, exits and performance.
- **Market** — regime, breadth, sectors and institutional context.

### Research

- What We’ve Learned
- Backtest
- Reports
- Research Laboratory

### Operations

- Paper Trading
- Automation
- Alerts
- Data & Zerodha

### More

- Market News
- Settings
- Help

## Delivery sequence

### Phase 1 — Product shell and command surface

- Global professional dark design system.
- Compact navigation and QuantTerm brand rail.
- New default Command Center.
- Unified Scanner workspace.
- Existing engines and pages preserved.

### Phase 2 — Stock Intelligence

- Select any stock from scanner, alerts, portfolio or watchlist.
- Unified chart, setup, delivery, relative strength, sector, fundamentals, valuation, news and evidence trail.
- One plain-language thesis and one explicit invalidation.

### Phase 3 — Portfolio console

- Paper/live separation.
- Open-risk and sector/correlation map.
- Equity curve, expectancy, drawdown and decision attribution.
- Position detail and exit reasoning.

### Phase 4 — Research productization

- Experiment cards, strategy health and evidence tiers.
- Rejected ideas and negative evidence become first-class outputs.
- Beginner view and advanced scientific view share the same records.

### Phase 5 — Operations polish

- Single supervisor timeline.
- Data freshness, alerts, failure recovery and scheduled-job visibility.
- Mobile Telegram parity for the most important decisions.

## Phase 1 acceptance criteria

- Command Center is the default route.
- Momentum, conviction and long-term discovery are consolidated into Scanner modes.
- UI never performs a scan or mutation directly.
- Missing market/history/fundamental data is shown honestly.
- Current backend tests continue to pass.
- Product remains usable at laptop and tablet widths.
