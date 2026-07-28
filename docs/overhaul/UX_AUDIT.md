# UX Audit — Simple Mode milestone (repository-first)

**Scope:** inspect the real Streamlit app before changing anything. Code is the source
of truth. This records what exists; it invents no functionality. Simple Mode is a
presentation layer on top — no business logic, execution rule or risk control changed.

## Navigation as it exists today

- **Sidebar** (`app.py`): logo → search (NSE + US, hits yfinance/live) → US index picker
  → market selector (Auto/India/US) → **primary nav** `option_menu`
  **[Pulse · Markets · Autopilot · JARVIS]** (Markets fans to Stocks/Options/US) →
  **"More Tools"** expander (Dashboard, My Holdings, Terminal, Live Watch, My Watchlist,
  Journal, Smart Alerts, IPO Calendar, Research, Tools) → **Diagnostics** → Plain-English
  toggle (`ui/plain_english.py`) → status strip → Copilot.
- **Routing:** `sidebar_nav` session key drives a long `if _page == …` dispatch. New-user
  landing was effectively "Daily Pulse".
- **~65 `ui/*.py` pages** exist. The beginner-relevant ones: `command_center` (Today),
  `autopilot_page` (paper autopilot controls), `scanner` (setups), `research_dashboard`
  /`stock_research` (research), `system_pulse`/`error_guard` (health), `alerts_page`,
  `street_pulse_page` (reports), `positions_panel`, `journal`.

## Per-page findings (beginner lens)

| Page | Current purpose | Intended user | Primary action | Technical terms shown | Confusing / dead-ends | Keep in Advanced |
|---|---|---|---|---|---|---|
| Pulse / Daily Pulse | Market + Brain posture | trader | read | regime, breadth, posture, VIX | jargon-dense; Hinglish mixed | regime internals, scores |
| Autopilot | Arm/disarm paper autopilot, gates | trader | arm/disarm | gates, circuit breaker, R, pool | many metrics; safety wording technical | gate funnel, R math |
| Markets/Scanner | Setups + verdicts | trader | scan | conviction, ATR, RSI, CLV | verdict without plain "why" | raw features, scores |
| JARVIS | LLM chat | power user | ask | model, tokens | needs API key; opaque when off | prompts, model id |
| Research (dashboard) | Experiment/evidence views | researcher | inspect | experiment, provenance, config hash, DSR | very technical; no plain verdict | all stats |
| Diagnostics | Config self-check | operator | check | IST clock, build, tokens | technical labels | full check list |
| Plain English toggle | Jargon translation | beginner | toggle | — | partial (regimes/VIX only), not a full mode | — |
| Search | Symbol lookup + setup | any | search | verdict, conviction, E/S/T | fetches live (network) | — |

## Major usability problems found

1. **No single "where am I / is it safe?" screen.** The user had to assemble mode, data
   health, market status, autopilot state and limits from several places.
2. **Decisions lacked a plain "why".** Setups showed a verdict/score but not a
   beginner-readable reason for skip/accept/wait.
3. **Jargon-first.** point-in-time, provenance, config hash, expectancy, drawdown,
   circuit breaker, migration interlock appeared with no plain equivalent.
4. **No onboarding / no safe walkthrough.** A newcomer had no guided first run.
5. **Empty/technical states.** DATA_UNAVAILABLE and stale data could read as blank or
   raw rather than an honest explanation with a next step.
6. **Colour-dependent status.** Some states leaned on colour without a standalone word.
7. **Generic confirmations.** Safety-sensitive changes risked a generic "are you sure?".
8. **Partial plain-English mode.** `ui/plain_english.py` translated a few regime/VIX
   terms only — not a full Simple/Advanced experience.

## Safety-sensitive actions inventory (must stay enforced beneath the UI)

- Arm/disarm autopilot (`execution/autopilot.arm/disarm`) — LIVE arming is behind the
  `QT_LIVE_ENABLED` **temporary migration lock** and the exact phrase.
- Change risk/limit settings (`autopilot.set_config`, clamped).
- Telegram paper action (`alerts/telegram_actions._do_paper_trade`, hard `paper=True`).
- Daily-loss circuit breaker + trades-per-day limit + per-trade risk (autopilot gates).
- EXP-006 research run (`research/momentum_breakout/runner`) — execution-isolated.

## Decision: presentation-only Simple Mode

- Build on the existing plain-English toggle, not a parallel jargon system.
- Put ALL plain content + logic in one **pure, tested** module `core/simple_language.py`
  (glossary, mode meanings, status labels, decision explanations, behaviour matrix,
  onboarding, walkthrough, safety confirmations, page help). Docs + UI + tests read it,
  so they cannot drift.
- Thin Streamlit layer `ui/simple_mode.py` renders it and reads status **read-only**
  (`autopilot.get_status()`), importing no order path.
- Default new users to a Getting-Started page; add Simple Home / Practice Walkthrough /
  User Guide routes. Advanced Mode is opt-in and changes presentation depth only.

## What stays in Advanced Mode

Raw features, component scores, hashes, dataset ids, experiment specs, evidence
statistics (DSR/PSR/alpha/block-CI), technical logs, broker reconciliation, detailed
risk math, gate funnels, developer diagnostics — all unchanged and still reachable.
