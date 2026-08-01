# QuantTerm through a pro trader's eyes — enhancement ideas & feasibility

Written from the seat of a desk trader at a large firm, but for retail users. The question a pro asks
at every idea is not "is this a buy?" — it is **"what is my risk, in rupees and in R, and what does
this do to my whole book?"** Retail apps almost never answer that at the moment of decision. QuantTerm
already has the institutional machinery (position sizer, portfolio-risk, correlation clusters,
concentration, TCA, regime, evidence). The gap is *synthesis at the point of action*.

All ideas below respect the non-negotiables: read-only projection over authoritative backend state,
missing data stays missing, no fabricated metrics, PAPER-only, LIVE locked, no duplicate risk logic.

| # | Idea (pro instinct) | What it answers | Reuses | Feasibility |
|---|---------------------|-----------------|--------|-------------|
| 1 | **Risk-first Trade Plan (the "R lens")** | "How many shares for 1% risk? What's my ₹ risk and reward:risk? What does adding this do to my open-risk %? Is it a *new* bet or piling onto one I already have?" | `risk/position_sizer.size_position`, `risk/portfolio_risk.portfolio_risk_report`, `risk/correlation.clusters_from_corr` | **HIGH value / LOW risk — DOING THIS.** Pure projection composing existing functions; fully testable. |
| 2 | Portfolio "real bets" view | "You think you have 6 positions; correlation says you have 3 bets." | `risk/correlation.book_correlation_report` | Medium — `book_correlation_report` exists; needs a retail projection + surface. Folded partially into #1. |
| 3 | Regime-aware risk throttle | "Breadth is narrow / RISK_OFF → size at 0.5% not 1%, and here's why." | `product/market_view`, `core/regime_engine` | Med — compose regime into #1's `suggested_risk_pct`. **Folded into #1.** |
| 4 | Cost-drag honesty per idea | "Round-trip costs eat X% of your expected R — this edge may not survive costs." | `execution/cost_model`, `execution/tca` | Medium — a later projection; needs an expected-move input to be honest. |
| 5 | Expectancy scoreboard (REAL closed trades only) | "Your live expectancy is +0.12R over 41 trades — not a backtest." | `product/paper_status`, EV/verdict | Medium — must gate on sample size; partly in `paper_status`. |
| 6 | Behavioral guardrails at point of action | "4th trade today; 3rd banking long; adding risk after 2 losses." | `core/decision_journal`, `reports/trade_coach` | Medium — needs a session-behaviour projection. |
| 7 | Liquidity/impact reality | "At your size you're 0.2% of ADV — fillable" vs "you'd move it." | volume in scan data | Low-med — a turnover check. |
| 8 | Invalidation clarity ("what makes me wrong") | The stop as a *thesis-invalidation* level, not just a number. | candidate stop | Trivial — **folded into #1.** |

## Chosen implementation: #1 — the Risk-first Trade Plan (with #3 and #8 folded in)

**Why this one.** It is the single highest-leverage decision-support tool and the clearest thing that
separates a disciplined professional from a gambler: *risk before reward, book before position*. It is
also the safest to build — a **read-only projection that composes authoritative functions** and adds
**no new risk math** (single source of truth preserved), degrades honestly when inputs are missing,
and is fully deterministic/network-free for testing.

**What it computes for any candidate `(symbol, entry, stop, target)` at a given capital:**
- exact `qty`, `invested`, `rupee_risk`, `% of capital`, `risk % of capital` — from `size_position`;
- `reward:risk` (R to target) — or `None` when target is missing (never fabricated);
- `invalidation` — the stop as the thesis-kill level, in ₹ and %;
- portfolio **open-risk % before vs after** adding this, and the book **verdict** (OK/CAUTION/DANGER)
  — from `portfolio_risk_report(extra_trade=…)`, the authoritative account-risk function;
- **correlation read**: does this join an *existing* cluster (piling onto a bet) or is it a *new*
  independent bet — from `clusters_from_corr`; `unknown` when history is unavailable;
- a **regime-throttled `suggested_risk_pct`** (e.g. 1% → 0.5% in a weak tape) via an injected factor;
- a plain-language, **risk-first** summary a retail user can act on.

**Non-goals / honesty rails.** No fabricated size when the stop is invalid (`tradeable=False` with a
reason). No correlation claim without data. No live orders. No new store. It only *reads and
composes*; the OMS, Risk Governor, PaperBook and sizer remain the sole owners of truth.

**Surfaces.** A pure projection `product/trade_plan.py` + deterministic tests, then a thin retail
surface (Streamlit expander per candidate) and a function shape the React terminal API can expose next.
