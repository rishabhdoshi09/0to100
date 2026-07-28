"""
🧒 Simple Language — the ONE source of plain-English content + presentation logic.

This module is PURE (no streamlit, no network, no clock, no order paths). It is the
single place that turns QuantTerm's internal reality into words an ordinary person —
the clarity benchmark is an intelligent 12-year-old — can act on. The Streamlit layer
(`ui/simple_mode.py`) and the user manual (`docs/user-guide/`) both read from here, so
the UI and the docs can never drift apart, and every statement is unit-testable.

It is PRESENTATION ONLY. It changes no trading behaviour, no configuration, no
permission, and it can place no order. Simple Mode hides implementation detail, never
risk: a green "Ready" is refused whenever required data is missing, stale or unavailable.
"""
from __future__ import annotations

# ── the two presentation depths (Simple is the new-user default) ───────────────
SIMPLE = "simple"
ADVANCED = "advanced"
DEFAULT_MODE = SIMPLE


def is_simple(mode: str | None) -> bool:
    """Simple unless Advanced is explicitly chosen (a missing/unknown value = Simple,
    so a brand-new user always starts Simple)."""
    return (mode or DEFAULT_MODE) != ADVANCED


# ══════════════════════════════════════════════════════════════════════════════
# Canonical terminology dictionary (UI plain labels ↔ exact technical names)
# ══════════════════════════════════════════════════════════════════════════════

GLOSSARY: dict[str, str] = {
    "Candidate": "Possible setup",
    "Eligibility": "Does it qualify?",
    "Rejection reason": "Why it was skipped",
    "Evidence Lab": "Research Lab",
    "Experiment": "Structured historical test",
    "Dataset snapshot": "Exact data version",
    "Point-in-time safe": "Uses only information available at that time",
    "Provenance": "Record of where the result came from",
    "Config hash": "Fingerprint of the rules",
    "Expectancy": "Average result per trade",
    "Drawdown": "Fall from the previous account high",
    "Slippage": "Difference between expected and actual price",
    "Circuit breaker": "Daily safety stop",
    "Trades-per-day limit": "Daily trade limit",
    "Migration lock": "Temporary live-trading lock",
    "Broker reconciliation": "Checking that system and broker records match",
    "PASS": "Promising under the registered test, not guaranteed",
    "FAIL": "Did not meet the required test",
    "INCONCLUSIVE": "Not enough trustworthy evidence",
    "DATA_UNAVAILABLE": "Required historical data is not installed",
    "No fill": "The assumed trade could not realistically happen",
    "Structural stop": "Exit level based on the setup's price structure",
    "Autopilot armed": "Automatic paper decisions are permitted",
    "Autopilot disarmed": "Automatic decisions are paused",
    "Pivot": "The price the stock must break above",
    "Base": "A long, quiet price range before a breakout",
    "Sector strength": "Whether the stock's group is doing well too",
    "Regime": "What kind of market we are in right now",
}


def define(term: str) -> str:
    """Plain meaning of a technical term (for tooltips / glossary). Empty if unknown —
    an unknown term is never guessed at."""
    return GLOSSARY.get(term, "")


# ══════════════════════════════════════════════════════════════════════════════
# The three operating modes — explained identically everywhere
# ══════════════════════════════════════════════════════════════════════════════

MODES: dict[str, dict] = {
    "RESEARCH": {
        "label": "Research",
        "meaning": "The system studies historical information to check whether an "
                   "idea really worked in the past.",
        "money": "No orders at all — real or pretend. It only studies.",
        "can_order": False,
    },
    "PAPER": {
        "label": "Paper practice",
        "meaning": "The system practises with imaginary money. No real money is ever "
                   "sent to the broker.",
        "money": "Pretend money only. Nothing reaches a real trading account.",
        "can_order": False,   # no REAL order; paper 'orders' are simulated only
    },
    "LIVE": {
        "label": "Live (real money)",
        "meaning": "The system could use real money — but only after every safety, "
                   "evidence, deployment and reconciliation requirement is satisfied.",
        "money": "Real money. Currently held behind a temporary live-trading lock.",
        "can_order": False,   # locked during the overhaul
    },
}


def mode_meaning(mode: str) -> dict:
    key = (mode or "").upper()
    return MODES.get(key, {"label": mode or "Unknown", "meaning": "",
                           "money": "", "can_order": False})


# ══════════════════════════════════════════════════════════════════════════════
# Plain status labels (colour-independent: each carries a word + a shape hint)
# ══════════════════════════════════════════════════════════════════════════════

# tone ∈ good | warn | bad | info  — the UI adds colour, but the WORD stands alone,
# so status is never communicated by colour only.
STATUS: dict[str, dict] = {
    "READY":                {"label": "Ready", "tone": "good", "icon": "●",
                             "plain": "The system is set up and watching the market."},
    "NEEDS_ATTENTION":      {"label": "Needs attention", "tone": "warn", "icon": "▲",
                             "plain": "Something needs you to look at it before you go on."},
    "WAITING_FOR_MARKET":   {"label": "Waiting for market", "tone": "info", "icon": "◷",
                             "plain": "The market is closed. Nothing to do until it opens."},
    "DATA_MISSING":         {"label": "Data missing", "tone": "bad", "icon": "▢",
                             "plain": "Required data is not installed. Results cannot be trusted yet."},
    "PAPER_PRACTICE_ACTIVE":{"label": "Paper practice active", "tone": "good", "icon": "●",
                             "plain": "Practising with pretend money. No real money involved."},
    "LIVE_LOCKED":          {"label": "Live trading locked", "tone": "info", "icon": "▣",
                             "plain": "Real-money trading is switched off on purpose."},
    "DAILY_SAFETY_STOP":    {"label": "Daily safety stop active", "tone": "warn", "icon": "▲",
                             "plain": "Today's loss limit was reached. No new trades today."},
    "NO_VALID_OPPORTUNITY": {"label": "No valid opportunity", "tone": "info", "icon": "◌",
                             "plain": "Nothing worth trading right now. That is a normal, good outcome."},
    "RESEARCH_NOT_PROVEN":  {"label": "Research not proven", "tone": "info", "icon": "◌",
                             "plain": "No idea has passed the honest historical test yet."},
    "STALE_DATA":           {"label": "Data is old", "tone": "warn", "icon": "▲",
                             "plain": "The last data is not fresh. Treat prices with care."},
}


def status(name: str) -> dict:
    return STATUS.get(name, {"label": name, "tone": "info", "icon": "•", "plain": ""})


# ══════════════════════════════════════════════════════════════════════════════
# The five questions every important thing must answer
# ══════════════════════════════════════════════════════════════════════════════

def five_questions(what: str, why: str, do: str, will_happen: str, next_step: str) -> dict:
    """The canonical shape for any page/card/warning/action."""
    return {"what_is_this": what, "why_it_matters": why, "what_should_i_do": do,
            "what_will_happen": will_happen, "what_next": next_step}


# ══════════════════════════════════════════════════════════════════════════════
# Decision explanations — Decision · Main reason · Supporting · Risk · Next step
# ══════════════════════════════════════════════════════════════════════════════

def explain_decision(decision: str, main_reason: str, supporting=None,
                     risk: str = "", next_step: str = "") -> dict:
    return {"decision": decision, "main_reason": main_reason,
            "supporting_reasons": list(supporting or [])[:3],
            "risk": risk, "next_step": next_step}


# maps the system's internal reason codes (detector rejection codes, autopilot funnel
# reasons, research verdicts) → a plain 5-part explanation. Advanced Mode still shows
# the exact code + numbers; Simple Mode shows this.
DECISION_EXPLANATIONS: dict[str, dict] = {
    "OVEREXTENDED_CHASE": explain_decision(
        "Skipped", "The stock was already too far above its normal trend.",
        ["Buying this high leaves little room before the exit.",
         "Chasing a fast move usually means a worse price."],
        "It could keep rising, so it can feel like a missed chance.",
        "Wait for a calmer setup — there will be others."),
    "STRUCTURAL_RISK_TOO_HIGH": explain_decision(
        "Skipped", "The possible loss was larger than the allowed limit.",
        ["The safe exit level was too far below the entry.",
         "A bigger risk per trade breaks the account's safety rule."],
        "A safer version of this setup may appear later.",
        "Nothing to do. The rule protected the account."),
    "UNCONFIRMED_BREAKOUT": explain_decision(
        "Waiting", "The breakout has not closed above the required price yet.",
        ["A price can poke above and fall back — only the close counts."],
        "It may never confirm, or it may confirm tomorrow.",
        "Wait for the candle to close above the level."),
    "WEAK_PRIOR_RS": explain_decision(
        "Skipped", "The stock was not a leader before this — it lagged the market.",
        ["Breakouts work better from stocks already doing well."],
        "It might still go up, but the odds are weaker.",
        "No action needed."),
    "WEAK_SECTOR": explain_decision(
        "Skipped", "The stock's sector (its group) was not strong.",
        ["Leaders usually rise together with a strong group."],
        "A lone strong stock in a weak group is riskier.",
        "No action needed."),
    "NO_BASE_CONTRACTION": explain_decision(
        "Skipped", "The quiet build-up before the breakout was not tight enough.",
        ["A good base gets calmer and tighter before it breaks out."],
        "The move may be less reliable.",
        "No action needed."),
    "DAILY_TRADE_LIMIT": explain_decision(
        "No trade", "The daily trade limit has already been reached.",
        ["A limit on trades per day stops over-trading."],
        "A great setup after the limit is still skipped — on purpose.",
        "Come back tomorrow. Review today's trades if you like."),
    "DAILY_SAFETY_STOP": explain_decision(
        "Paper trade blocked", "Today's safety-loss limit has activated.",
        ["After a set daily loss, the system stops for the day to protect capital."],
        "You might feel like 'making it back' — that is exactly the risk it blocks.",
        "Stop for today. The limit resets on the next trading day."),
    "NO_VALID_SETUP": explain_decision(
        "No trade", "Nothing today met all the rules.",
        ["The system only acts when every check passes."],
        "Doing nothing feels unproductive but is often the correct move.",
        "No action needed. 'No trade' can be the right answer."),
    "INCONCLUSIVE": explain_decision(
        "Research result unclear", "There was not enough trustworthy data to decide.",
        ["The honest answer is 'we don't know yet', not a guess."],
        "Acting on an unproven idea risks real money on luck.",
        "Add more/better data, then run the test again."),
    "DATA_UNAVAILABLE": explain_decision(
        "Historical data missing", "The data needed for this test is not installed.",
        ["Without the data, no honest pass/fail is possible."],
        "A verdict now would be made up, not measured.",
        "Ask the operator to install the historical data (see Data Health)."),
    "NO_FILL": explain_decision(
        "No trade taken", "The assumed trade could not realistically have happened.",
        ["The price gapped past the level, so a real order would not have filled there."],
        "Pretending it filled would flatter the results.",
        "No action needed — this keeps the practice honest."),
}


def decision_for(code: str) -> dict:
    """Plain 5-part explanation for a reason code. Unknown codes fall back to a safe,
    honest generic (never a fabricated specific reason)."""
    if code in DECISION_EXPLANATIONS:
        return DECISION_EXPLANATIONS[code]
    return explain_decision(
        "Skipped", "This setup did not pass one of the system's rules.",
        [], "There is always some risk in any trade.",
        "See Advanced Mode for the exact technical reason.")


# ══════════════════════════════════════════════════════════════════════════════
# Research verdicts in plain words
# ══════════════════════════════════════════════════════════════════════════════

VERDICTS: dict[str, dict] = {
    "PASS": {"plain": "Promising under the registered test — NOT a promise of profit.",
             "next": "It still must earn real evidence before any real money."},
    "FAIL": {"plain": "The idea did not meet the required test.",
             "next": "That is useful — it saves you from a losing idea."},
    "INCONCLUSIVE": {"plain": "Not enough trustworthy evidence to say pass or fail.",
                     "next": "Get more/better data, then test again."},
    "DATA_UNAVAILABLE": {"plain": "The required historical data is not installed, so the "
                                  "test cannot be judged.",
                         "next": "Install or locate the data (see Data Health)."},
}


def verdict_meaning(verdict: str) -> dict:
    return VERDICTS.get((verdict or "").upper(),
                        {"plain": "Unknown result.", "next": "See the Research Lab."})


# ══════════════════════════════════════════════════════════════════════════════
# Home page status — the whole 'where am I?' picture from a simple state dict
# ══════════════════════════════════════════════════════════════════════════════

def home_status(state: dict) -> dict:
    """Turn a plain state dict into the Home answers. PURE. `state` keys (all optional,
    safe defaults):
      mode: 'RESEARCH'|'PAPER'|'LIVE'   data_ok: bool   data_stale: bool
      market_open: bool   autopilot_armed: bool   safety_stop: bool
      trades_allowed: int   trades_used: int   open_positions: int
      attention: list[str]   live_enabled: bool
    RULE: never report READY when data is missing or stale — Simple Mode hides detail,
    never risk."""
    mode = (state.get("mode") or "PAPER").upper()
    data_ok = bool(state.get("data_ok", False))
    data_stale = bool(state.get("data_stale", False))
    market_open = bool(state.get("market_open", False))
    armed = bool(state.get("autopilot_armed", False))
    safety_stop = bool(state.get("safety_stop", False))
    allowed = int(state.get("trades_allowed", 0))
    used = int(state.get("trades_used", 0))
    open_pos = int(state.get("open_positions", 0))
    attention = list(state.get("attention") or [])

    # headline status — data problems and safety come FIRST (never masked by a green)
    if not data_ok:
        head = "DATA_MISSING"
    elif data_stale:
        head = "STALE_DATA"
    elif safety_stop:
        head = "DAILY_SAFETY_STOP"
    elif attention:
        head = "NEEDS_ATTENTION"
    elif mode == "PAPER" and armed:
        head = "PAPER_PRACTICE_ACTIVE"
    elif not market_open:
        head = "WAITING_FOR_MARKET"
    else:
        head = "READY"

    trading_allowed = (data_ok and not safety_stop and market_open
                       and used < allowed and mode == "PAPER")

    answers = {
        "is_ready": status(head),
        "data": ("Data is healthy." if data_ok and not data_stale
                 else "Data is old — treat prices with care." if data_stale
                 else "Required data is missing."),
        "market": "Market is OPEN." if market_open else "Market is closed.",
        "mode": mode_meaning(mode),
        "autopilot": ("Autopilot is ON (automatic paper decisions permitted)." if armed
                      else "Autopilot is OFF (no automatic decisions)."),
        "trading_allowed": ("Trading is allowed right now." if trading_allowed
                            else "Trading is not happening right now."),
        "safety_stop": ("Daily safety stop is ACTIVE — no new trades today." if safety_stop
                        else "Daily safety stop is not active."),
        "trades": f"{used} of {allowed} paper trades used today.",
        "open_positions": (f"{open_pos} open paper position(s)." if open_pos
                           else "No open positions."),
        "attention": attention or ["Nothing is waiting for you."],
        "live": ("Live real-money trading is LOCKED (on purpose)."),
        "next_action": next_best_action(state),
    }
    return {"headline": head, "headline_status": status(head), "answers": answers}


# ══════════════════════════════════════════════════════════════════════════════
# Next Best Action — exactly one useful next step for the current state
# ══════════════════════════════════════════════════════════════════════════════

def next_best_action(state: dict) -> str:
    if not state.get("data_ok", False):
        return "Install the historical market data (open Data Health for the exact steps)."
    if state.get("data_stale", False):
        return "Refresh the market data before relying on any price."
    if state.get("safety_stop", False):
        return "Daily safety stop is active — review today's trades. No new trades today."
    if state.get("attention"):
        return "Handle the item waiting for your attention above."
    if int(state.get("trades_used", 0)) >= int(state.get("trades_allowed", 0)) \
            and int(state.get("trades_allowed", 0)) > 0:
        return "Daily trade limit reached — you're done for today. Come back tomorrow."
    if not state.get("market_open", False):
        return "Market is closed. A good time to do the PAPER practice walkthrough."
    if (state.get("mode") or "PAPER").upper() == "PAPER" and not state.get("autopilot_armed", False):
        return "Start the PAPER practice walkthrough, or arm paper autopilot when ready."
    return "No action required. Let the system watch and record."


# ══════════════════════════════════════════════════════════════════════════════
# Behaviour matrix — one machine-readable source the UI + tests + docs all share
# ══════════════════════════════════════════════════════════════════════════════

BEHAVIOUR_MATRIX: list[dict] = [
    {"name": "research_data_available", "mode": "RESEARCH", "data": "available",
     "market": "any", "autopilot": "n/a", "safety": "n/a",
     "available_actions": ["run a historical test", "read past results"],
     "blocked_actions": ["place any order"],
     "explanation": "Research only studies history. It can never place an order.",
     "next_action": "Open the Research Lab and read a result."},
    {"name": "research_data_unavailable", "mode": "RESEARCH", "data": "unavailable",
     "market": "any", "autopilot": "n/a", "safety": "n/a",
     "available_actions": ["read the honest 'data unavailable' explanation"],
     "blocked_actions": ["judge the test", "place any order"],
     "explanation": "Without the historical data the test cannot honestly be judged.",
     "next_action": "Install the data (see Data Health)."},
    {"name": "paper_market_open", "mode": "PAPER", "data": "available",
     "market": "open", "autopilot": "armed", "safety": "clear",
     "available_actions": ["take a paper trade", "arm/disarm autopilot"],
     "blocked_actions": ["place a REAL order"],
     "explanation": "Paper practice uses pretend money. No real money moves.",
     "next_action": "Watch, or read why a setup qualified."},
    {"name": "paper_market_closed", "mode": "PAPER", "data": "available",
     "market": "closed", "autopilot": "any", "safety": "clear",
     "available_actions": ["review past paper trades", "do the walkthrough"],
     "blocked_actions": ["enter a new trade now", "place a REAL order"],
     "explanation": "The market is closed, so no new entries happen.",
     "next_action": "Do the PAPER walkthrough or review results."},
    {"name": "paper_safety_stop", "mode": "PAPER", "data": "available",
     "market": "open", "autopilot": "any", "safety": "stopped",
     "available_actions": ["review today's trades"],
     "blocked_actions": ["new paper trades today", "place a REAL order"],
     "explanation": "Today's loss limit was reached; new trades are blocked to protect capital.",
     "next_action": "Stop for today. It resets next trading day."},
    {"name": "paper_trade_limit", "mode": "PAPER", "data": "available",
     "market": "open", "autopilot": "any", "safety": "clear",
     "available_actions": ["review today's trades"],
     "blocked_actions": ["more trades today", "place a REAL order"],
     "explanation": "The daily trade limit stops over-trading.",
     "next_action": "Come back tomorrow."},
    {"name": "live_migration_lock", "mode": "LIVE", "data": "any",
     "market": "any", "autopilot": "any", "safety": "any",
     "available_actions": ["read why live is locked"],
     "blocked_actions": ["arm live", "place a REAL order"],
     "explanation": "Live real-money trading is held behind a temporary lock. An "
                    "environment variable alone does NOT make a strategy safe or eligible.",
     "next_action": "Keep practising in PAPER. Live needs formal evidence + deployment sign-off."},
    {"name": "telegram_paper_action", "mode": "PAPER", "data": "available",
     "market": "open", "autopilot": "any", "safety": "clear",
     "available_actions": ["accept a paper trade from Telegram"],
     "blocked_actions": ["place a REAL order from Telegram — ever"],
     "explanation": "Telegram can only ever trigger PAPER practice, never a real order.",
     "next_action": "Tap it to record a paper trade, or ignore it."},
    {"name": "stale_market_data", "mode": "PAPER", "data": "stale",
     "market": "any", "autopilot": "any", "safety": "any",
     "available_actions": ["refresh data"],
     "blocked_actions": ["trust the shown prices as live"],
     "explanation": "Old data can be wrong. It is labelled as stale, never shown as fresh.",
     "next_action": "Refresh before relying on any price."},
    {"name": "broker_mismatch", "mode": "LIVE", "data": "any",
     "market": "any", "autopilot": "any", "safety": "any",
     "available_actions": ["read the mismatch warning"],
     "blocked_actions": ["proceed as if records agree"],
     "explanation": "If the system's records and the broker's records disagree, it stops "
                    "and warns rather than guessing.",
     "next_action": "Resolve the mismatch before anything live (operator task)."},
    {"name": "no_valid_setup", "mode": "PAPER", "data": "available",
     "market": "open", "autopilot": "armed", "safety": "clear",
     "available_actions": ["wait"],
     "blocked_actions": ["force a trade"],
     "explanation": "Nothing met all the rules. 'No trade' is a correct outcome.",
     "next_action": "No action needed."},
    {"name": "eligible_candidate", "mode": "PAPER", "data": "available",
     "market": "open", "autopilot": "any", "safety": "clear",
     "available_actions": ["read why it qualified", "take the paper trade"],
     "blocked_actions": ["place a REAL order"],
     "explanation": "A possible setup passed every rule. It is still only a possibility.",
     "next_action": "Read the entry, stop and maximum loss before anything."},
    {"name": "rejected_candidate", "mode": "PAPER", "data": "available",
     "market": "open", "autopilot": "any", "safety": "clear",
     "available_actions": ["read why it was skipped"],
     "blocked_actions": ["override the skip"],
     "explanation": "A possible setup failed a rule. The reason is always recorded.",
     "next_action": "Read the reason to learn the rules."},
]


def matrix_row(name: str) -> dict:
    for row in BEHAVIOUR_MATRIX:
        if row["name"] == name:
            return row
    return {}


# any state that can affect real money — used by tests to prove nothing in Simple Mode
# ever exposes a real-order action.
def any_row_allows_real_order() -> bool:
    for row in BEHAVIOUR_MATRIX:
        for a in row["available_actions"]:
            if "REAL order" in a or "real order" in a or "live order" in a.lower():
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# What NOT to do — practical consequence, no moralising
# ══════════════════════════════════════════════════════════════════════════════

WHAT_NOT_TO_DO: list[dict] = [
    {"dont": "Treat every alert as 'buy now'.",
     "because": "Alerts are things to look at, not commands. Many are skipped after a closer look."},
    {"dont": "Assume a high score means guaranteed profit.",
     "because": "A score ranks setups; it does not predict the future. High-scoring setups still fail."},
    {"dont": "Switch to LIVE because a synthetic or practice test passed.",
     "because": "Practice profit is not real evidence. Live needs formal, measured proof."},
    {"dont": "Change lots of settings after a bad run.",
     "because": "Tweaking to fit past losses usually makes the next result worse, not better."},
    {"dont": "Bypass the daily safety stop.",
     "because": "The stop exists to end bad days early. Trading through it is how small losses become big ones."},
    {"dont": "Keep re-arming after a risk lock.",
     "because": "The lock is telling you today is done. Forcing more trades fights your own safety net."},
    {"dont": "Treat paper profit as proof you'll profit for real.",
     "because": "Real trading adds slippage, gaps and emotion that paper does not."},
    {"dont": "Assume missing data means zero.",
     "because": "Missing is unknown, not zero. Zero would quietly corrupt every result."},
    {"dont": "Buy just because a price moved or 'looks cheap/expensive'.",
     "because": "Price and valuation alone are not the system's setup. That's guessing."},
    {"dont": "Rely on Telegram as a live-order channel.",
     "because": "Telegram is paper-only, by design. It can never send a real order."},
    {"dont": "Expect a trade every day.",
     "because": "Most days have no valid setup. Forcing trades is the fastest way to lose."},
    {"dont": "Use money you cannot afford to lose.",
     "because": "Even a good process has losing streaks. Only risk what won't hurt your life."},
]


# ══════════════════════════════════════════════════════════════════════════════
# Success = following the process (not winning every trade)
# ══════════════════════════════════════════════════════════════════════════════

SUCCESS_CHECKLIST: list[str] = [
    "Confirm the system says data is healthy.",
    "Confirm which mode you are in (Research / Paper / Live).",
    "Begin in PAPER mode.",
    "Read WHY a setup qualified or was skipped.",
    "Check the entry, the stop, and the maximum possible loss.",
    "Do not override safety limits.",
    "Let the system record every outcome.",
    "Review results over a meaningful number of trades, not one.",
    "Accept 'no trade' when nothing valid exists.",
    "Move towards LIVE only after formal evidence and deployment sign-off exist.",
]

GOOD_DAY = ("A good day is not necessarily a profitable day. A good day is one where "
            "the system followed its rules and protected the account.")

DAILY_CHECKLIST: list[dict] = [
    {"key": "data_healthy", "label": "Data is healthy"},
    {"key": "mode_confirmed", "label": "Correct mode selected"},
    {"key": "limits_reviewed", "label": "Safety limits reviewed"},
    {"key": "autopilot_confirmed", "label": "Autopilot state confirmed"},
    {"key": "positions_reviewed", "label": "Open positions reviewed"},
    {"key": "warnings_checked", "label": "Unresolved warnings checked"},
]


# ══════════════════════════════════════════════════════════════════════════════
# Learning cards — short lessons triggered by real system states
# ══════════════════════════════════════════════════════════════════════════════

LEARNING_CARDS: dict[str, dict] = {
    "no_trade_is_good": {"title": "Why 'no trade' can be a good day",
        "body": "The system only acts when every rule passes. On most days nothing "
                "qualifies. Sitting out a bad setup protects your money for the good one."},
    "paper_vs_live": {"title": "Why paper results differ from live",
        "body": "Real orders face slippage, gaps and delays that pretend trades skip. "
                "So live results are usually a bit worse than paper."},
    "gap_stop": {"title": "Why a gap can give a worse exit",
        "body": "If a stock opens far below your stop, the real exit fills at that worse "
                "open — not neatly at the stop. The system assumes the worse price."},
    "high_score_can_fail": {"title": "Why a high-scoring setup can still fail",
        "body": "A score ranks how well a setup fits the pattern. It cannot see the future. "
                "Even the best-looking setups lose sometimes."},
    "missing_data_blocks_verdict": {"title": "Why missing data blocks a verdict",
        "body": "Without trustworthy history, a pass/fail would be a guess. The honest "
                "answer is INCONCLUSIVE until the data exists."},
}


# ══════════════════════════════════════════════════════════════════════════════
# First-run onboarding — at most seven short steps
# ══════════════════════════════════════════════════════════════════════════════

ONBOARDING_STEPS: list[dict] = [
    {"title": "What QuantTerm is",
     "body": "A careful assistant that studies the stock market, practises trades with "
             "pretend money, and only ever suggests — it does not gamble your money."},
    {"title": "What QuantTerm cannot promise",
     "body": "It cannot promise profit. The market is uncertain. Even a good process has "
             "losing days. Past results do not guarantee the future."},
    {"title": "Research, Paper and Live",
     "body": "Research = study history. Paper = practise with pretend money. Live = real "
             "money, and it is locked until strict safety and evidence rules are met."},
    {"title": "Data-health check",
     "body": "Before anything, the system checks its data is present and fresh. If data "
             "is missing, it says so honestly instead of showing fake results."},
    {"title": "Safety limits",
     "body": "There is a daily loss stop, a limit on trades per day, and a fixed risk per "
             "trade. These protect the account and cannot be quietly bypassed."},
    {"title": "Your first PAPER walkthrough",
     "body": "A short, safe demo with made-up data shows a setup qualifying, one being "
             "skipped, a win, a loss, and the safety stop — no real money involved."},
    {"title": "Where to get help",
     "body": "Every page has a 'What is this?' panel, and the manual under docs/user-guide "
             "explains everything in plain language. You can reopen this tour any time."},
]


# ══════════════════════════════════════════════════════════════════════════════
# 'When I do this, what happens?' — the user-action behaviour guide
# ══════════════════════════════════════════════════════════════════════════════

def _action(action, where, requires, checks, changes, unchanged, money, undo, verify):
    return {"action": action, "where": where, "requires": requires, "checks": checks,
            "changes": changes, "unchanged": unchanged, "real_money": money,
            "undo": undo, "verify": verify}


USER_ACTION_GUIDE: list[dict] = [
    _action("Open the app", "Anywhere", "Nothing",
            "Data health, market status, current mode", "Shows you the Home status",
            "No settings change", False, "Just close it", "Read the Home status line."),
    _action("Refresh data", "Home / Data Health", "Nothing",
            "Whether fresh data can be loaded", "Updates the prices shown",
            "No trade is placed", False, "Nothing to undo",
            "The data-health line should say 'healthy'."),
    _action("Select PAPER mode", "Settings / mode selector", "Nothing",
            "That paper mode is available", "Marks decisions as pretend-money",
            "No real money is ever used", False, "Switch mode again",
            "The mode indicator reads 'Paper practice'."),
    _action("Arm PAPER autopilot", "Autopilot page", "Paper mode + healthy data",
            "Gates, limits, safety stop", "Allows automatic PAPER decisions",
            "No real order is possible", False, "Disarm autopilot",
            "Status shows 'Autopilot ON' and it's still PAPER."),
    _action("Disarm autopilot", "Autopilot page", "Nothing",
            "Nothing risky", "Pauses automatic decisions", "Open positions stay as they are",
            False, "Arm again", "Status shows 'Autopilot OFF'."),
    _action("Accept a Telegram paper action", "Telegram", "Paper mode",
            "It is a paper action (always)", "Records a PRETEND trade",
            "No real order — ever, from Telegram", False, "Close the paper position",
            "It appears as a paper position, not a live order."),
    _action("Dismiss an alert", "Alerts", "Nothing", "Nothing risky",
            "Hides that alert", "No trade happens", False, "Alerts can reappear",
            "The alert list updates."),
    _action("Open a candidate explanation", "Opportunities", "Nothing", "Nothing risky",
            "Shows why it qualified/was skipped", "No trade happens", False, "Just close it",
            "You can read the main reason in plain words."),
    _action("Open a paper position", "My Positions", "An open paper position", "Nothing risky",
            "Shows entry, stop, max loss", "Nothing changes", False, "Just close the view",
            "You can see the maximum planned loss."),
    _action("Close a paper position", "My Positions", "An open paper position",
            "Current paper price", "Records a pretend result", "No real money moves",
            False, "It's recorded; you can't un-close history", "It moves to your results."),
    _action("Change a risk setting", "Settings", "Confirmation",
            "New value is within safe bounds", "Changes future PAPER sizing/limits",
            "Does NOT enable live trading", False, "Change it back",
            "The confirmation showed the exact before/after."),
    _action("View a rejected setup", "Opportunities", "Nothing", "Nothing risky",
            "Shows the skip reason", "No trade happens", False, "Just close it",
            "You understand which rule it failed."),
    _action("Open the Research Lab", "Research Lab", "Nothing", "Nothing risky",
            "Shows past historical tests", "No order — research is isolated", False,
            "Just leave the page", "You see PASS/FAIL/INCONCLUSIVE, not orders."),
    _action("Read PASS/FAIL/INCONCLUSIVE", "Research Lab", "Nothing", "Nothing risky",
            "Nothing", "Nothing", False, "n/a",
            "PASS ≠ guaranteed; INCONCLUSIVE = not enough data."),
    _action("Encounter DATA_UNAVAILABLE", "Research Lab / Data Health", "Nothing",
            "That the data is missing", "Shows an honest explanation", "No fake verdict",
            False, "n/a", "Follow the operator step to install data."),
    _action("Encounter stale data", "Home / any price", "Nothing", "Data freshness",
            "Labels prices as old", "No trade on bad data", False, "Refresh data",
            "Prices are marked stale until refreshed."),
    _action("Hit the daily safety stop", "Home / Autopilot", "A day's losses reaching the limit",
            "Day P&L vs the limit", "Blocks new trades today", "Open positions are managed normally",
            False, "It resets next trading day", "Status shows 'Daily safety stop active'."),
    _action("Hit the trades-per-day limit", "Home / Autopilot", "Using all allowed trades",
            "Trades used vs allowed", "Blocks more trades today", "Nothing else changes",
            False, "It resets next trading day", "Status shows trades used = allowed."),
    _action("Attempt to access LIVE mode", "Settings / Autopilot", "All safety + evidence met",
            "The live migration lock", "Nothing — live stays locked", "Paper is unaffected",
            False, "n/a", "You see the 'Live trading locked' message and why."),
    _action("View broker-reconciliation status", "Safety and Limits (Advanced)", "Nothing",
            "Whether system and broker records match", "Nothing", "Nothing", False, "n/a",
            "It shows matched/mismatch — a mismatch blocks anything live."),
]


# ══════════════════════════════════════════════════════════════════════════════
# Data-unavailable panel + safety confirmations
# ══════════════════════════════════════════════════════════════════════════════

def data_unavailable_panel(operator_step: str = "", advanced_detail: str = "") -> dict:
    """The honest empty-state (no stack trace, no blank page)."""
    return {
        "what_happened": "The historical market data needed for this research test is not installed.",
        "what_it_means": "The system cannot honestly decide whether this idea passed or failed.",
        "current_status": "INCONCLUSIVE — DATA UNAVAILABLE",
        "what_still_works": ["Moving around the app", "Paper features where live prices exist",
                             "Previous research records", "The documentation",
                             "Reviewing your settings"],
        "what_to_do_next": operator_step or ("Ask the operator to install / locate / "
                                             "validate the historical dataset."),
        "advanced_detail": advanced_detail,   # exact paths / snapshot diagnostics (Advanced only)
    }


def safety_confirmation(setting: str, current, proposed, effect: str, affects: str,
                        reverse: str, max_consequence: str | None = None) -> dict:
    """Specific confirmation for a safety-sensitive change — never a generic 'Are you
    sure?'. `affects` ∈ 'PAPER' | 'LIVE' | 'both'."""
    msg = (f"You are changing {setting} from {current} to {proposed}. {effect} "
           f"It affects {affects}.")
    if affects != "LIVE":
        msg += " It does NOT enable LIVE trading."
    return {"setting": setting, "current": current, "proposed": proposed,
            "effect": effect, "affects": affects, "max_consequence": max_consequence,
            "reverse": reverse, "message": msg}


# ══════════════════════════════════════════════════════════════════════════════
# Contextual page help — the five questions per major page
# ══════════════════════════════════════════════════════════════════════════════

PAGE_HELP: dict[str, dict] = {
    "home": five_questions(
        "Your starting screen — the whole picture at a glance.",
        "It tells you if the system is ready and safe before you do anything.",
        "Read the status line and the one recommended next step.",
        "Nothing changes just by looking. It's a dashboard.",
        "Do the single 'Next best action' it suggests."),
    "opportunities": five_questions(
        "Possible setups the system found (and ones it skipped).",
        "It shows what qualified and, importantly, WHY things were skipped.",
        "Open one and read its plain-language reason.",
        "Opening an explanation changes nothing — no trade happens.",
        "Learn the rules from the skip reasons before any paper trade."),
    "positions": five_questions(
        "Your open PAPER positions and their planned exits.",
        "It shows your entry, your stop, and the most you could lose.",
        "Check the maximum planned loss on each.",
        "Viewing changes nothing; closing records a pretend result.",
        "Only close when your plan says to."),
    "research_lab": five_questions(
        "Where the system studies history to test ideas honestly.",
        "It decides PASS / FAIL / INCONCLUSIVE — and never places an order.",
        "Open a result and read its plain meaning.",
        "Nothing here can trade. It only studies the past.",
        "Remember: PASS is promising, not a promise."),
    "safety": five_questions(
        "Your safety limits and locks (daily stop, trade limit, live lock).",
        "These protect the account and cannot be quietly bypassed.",
        "Review the limits so you know what will stop you and when.",
        "Viewing changes nothing; changing asks for a specific confirmation.",
        "Keep the defaults until you truly understand each one."),
    "data_health": five_questions(
        "Whether the system's data is present and fresh.",
        "Bad or missing data makes every result untrustworthy.",
        "Confirm it says 'healthy'. If not, follow the fix steps.",
        "Refreshing updates prices; it places no trade.",
        "Never rely on a green 'Ready' when data is missing or stale."),
    "help": five_questions(
        "Plain-language help and the full manual.",
        "You should never be stuck wondering what a screen means.",
        "Search for a word, or open the guide for this page.",
        "Reading help changes nothing.",
        "Return to Home and do your next action."),
}


def page_help(page: str) -> dict:
    return PAGE_HELP.get(page, five_questions(
        "A QuantTerm page.", "It is part of your workflow.",
        "Read the labels and the next-step hint.",
        "Most views change nothing by themselves.",
        "Return Home if unsure."))


# ══════════════════════════════════════════════════════════════════════════════
# Beginner PAPER walkthrough — a safe story on FICTIONAL data (no broker/network)
# ══════════════════════════════════════════════════════════════════════════════
# Every step is made-up ("Practice Ltd") for teaching only. Nothing here calls the
# broker, Telegram, or any live service — it is a script, not a live trade.

WALKTHROUGH_FICTIONAL = True   # asserted by tests: the walkthrough never touches live

WALKTHROUGH_STEPS: list[dict] = [
    {"title": "A possible setup appears",
     "body": "Pretend stock 'PRACTICE LTD' has been quiet for weeks, then jumps above "
             "its resistance price on heavy volume. The system flags it as a possible setup."},
    {"title": "Why it qualified",
     "body": "It was already a leader, its group was strong, the quiet base was tight, and "
             "the breakout closed above the required price with a small, sensible risk."},
    {"title": "Why another one was skipped",
     "body": "Pretend stock 'CHASE LTD' also jumped — but it was already far above its "
             "normal trend. Buying that high is chasing, so the system skipped it."},
    {"title": "Entry is known only AFTER the signal",
     "body": "The system waits for the breakout candle to close, then plans to enter on "
             "the NEXT day. It never pretends to buy at a price it couldn't have known."},
    {"title": "The structural stop",
     "body": "The exit-if-wrong level sits just below the setup's structure. If price falls "
             "there, the practice trade closes to limit the loss."},
    {"title": "Your maximum possible loss",
     "body": "Before anything, you can see the worst case: (entry − stop) × size. You always "
             "know the most you could lose on a practice trade before it starts."},
    {"title": "A no-fill day",
     "body": "Sometimes the stock gaps far past the entry, so a real order couldn't have "
             "happened there. The system records 'no fill' instead of a pretend perfect entry."},
    {"title": "A winning outcome",
     "body": "'PRACTICE LTD' keeps rising and later closes below its trailing line. The "
             "practice trade exits with a pretend gain. It is recorded honestly."},
    {"title": "A losing outcome",
     "body": "Another practice trade drops to its stop and exits for a pretend loss. Losses "
             "are normal — the rules keep each one small."},
    {"title": "The daily safety stop",
     "body": "After a few pretend losses hit the day's limit, the system stops trading for "
             "the day. No 'making it back'. It resets next trading day."},
    {"title": "No valid opportunity",
     "body": "On many days nothing qualifies. The system does nothing — and that is a good "
             "day, because it protected the account."},
    {"title": "Research: PASS, FAIL, INCONCLUSIVE",
     "body": "Separately, the Research Lab tests ideas on history. PASS = promising (not "
             "guaranteed). FAIL = didn't meet the test. INCONCLUSIVE = not enough data."},
]


WHAT_QUANTTERM_CANNOT_PROMISE = [
    "It cannot promise profit — the market is uncertain.",
    "A PASS in research is not a promise of future money.",
    "Paper (practice) profit is not proof of real profit.",
    "It will not produce a trade every day — most days there is nothing valid.",
    "It cannot protect you from risking money you can't afford to lose.",
]
