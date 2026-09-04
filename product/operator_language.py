"""Simple-mode translations. Logic stays exact; only the explanation changes."""

from __future__ import annotations

from typing import Any, Mapping

# Machine codes → child-friendly sentences. Professional mode still shows the code.
SIMPLE_REASONS: dict[str, str] = {
    "ENTRY_TOO_EXTENDED": "Good company, bad price right now. Waiting.",
    "DD_GATE_FAILED": "Skipped — the company check did not pass.",
    "LOW_QUALITY_SETUP": "Not a strong enough setup today.",
    "NO_TRADE": "Nothing was good enough today.",
    "NOT_SURFACED": "Seen by the scan, but not a final pick.",
    "STALE_RECOMMENDATION": "Yesterday's list is too old to trade from.",
    "PAPER_TRADING_DISABLED": "The paper bot is paused.",
    "OUTSIDE_ENTRY_WINDOW": "The market window for new entries is closed.",
    "DUPLICATE_POSITION": "Already holding this name on paper.",
    "MAX_PORTFOLIO_RISK": "Skipped because the portfolio already has too much risk.",
    "SECTOR_CAP": "Skipped because the portfolio already has too much similar risk.",
    "CORRELATION_CAP": "Skipped because the portfolio already has too much similar risk.",
    "PORTFOLIO_BLOCK": "Skipped because the portfolio already has too much similar risk.",
    "INSUFFICIENT_CAPITAL": "Not enough paper cash left for a safe size.",
    "INVALID_STOP": "Skipped — the stop price was not usable.",
    "EVIDENCE_POLICY_BLOCK": "Skipped — recent evidence said this kind of setup is weak.",
    "CHASE_RISK": "Good company, bad price right now. Waiting.",
    "NO_ELIGIBLE_TRADE": "Nothing was good enough today.",
    "NO_CYCLE_RECORDED": "The paper bot has not decided yet today.",
    "INSUFFICIENT EVIDENCE": "Too early to judge.",
    "FORWARD_SAMPLE_TOO_SMALL": "Too early to judge.",
    "PIT evidence missing": "We don't have enough information from that exact time.",
    "UNANCHORED_CORRELATION": "We don't have enough information from that exact time.",
    "EXECUTION_EVIDENCE_INCOMPLETE": "We do not yet know how costs would have changed the result.",
    "LIVE_MONEY_LOCKED": "Real money stays locked.",
    "INVALID_SYMBOL": "That ticker is not a real stock we can judge.",
    "DATA_UNAVAILABLE": "We could not load market data for that name.",
    "ANALYSIS_ERROR": "The analysis did not finish, so this is not a buy or avoid call.",
    "NO_JUDGMENT": "This is not an investment judgment.",
    "WAITING FOR ZERODHA LOGIN": "Zerodha login is needed.",
    "KITE_ACCESS_TOKEN_MISSING": "Zerodha login is needed.",
    "BROKER_LOGIN_REQUIRED": "The stock decision is ready; broker execution waits for Zerodha login.",
    "BLOCKED_BROKER_AUTH": "The stock decision is ready; broker execution waits for Zerodha login.",
}

SIMPLE_TERMS: dict[str, str] = {
    "PIT": "information from that exact time",
    "OOS": "a later test that the system did not peek at",
    "R multiple": "how many times the planned risk was won or lost",
    "evidence policy": "a written rule from past results — it cannot invent a buy",
    "execution-adjusted expectancy": "average result after estimated trading costs",
    "provenance": "where the evidence came from",
    "correlation matrix": "how similarly names move together",
    "regime model": "a guess about the market's mood",
    "challenger": "an idea being tested in the shadows, not trusted yet",
    "REAL_FORWARD_N": "real market observations, not practice tests",
}

LANE_SIMPLE = {
    "Ready": "Ready",
    "Working": "Working",
    "Needs you": "Needs you",
    "Waiting": "Waiting",
    "Problem": "Problem",
}


def simple_reason(code: str | None, *, fallback: str = "") -> str:
    raw = str(code or "").strip()
    if not raw:
        return fallback
    if raw in SIMPLE_REASONS:
        return SIMPLE_REASONS[raw]
    upper = raw.upper()
    if upper in SIMPLE_REASONS:
        return SIMPLE_REASONS[upper]
    return fallback or raw.replace("_", " ").lower()


def simple_term(term: str) -> str:
    return SIMPLE_TERMS.get(term, term)


def explain_opportunity(row: Mapping[str, Any]) -> dict[str, str]:
    """Four questions for a Home card. Does not change ranking or decisions.

    When a persisted committee decision is present, it is the headline truth.
    Scanner/recommendation-only rows remain research setups and are never
    promoted to BUY merely because their technical verdict says BUY.
    """
    symbol = str(row.get("symbol") or "")
    setup = str(row.get("setup_label") or row.get("status") or "setup")
    chase = bool(row.get("chase_risk"))
    reason = str(row.get("reason_code") or row.get("reason") or "")
    decision = str(row.get("decision") or row.get("committee_decision") or "").upper()
    execution = str(row.get("execution_state") or "").upper()
    entry_state = str(row.get("entry_state") or "").upper()

    if decision == "BUY":
        if execution.startswith("BLOCKED_"):
            label = "BUY — execution blocked"
            meaning = simple_reason(execution, fallback="The investment decision is BUY; execution is waiting on an operational gate.")
        elif entry_state and entry_state not in {"ENTER_NOW", "READY", "ELIGIBLE"}:
            label = "BUY — waiting for entry"
            meaning = "The investment thesis passed, but the planned entry condition is not ready yet."
        else:
            label = "BUY"
            meaning = "The decision committee passed this name. Read the evidence and risk plan before any paper entry."
    elif decision == "WAIT":
        label = "WAIT"
        meaning = simple_reason(reason, fallback="The company/setup may be interesting, but QuantTerm is waiting for a better condition.")
    elif decision in {"AVOID", "REJECTED"}:
        label = "AVOID"
        meaning = simple_reason(reason, fallback="The decision committee rejected this opportunity under the current evidence and risk rules.")
    elif decision == "NO_JUDGMENT":
        label = "NO JUDGMENT"
        meaning = simple_reason(reason or "NO_JUDGMENT")
    elif chase or "EXTENDED" in reason.upper():
        label = "Waiting for better entry"
        meaning = simple_reason("ENTRY_TOO_EXTENDED")
    elif str(row.get("reco_tier") or "") in {"watch", "avoid"}:
        label = "Worth watching"
        meaning = "Interesting, but not ready to take."
    elif reason and reason.upper() not in {"ELIGIBLE", "ENTER_NOW", ""}:
        label = "Rejected — too risky / too extended / evidence weak"
        meaning = simple_reason(reason)
    else:
        label = "Research candidate"
        meaning = f"{symbol} is a current research name. The scanner did not make an investment BUY decision."

    action = (
        "Open the name to inspect committee evidence, entry and portfolio state."
        if decision
        else "Open the name to read the evidence. The decision committee decides BUY / WAIT / AVOID."
    )
    return {
        "what": (
            "A persisted committee decision."
            if decision else
            "A research name from the saved market scan — not a new scanner."
        ),
        "found": f"{symbol} · {setup}" if symbol else "No name yet.",
        "meaning": meaning,
        "action": action,
        "label": label,
        "why": simple_reason(reason) if reason else meaning,
        "technical": reason or str(row.get("setup_label") or ""),
    }
