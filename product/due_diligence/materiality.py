"""Deterministic event taxonomy and materiality. No LLM sentiment."""
from __future__ import annotations

import re
from typing import Any, Mapping

# User-facing taxonomy. Mapped from curator event_type + headline tokens.
TAXONOMY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Results", ("result", "earnings", "quarterly", "q1", "q2", "q3", "q4", "profit")),
    ("Guidance", ("guidance", "outlook", "forecast")),
    ("Order Win", ("order win", "wins order", "bagged", "secures order", "purchase order")),
    ("Order Cancellation", ("cancel", "terminated contract", "order loss")),
    ("Acquisition", ("acqui", "takeover", "stake buy")),
    ("Divestment", ("divest", "stake sale", "sells stake")),
    ("Merger", ("merger", "amalgamat", "scheme of")),
    ("Fundraising", ("qip", "preferential", "rights issue", "fund raise", "fpo")),
    ("Debt", ("bond", "ncd", "debenture", "borrow", "refinanc")),
    ("Credit Rating", ("rating", "outlook revised", "creditwatch", "downgrade", "upgrade")),
    ("Regulatory Action", ("sebi", "rbi", "usfda", "warning letter", "show cause", "penalty", "ban")),
    ("Management Change", ("ceo", "cfo", "managing director", "resigns", "appointed")),
    ("Auditor Change", ("auditor", "statutory audit")),
    ("Promoter Transaction", ("promoter", "pledge")),
    ("Insider Transaction", ("insider", "sast", "bulk deal", "block deal")),
    ("Capacity Expansion", ("capacity", "plant expansion", "greenfield")),
    ("Product Launch", ("launch", "introduces")),
    ("Plant Shutdown", ("shutdown", "halted production", "plant closed")),
    ("Litigation", ("litigation", "lawsuit", "court", "nclt")),
    ("Dividend", ("dividend",)),
    ("Buyback", ("buyback", "buy-back")),
    ("Bonus", ("bonus issue", "bonus share")),
    ("Split", ("stock split", "sub-division", "subdivision")),
)

_CRORE = re.compile(
    r"(?:₹|rs\.?|inr)?\s*([\d,.]+)\s*(?:crore|cr)\b",
    re.IGNORECASE,
)
_LEGACY_TO_TAXONOMY = {
    "order_or_contract": "Order Win",
    "merger_or_acquisition": "Acquisition",
    "regulatory": "Regulatory Action",
    "regulatory_action": "Regulatory Action",
    "promoter_or_insider": "Promoter Transaction",
    "rating": "Credit Rating",
    "fund_raising": "Fundraising",
    "results": "Results",
    "pledge": "Promoter Transaction",
    "governance": "Management Change",
}


def classify_taxonomy(headline: str, event_type: str = "") -> str:
    text = f"{headline} {event_type}".lower()
    for label, tokens in TAXONOMY:
        if any(tok in text for tok in tokens):
            return label
    mapped = _LEGACY_TO_TAXONOMY.get(str(event_type or "").strip())
    return mapped or "Others"


def extract_crore(text: str) -> float | None:
    match = _CRORE.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _bucket(ratio: float | None) -> str:
    if ratio is None:
        return "Unmeasured"
    if ratio >= 0.20:
        return "Very High"
    if ratio >= 0.05:
        return "High"
    if ratio >= 0.01:
        return "Moderate"
    return "Low"


def materiality(
    article: Mapping[str, Any],
    *,
    revenue_cr: float | None = None,
    market_cap_cr: float | None = None,
    pat_cr: float | None = None,
    debt_cr: float | None = None,
    promoter_pct: float | None = None,
) -> dict[str, Any]:
    headline = str(article.get("headline") or "")
    category = classify_taxonomy(headline, str(article.get("event_type") or ""))
    amount = extract_crore(headline + " " + str(article.get("summary") or ""))
    ratio = None
    basis = "No contract/penalty amount could be parsed from the headline."
    if amount is not None and category in {"Order Win", "Order Cancellation"} and revenue_cr:
        ratio = round(amount / abs(revenue_cr), 4)
        basis = f"Order value / previous FY revenue ≈ {amount} / {revenue_cr} = {ratio:.2%}"
    elif amount is not None and category in {"Acquisition", "Divestment", "Merger"} and market_cap_cr:
        ratio = round(amount / abs(market_cap_cr), 4)
        basis = f"Deal value / market cap ≈ {amount} / {market_cap_cr} = {ratio:.2%}"
    elif amount is not None and category == "Debt" and debt_cr:
        ratio = round(amount / abs(debt_cr), 4)
        basis = f"Debt raised / existing debt ≈ {amount} / {debt_cr} = {ratio:.2%}"
    elif amount is not None and category == "Regulatory Action" and pat_cr:
        ratio = round(amount / abs(pat_cr), 4)
        basis = f"Penalty / PAT ≈ {amount} / {pat_cr} = {ratio:.2%}"
    elif amount is not None and category == "Promoter Transaction" and promoter_pct:
        ratio = None
        basis = f"Headline amount {amount} cr; promoter stake on file is {promoter_pct}% — sale/stake ratio not computed without share count."

    bucket = _bucket(ratio)
    if category == "Regulatory Action" and bucket in {"Unmeasured", "Low"}:
        bucket = "High"
        if "unmeasured" in basis.lower():
            basis = "Regulatory actions are High materiality even without a parsed rupee amount."
    if category in {"Auditor Change", "Plant Shutdown"} and bucket == "Unmeasured":
        bucket = "High"
        basis = f"{category} is treated as High when no rupee amount is parsed."
    if bool(article.get("official")) and bucket == "Low":
        bucket = "Moderate"

    return {
        "category": category,
        "materiality": bucket,
        "amount_cr": amount,
        "ratio": ratio,
        "basis": basis,
        "original_source": str(article.get("url") or article.get("source") or "Source unavailable"),
    }
