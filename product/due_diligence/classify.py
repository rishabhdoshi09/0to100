"""Map a company onto a sector due-diligence framework. Easy to extend."""
from __future__ import annotations

import re
from typing import Any, Mapping

from product.due_diligence.series import find_row, normalize_label

# NSE comment-group names from data/nse_universe.py → framework id.
_SECTOR_TO_FRAMEWORK: dict[str, str] = {
    "banking & finance": "nbfc",  # refined by about-text / NPA rows
    "it / software": "it",
    "pharma & healthcare": "pharma",
    "manufacturing & capital goods": "industrials",
    "infrastructure & construction": "industrials",
    "metals & mining": "industrials",
    "auto": "auto",
    "fmcg": "fmcg",
    "consumer & retail / apparel": "fmcg",
    "real estate": "realty",
    "energy & power": "metals",
    "specialty chemicals": "industrials",
}

_NBFC_RE = re.compile(r"nbfc|non[\s-]?banking|housing finance|vehicle financ|finance company")
_BANK_RE = re.compile(
    r"\bsmall finance bank\b|\bprivate sector bank\b|\bpublic sector bank\b|"
    r"\bcommercial bank|\bbanks?\b"
)
# Match "banking" but not the NBFC phrase "non-banking".
_BANKING_RE = re.compile(r"(?<!non-)(?<!non )banking")
_IT_WORDS = ("software", "it services", "information technology", "saas")
_PHARMA_WORDS = ("pharma", "pharmaceutical", "drug", "formulation")
_INDUSTRIAL_WORDS = ("capital goods", "engineering", "industrial", "pipes", "order book", "infrastructure")


def _mentions_nbfc(about: str) -> bool:
    return bool(_NBFC_RE.search(about))


def _mentions_bank(about: str) -> bool:
    """Word-aware bank match. 'non-banking' / 'vehicle financing' are not banks."""
    if _mentions_nbfc(about) and not _BANK_RE.search(about):
        return False
    return bool(_BANK_RE.search(about) or _BANKING_RE.search(about))


def classify_company(
    symbol: str,
    *,
    sector: str = "",
    about: str = "",
    quarterly_rows: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pick a framework. Unknown stays generic — never guessed into a bank."""
    sector_name = str(sector or "").strip()
    if not sector_name:
        try:
            from scan.sector_heat import sector_of
            sector_name = str(sector_of(symbol) or "")
        except Exception:
            sector_name = ""
    sector_l = sector_name.lower()
    about_l = str(about or "").lower()
    blob = f"{about_l} {sector_l}"
    has_npa = find_row(quarterly_rows, ("gross npa", "net npa", "gnpa", "nnpa")) is not None
    framework = "generic"
    reason = "No sector-specific table match; using the generic quality framework."

    mapped = _SECTOR_TO_FRAMEWORK.get(normalize_label(sector_name))
    # Match bank/NBFC keywords against company about-text only. Do not search
    # the NSE bucket — "Banking & Finance" would classify every NBFC as a bank.
    if _mentions_bank(about_l):
        framework = "bank"
        reason = "Company description classifies this as a bank."
    elif _mentions_nbfc(about_l) or mapped == "nbfc" or (
        has_npa and not _mentions_bank(about_l)
    ):
        framework = "nbfc"
        reason = "Finance company with lending metrics — NBFC framework."
    elif mapped == "it" or any(w in blob for w in _IT_WORDS):
        framework = "it"
        reason = "IT / software sector map or description."
    elif mapped == "pharma" or any(w in blob for w in _PHARMA_WORDS):
        framework = "pharma"
        reason = "Pharma / healthcare sector map or description."
    elif mapped == "industrials" or any(w in blob for w in _INDUSTRIAL_WORDS):
        framework = "industrials"
        reason = "Capital-goods / industrials sector map or description."
    elif mapped:
        framework = mapped
        reason = f"Sector map '{sector_name}'."
    elif sector_name:
        reason = f"Sector '{sector_name}' has no dedicated framework yet — generic checks only."

    return {
        "symbol": str(symbol or "").upper(),
        "sector": sector_name or "Unclassified",
        "industry": sector_name or "Unclassified",
        "framework_id": framework,
        "framework_reason": reason,
        "about": str(about or "").strip(),
        "business_model": str(about or "").strip() or "Data unavailable",
        "revenue_drivers": "Data unavailable — no segment table on file.",
    }
