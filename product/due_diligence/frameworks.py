"""Sector KPI frameworks. Add a FRAMEWORKS entry to cover a new sector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KpiSpec:
    id: str
    label: str
    table: str
    needles: tuple[str, ...]
    higher_is_better: bool
    kind: str  # level | rate
    unit: str
    pillar: str
    weight: float
    missing_ok: bool = False


def _k(
    kpi_id: str,
    label: str,
    table: str,
    needles: tuple[str, ...],
    *,
    higher_is_better: bool,
    kind: str,
    unit: str,
    pillar: str,
    weight: float,
    missing_ok: bool = False,
) -> KpiSpec:
    return KpiSpec(
        id=kpi_id, label=label, table=table, needles=needles,
        higher_is_better=higher_is_better, kind=kind, unit=unit,
        pillar=pillar, weight=weight, missing_ok=missing_ok,
    )


_GROWTH_SALES = _k(
    "sales", "Revenue / sales", "quarterly_results", ("sales", "revenue"),
    higher_is_better=True, kind="level", unit="₹ cr", pillar="growth", weight=18,
)
_OPM = _k(
    "opm", "Operating margin", "quarterly_results",
    ("opm", "operating profit margin", "financing margin"),
    higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=16,
)
_PAT = _k(
    "pat", "Net profit", "quarterly_results", ("net profit",),
    higher_is_better=True, kind="level", unit="₹ cr", pillar="profitability", weight=14,
)
_EPS = _k(
    "eps", "EPS", "quarterly_results", ("eps in rs", "eps"),
    higher_is_better=True, kind="level", unit="₹", pillar="profitability", weight=8,
)
_PROMOTER = _k(
    "promoter", "Promoter holding", "shareholding", ("promoters",),
    higher_is_better=True, kind="rate", unit="%", pillar="governance", weight=10,
)
_PLEDGE = _k(
    "pledge", "Promoter pledge", "shareholding", ("pledge",),
    higher_is_better=False, kind="rate", unit="%", pillar="governance", weight=8,
)
_CFO = _k(
    "cfo", "Cash from operations", "cash_flow", ("cash from operating",),
    higher_is_better=True, kind="level", unit="₹ cr", pillar="cash", weight=12,
)
_FII = _k(
    "fii", "FII holding", "shareholding", ("fiis", "fii", "foreign institutional"),
    higher_is_better=True, kind="rate", unit="%", pillar="governance", weight=4, missing_ok=True,
)
_DII = _k(
    "dii", "DII holding", "shareholding", ("diis", "dii", "domestic institutional"),
    higher_is_better=True, kind="rate", unit="%", pillar="governance", weight=3, missing_ok=True,
)
_PUBLIC = _k(
    "public", "Public holding", "shareholding", ("public",),
    higher_is_better=False, kind="rate", unit="%", pillar="governance", weight=2, missing_ok=True,
)
_ROE = _k(
    "roe", "Return on equity", "key_ratios", ("roe", "return on equity"),
    higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=6, missing_ok=True,
)
_ROCE = _k(
    "roce", "Return on capital employed", "key_ratios", ("roce", "return on capital"),
    higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=6, missing_ok=True,
)
_DEBT = _k(
    "borrowings", "Borrowings", "balance_sheet", ("borrowings",),
    higher_is_better=False, kind="level", unit="₹ cr", pillar="leverage", weight=8, missing_ok=True,
)

BANK = (
    _k(
        "nii", "Net interest / financing income", "quarterly_results",
        ("revenue", "financing profit"),
        higher_is_better=True, kind="level", unit="₹ cr", pillar="growth", weight=16,
    ),
    _k(
        "nim", "Financing margin (NIM proxy)", "quarterly_results",
        ("financing margin",),
        higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=16,
    ),
    _k(
        "gnpa", "Gross NPA", "quarterly_results",
        ("gross npa", "gnpa", "gross non performing"),
        higher_is_better=False, kind="rate", unit="%", pillar="asset_quality", weight=20,
    ),
    _k(
        "nnpa", "Net NPA", "quarterly_results",
        ("net npa", "nnpa", "net non performing"),
        higher_is_better=False, kind="rate", unit="%", pillar="asset_quality", weight=16,
    ),
    _PAT,
    _PROMOTER,
    _PLEDGE,
    _k(
        "casa", "CASA ratio", "quarterly_results",
        ("casa", "current account savings"),
        higher_is_better=True, kind="rate", unit="%", pillar="funding", weight=8, missing_ok=True,
    ),
    _k(
        "cet1", "CET1 / capital adequacy", "quarterly_results",
        ("cet1", "capital adequacy", "crar", "tier 1"),
        higher_is_better=True, kind="rate", unit="%", pillar="capital", weight=8, missing_ok=True,
    ),
    _k(
        "advances", "Advances / loans", "quarterly_results",
        ("advances", "loans", "credit growth"),
        higher_is_better=True, kind="level", unit="₹ cr", pillar="growth", weight=6, missing_ok=True,
    ),
    _FII, _DII, _PUBLIC,
)

NBFC = (
    _k(
        "nii", "Financing / NII income", "quarterly_results",
        ("revenue", "financing profit", "sales"),
        higher_is_better=True, kind="level", unit="₹ cr", pillar="growth", weight=16,
    ),
    _k(
        "nim", "Financing margin / spread", "quarterly_results",
        ("financing margin", "nim"),
        higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=14,
    ),
    _k(
        "gnpa", "Gross NPA", "quarterly_results",
        ("gross npa", "gnpa", "gross non performing"),
        higher_is_better=False, kind="rate", unit="%", pillar="asset_quality", weight=18,
    ),
    _k(
        "nnpa", "Net NPA", "quarterly_results",
        ("net npa", "nnpa", "net non performing"),
        higher_is_better=False, kind="rate", unit="%", pillar="asset_quality", weight=14,
    ),
    _PAT,
    _PROMOTER,
    _PLEDGE,
    _DEBT,
    _k(
        "aum", "AUM", "quarterly_results",
        ("aum", "assets under management"),
        higher_is_better=True, kind="level", unit="₹ cr", pillar="growth", weight=8, missing_ok=True,
    ),
    _FII, _DII, _PUBLIC,
)

IT = (_GROWTH_SALES, _OPM, _PAT, _EPS, _PROMOTER, _PLEDGE, _CFO, _ROE, _FII, _DII, _PUBLIC)
PHARMA = (_GROWTH_SALES, _OPM, _PAT, _EPS, _PROMOTER, _PLEDGE, _CFO, _ROE, _FII, _DII, _PUBLIC)
INDUSTRIALS = (_GROWTH_SALES, _OPM, _PAT, _CFO, _PROMOTER, _PLEDGE, _DEBT, _ROE, _ROCE, _FII, _DII, _PUBLIC)
AUTO = (_GROWTH_SALES, _OPM, _PAT, _EPS, _CFO, _PROMOTER, _PLEDGE, _DEBT, _ROE, _FII, _DII, _PUBLIC)
FMCG = (_GROWTH_SALES, _OPM, _PAT, _EPS, _CFO, _PROMOTER, _PLEDGE, _ROE, _ROCE, _FII, _DII, _PUBLIC)
REALTY = (_GROWTH_SALES, _OPM, _PAT, _CFO, _PROMOTER, _PLEDGE, _DEBT, _ROE, _FII, _DII, _PUBLIC)
METALS = (_GROWTH_SALES, _OPM, _PAT, _CFO, _PROMOTER, _PLEDGE, _DEBT, _ROE, _FII, _DII, _PUBLIC)
GENERIC = (_GROWTH_SALES, _OPM, _PAT, _PROMOTER, _PLEDGE, _CFO, _FII, _DII, _PUBLIC)

FRAMEWORKS: dict[str, dict[str, Any]] = {
    "bank": {
        "id": "bank",
        "label": "Banks",
        "blurb": "Asset quality, NIM/financing margin, and profit trend — not generic D/E.",
        "kpis": BANK,
        "watch": (
            "Next GNPA / NNPA print versus this quarter.",
            "Financing margin versus the last two quarters.",
            "Deposit vs credit commentary in the next result.",
            "Any RBI / regulatory headline tagged to this bank.",
        ),
    },
    "nbfc": {
        "id": "nbfc",
        "label": "NBFCs",
        "blurb": "AUM/revenue growth, financing margin, NPA trend, leverage via borrowings.",
        "kpis": NBFC,
        "watch": (
            "Gross / net NPA if the table fills in.",
            "Financing margin (spread) trend.",
            "Borrowing-cost commentary in results.",
            "Promoter holding and pledge disclosures.",
        ),
    },
    "it": {
        "id": "it",
        "label": "IT / Software",
        "blurb": "Revenue growth, operating margin, cash conversion. TCV/attrition stay Data unavailable unless filed.",
        "kpis": IT,
        "watch": (
            "Next-quarter sales versus this quarter.",
            "OPM versus the last two quarters.",
            "Large-deal / TCV headlines if they appear in curated news.",
            "Management guidance in the next result filing.",
        ),
    },
    "pharma": {
        "id": "pharma",
        "label": "Pharma",
        "blurb": "Growth and margins from results. USFDA / ANDA items only from sourced news — never inferred.",
        "kpis": PHARMA,
        "watch": (
            "USFDA / plant-observation headlines (only if sourced).",
            "OPM trend over the next two quarters.",
            "Export vs domestic mix if a segment filing is uploaded.",
            "Any warning-letter or import-alert news tagged to this symbol.",
        ),
    },
    "industrials": {
        "id": "industrials",
        "label": "Industrials / capital goods",
        "blurb": "Sales, margin and operating cash. Order-book is Data unavailable unless a filing is on disk.",
        "kpis": INDUSTRIALS,
        "watch": (
            "Next order-inflow / order-book disclosure.",
            "Working-capital / CFO versus PAT.",
            "Margin versus the last two quarters.",
            "Large contract wins or cancellations in curated news.",
        ),
    },
    "auto": {
        "id": "auto",
        "label": "Auto / auto ancillary",
        "blurb": "Volume-linked sales, margins, cash and leverage. Volumes stay Data unavailable unless a filing table exists.",
        "kpis": AUTO,
        "watch": (
            "Monthly volume print versus this quarter's sales.",
            "Margin versus commodity / RM headlines if sourced.",
            "Dealer-inventory commentary in results.",
            "EV mix only if a segment table is on file.",
        ),
    },
    "fmcg": {
        "id": "fmcg",
        "label": "FMCG / consumer",
        "blurb": "Sales, gross/operating margin, cash conversion. Volume vs realisation stays Data unavailable unless filed.",
        "kpis": FMCG,
        "watch": (
            "Volume versus realisation split in the next result.",
            "Gross margin versus commodity costs if disclosed.",
            "Rural/urban mix only from a sourced segment table.",
            "Promoter holding / pledge.",
        ),
    },
    "realty": {
        "id": "realty",
        "label": "Real estate",
        "blurb": "Sales, margin, cash and borrowings. Pre-sales / collections stay Data unavailable unless filed.",
        "kpis": REALTY,
        "watch": (
            "Pre-sales / collections if a filing table appears.",
            "Borrowings versus cash from operations.",
            "Promoter pledge.",
            "Project-delay or RERA headlines if sourced.",
        ),
    },
    "metals": {
        "id": "metals",
        "label": "Metals / energy",
        "blurb": "Realisation-sensitive sales and margins, cash and leverage. Commodity prices are not guessed.",
        "kpis": METALS,
        "watch": (
            "Next-quarter realisation / spread commentary.",
            "Net debt versus CFO.",
            "Power/RM cost if a segment note is uploaded.",
            "Any regulatory or mine-related headline tagged to this symbol.",
        ),
    },
    "generic": {
        "id": "generic",
        "label": "Generic quality",
        "blurb": "Sales, margin, profit, cash, ownership. Sector KPIs are skipped until a framework exists.",
        "kpis": GENERIC,
        "watch": (
            "Whether a sector framework should be added for this industry.",
            "Next quarterly sales and OPM.",
            "Promoter holding / pledge.",
            "Material filings in curated news.",
        ),
    },
}


def get_framework(framework_id: str) -> dict[str, Any]:
    return FRAMEWORKS.get(framework_id) or FRAMEWORKS["generic"]
