"""Shared KPI specs. Frameworks compose these; they do not score on their own."""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    importance: str = "supporting"


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
    importance: str = "supporting",
) -> KpiSpec:
    return KpiSpec(
        id=kpi_id, label=label, table=table, needles=needles,
        higher_is_better=higher_is_better, kind=kind, unit=unit,
        pillar=pillar, weight=weight, missing_ok=missing_ok,
        importance=importance,
    )


def extra(
    kpi_id: str,
    label: str,
    needles: tuple[str, ...],
    *,
    higher_is_better: bool = True,
    kind: str = "rate",
    unit: str = "%",
    pillar: str = "sector",
    weight: float = 8,
    importance: str = "important",
    table: str = "quarterly_results",
) -> KpiSpec:
    """Sector-specific print. Absent until a filing/table actually reports it."""
    return _k(
        kpi_id, label, table, needles,
        higher_is_better=higher_is_better, kind=kind, unit=unit,
        pillar=pillar, weight=weight, missing_ok=True, importance=importance,
    )


_GROWTH_SALES = _k(
    "sales", "Revenue / sales", "quarterly_results", ("sales", "revenue"),
    higher_is_better=True, kind="level", unit="₹ cr", pillar="growth", weight=18,
    importance="critical",
)
_OPM = _k(
    "opm", "Operating margin", "quarterly_results",
    ("opm", "operating profit margin"),
    higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=16,
    importance="critical",
)
_PAT = _k(
    "pat", "Net profit", "quarterly_results", ("net profit",),
    higher_is_better=True, kind="level", unit="₹ cr", pillar="profitability", weight=14,
    importance="important",
)
_EPS = _k(
    "eps", "EPS", "quarterly_results", ("eps in rs", "eps"),
    higher_is_better=True, kind="level", unit="₹", pillar="profitability", weight=8,
    importance="supporting",
)
_PROMOTER = _k(
    "promoter", "Promoter holding", "shareholding", ("promoters",),
    higher_is_better=True, kind="rate", unit="%", pillar="governance", weight=10,
    importance="optional", missing_ok=True,
)
_PLEDGE = _k(
    "pledge", "Promoter pledge", "shareholding", ("pledge",),
    higher_is_better=False, kind="rate", unit="%", pillar="governance", weight=8,
    importance="optional", missing_ok=True,
)
_CFO = _k(
    "cfo", "Cash from operations", "cash_flow", ("cash from operating",),
    higher_is_better=True, kind="level", unit="₹ cr", pillar="cash", weight=12,
    importance="important",
)
_FII = _k(
    "fii", "FII holding", "shareholding", ("fiis", "fii", "foreign institutional"),
    higher_is_better=True, kind="rate", unit="%", pillar="governance", weight=4,
    missing_ok=True, importance="optional",
)
_DII = _k(
    "dii", "DII holding", "shareholding", ("diis", "dii", "domestic institutional"),
    higher_is_better=True, kind="rate", unit="%", pillar="governance", weight=3,
    missing_ok=True, importance="optional",
)
_PUBLIC = _k(
    "public", "Public holding", "shareholding", ("public",),
    higher_is_better=False, kind="rate", unit="%", pillar="governance", weight=2,
    missing_ok=True, importance="optional",
)
_ROA = _k(
    "roa", "Return on assets", "key_ratios", ("roa", "return on assets"),
    higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=6,
    missing_ok=True, importance="important",
)
_ROE = _k(
    "roe", "Return on equity", "key_ratios", ("roe", "return on equity"),
    higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=6,
    missing_ok=True, importance="important",
)
_ROCE = _k(
    "roce", "Return on capital employed", "key_ratios", ("roce", "return on capital"),
    higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=6,
    missing_ok=True, importance="important",
)
_DEBT = _k(
    "borrowings", "Borrowings", "balance_sheet", ("borrowings",),
    higher_is_better=False, kind="level", unit="₹ cr", pillar="leverage", weight=8,
    missing_ok=True, importance="important",
)

_OWN = (_PROMOTER, _PLEDGE, _FII, _DII, _PUBLIC)
_CFO_CRITICAL = replace(_CFO, importance="critical")

BANK = (
    _k(
        "nii", "Net interest / financing income", "quarterly_results",
        ("revenue", "financing profit"),
        higher_is_better=True, kind="level", unit="₹ cr", pillar="growth", weight=16,
        importance="important",
    ),
    _k(
        "nim", "Net interest margin (NIM)", "quarterly_results",
        ("net interest margin", "nim"),
        higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=16,
        importance="critical",
    ),
    _k(
        "gnpa", "Gross NPA", "quarterly_results",
        ("gross npa", "gnpa", "gross non performing"),
        higher_is_better=False, kind="rate", unit="%", pillar="asset_quality", weight=20,
        importance="critical",
    ),
    _k(
        "nnpa", "Net NPA", "quarterly_results",
        ("net npa", "nnpa", "net non performing"),
        higher_is_better=False, kind="rate", unit="%", pillar="asset_quality", weight=16,
        importance="critical",
    ),
    _PAT,
    extra("casa", "CASA ratio", ("casa", "current account savings"), importance="important"),
    extra("cet1", "CET1", ("cet1", "common equity tier"), pillar="capital", importance="important"),
    extra("crar", "CRAR / capital adequacy", ("crar", "capital adequacy"), pillar="capital", weight=6, importance="important"),
    extra(
        "advances", "Advances / loans", ("gross advances", "net advances", "total advances"),
        kind="level", unit="₹ cr", pillar="growth", weight=6, table="balance_sheet",
    ),
    extra(
        "deposits", "Deposits", ("deposits", "total deposits"),
        kind="level", unit="₹ cr", pillar="funding", weight=6, table="balance_sheet",
    ),
    extra("pcr", "Provision coverage (PCR)", ("pcr", "provision coverage"), pillar="asset_quality", weight=6, importance="supporting"),
    extra("slippages", "Slippages", ("slippage", "slippages"), higher_is_better=False, pillar="asset_quality", weight=5),
    extra("credit_cost", "Credit cost", ("credit cost", "credit costs"), higher_is_better=False, pillar="asset_quality", weight=5),
    extra(
        "loan_deposit", "Loan / deposit ratio",
        ("credit deposit", "cd ratio", "loan deposit", "loan to deposit"),
        higher_is_better=False, pillar="funding", weight=4, importance="supporting",
    ),
    _ROA, _ROE, *_OWN,
)

NBFC = (
    _k(
        "nii", "Financing / NII income", "quarterly_results",
        ("revenue", "financing profit", "sales"),
        higher_is_better=True, kind="level", unit="₹ cr", pillar="growth", weight=16,
        importance="important",
    ),
    _k(
        "nim", "Financing margin / spread", "quarterly_results",
        ("financing margin", "nim", "net interest margin"),
        higher_is_better=True, kind="rate", unit="%", pillar="profitability", weight=14,
        importance="critical",
    ),
    _k(
        "gnpa", "Gross NPA", "quarterly_results",
        ("gross npa", "gnpa", "gross non performing"),
        higher_is_better=False, kind="rate", unit="%", pillar="asset_quality", weight=18,
        importance="critical",
    ),
    _k(
        "nnpa", "Net NPA", "quarterly_results",
        ("net npa", "nnpa", "net non performing"),
        higher_is_better=False, kind="rate", unit="%", pillar="asset_quality", weight=14,
        importance="critical",
    ),
    _PAT, _DEBT,
    extra("aum", "AUM", ("aum", "assets under management"), kind="level", unit="₹ cr", pillar="growth"),
    extra("credit_cost", "Credit cost", ("credit cost", "credit costs"), higher_is_better=False, pillar="asset_quality", weight=5),
    *_OWN,
)

NBFC_GOLD = NBFC + (
    extra("ltv", "Loan-to-value (LTV)", ("ltv", "loan to value"), higher_is_better=False, weight=8),
    extra("gold_price", "Gold-price sensitivity", ("gold price", "gold holding"), kind="level", unit="", weight=4, importance="supporting"),
)
NBFC_HOUSING = NBFC + (
    extra("ticket_size", "Average ticket size", ("ticket size", "average ticket"), kind="level", unit="₹", weight=6),
    extra("affordable", "Affordable-housing mix", ("affordable housing",), weight=6, importance="supporting"),
)

IT = (
    _GROWTH_SALES, _OPM, _PAT, _EPS, _CFO, _ROE,
    extra("tcv", "TCV / large-deal bookings", ("tcv", "total contract value", "large deal"), kind="level", unit="₹ cr"),
    extra("attrition", "Attrition", ("attrition",), higher_is_better=False),
    extra(
        "cc_growth", "Constant-currency growth",
        ("constant currency growth", "constant currency"),
        weight=8,
    ),
    extra("utilization", "Utilization", ("utilization", "utilisation"), importance="supporting", weight=5),
    extra("client_concentration", "Top-client concentration", ("top 10 client", "top-10 client", "client concentration"), higher_is_better=False, importance="supporting", weight=6),
    *_OWN,
)
SOFTWARE_PRODUCT = (
    _GROWTH_SALES, _OPM, _PAT, _EPS, _CFO, _ROE,
    extra("subscription", "Subscription / recurring mix", ("subscription", "recurring revenue", "arr"), weight=10),
    extra("retention", "Net retention", ("net retention", "nrr"), weight=8),
    *_OWN,
)
PHARMA = (
    _GROWTH_SALES, _OPM, _PAT, _EPS, _CFO, _ROE,
    extra("rnd", "R&D / sales", ("r&d", "research and development"), table="profit_loss", weight=8),
    extra("us_sales", "US / regulated-market mix", ("us sales", "regulated market"), weight=8),
    *_OWN,
)
HOSPITALS = (
    extra("occupancy", "Occupancy", ("occupancy", "bed occupancy"), importance="critical", weight=16),
    extra("arpob", "ARPOB", ("arpob", "revenue per occupied bed"), kind="level", unit="₹", importance="critical", weight=14),
    _GROWTH_SALES, _OPM, _PAT, _CFO, _DEBT,
    extra("beds", "Operational beds", ("operational beds", "bed capacity"), kind="level", unit="", weight=8),
    *_OWN,
)
DIAGNOSTICS = (
    extra("test_volumes", "Test volumes", ("test volume", "tests performed"), kind="level", unit="", importance="critical", weight=16),
    _GROWTH_SALES, _OPM, _PAT, _CFO,
    extra("realization", "Realization per test", ("realization", "revenue per test"), kind="level", unit="₹", weight=10),
    extra("centres", "Labs / collection centres", ("collection centre", "patient touch"), kind="level", unit="", importance="supporting", weight=5),
    *_OWN,
)

def _real_economy(*sector_kpis: KpiSpec, cfo_critical: bool = False) -> tuple[KpiSpec, ...]:
    cfo = _CFO_CRITICAL if cfo_critical else _CFO
    return (_GROWTH_SALES, _OPM, _PAT, cfo, _DEBT, _ROE) + sector_kpis + _OWN


AUTO = _real_economy(
    extra("volumes", "Unit volumes", ("volume", "units sold", "wholesale volume"), kind="level", unit="", weight=12, importance="important"),
    extra("realization", "Realization / unit", ("realization", "average selling price"), kind="level", unit="₹", weight=6, importance="supporting"),
)
AUTO_ANCILLARY = _real_economy(
    extra("oem_concentration", "OEM / customer concentration", ("oem", "customer concentration"), higher_is_better=False, weight=8),
    extra("content_per_vehicle", "Content per vehicle", ("content per vehicle",), kind="level", unit="₹", importance="supporting", weight=5),
)
FMCG = _real_economy(
    extra("volume_growth", "Volume growth", ("volume growth", "underlying volume"), weight=10),
    extra("gross_margin", "Gross margin", ("gross margin",), weight=8),
    _ROCE,
)
CONSUMER_DISC = _real_economy(
    extra("sss", "Same-store / like-for-like growth", ("same store", "sss", "like for like"), weight=10),
)
RETAIL = _real_economy(
    extra("sss", "Same-store sales growth", ("same store", "sss", "like for like"), importance="critical", weight=14),
    extra("store_count", "Store count", ("store count", "stores"), kind="level", unit="", weight=6),
    extra("inventory_days", "Inventory days", ("inventory days", "inventory turnover"), higher_is_better=False, kind="level", unit="days", weight=8),
)
CAPITAL_GOODS = _real_economy(
    extra("order_book", "Order book", ("order book", "order-book"), kind="level", unit="₹ cr", importance="critical", weight=14),
    extra("order_inflow", "Order inflow", ("order inflow", "order intake"), kind="level", unit="₹ cr", weight=10),
    extra("book_to_bill", "Book-to-bill", ("book to bill", "book-to-bill"), kind="level", unit="x", importance="supporting", weight=6),
    cfo_critical=True,
)
INDUSTRIALS = _real_economy(
    extra("order_book", "Order book", ("order book", "order-book"), kind="level", unit="₹ cr", weight=10),
    cfo_critical=True,
)
DEFENCE = _real_economy(
    extra("order_book", "Defence order book", ("order book", "order-book"), kind="level", unit="₹ cr", importance="critical", weight=14),
    extra("order_inflow", "Order inflow", ("order inflow",), kind="level", unit="₹ cr", weight=8),
    cfo_critical=True,
)
INFRASTRUCTURE = _real_economy(
    extra("order_book", "Order book", ("order book", "order-book"), kind="level", unit="₹ cr", importance="critical", weight=14),
    extra("debtor_days", "Receivable / debtor days", ("debtor days", "receivable days"), higher_is_better=False, kind="level", unit="days", weight=8),
    cfo_critical=True,
)
CEMENT = _real_economy(
    extra("cement_volume", "Cement volume", ("sales volume", "despatch", "cement volume"), kind="level", unit="mt", weight=12),
    extra("ebitda_t", "EBITDA / tonne", ("ebitda/t", "ebitda per tonne", "ebitda/tonne"), kind="level", unit="₹", weight=10),
    extra("realization", "Realization / tonne", ("realization",), kind="level", unit="₹", importance="supporting", weight=6),
)
METALS = _real_economy(
    extra("production", "Production volume", ("production", "steel production"), kind="level", unit="", weight=10),
    extra("realization", "Realization", ("realization", "nscr"), kind="level", unit="₹", weight=8),
    extra("ebitda_t", "EBITDA / tonne", ("ebitda/t", "ebitda per tonne"), kind="level", unit="₹", weight=8),
)
MINING = _real_economy(
    extra("production", "Production volume", ("production", "output"), kind="level", unit="", weight=12),
    extra("realization", "Realization", ("realization",), kind="level", unit="₹", weight=8),
)
CHEMICALS = _real_economy(
    extra("volume_growth", "Volume growth", ("volume growth",), weight=8),
    extra("spreads", "Spreads / contribution", ("spread", "contribution"), weight=8),
)
SPECIALTY_CHEMICALS = _real_economy(
    extra("volume_growth", "Volume growth", ("volume growth",), weight=8),
    extra("export_mix", "Export mix", ("export",), weight=6, importance="supporting"),
)
OIL_GAS = _real_economy(
    extra("production", "Production", ("production", "oil production", "gas production"), kind="level", unit="", weight=10),
    extra("realization", "Realization", ("realization", "gas price"), kind="level", unit="", weight=8),
)
REFINING = _real_economy(
    extra("grm", "Gross refining margin", ("grm", "gross refining margin"), kind="level", unit="$/bbl", importance="critical", weight=14),
    extra("throughput", "Throughput", ("throughput", "refining throughput"), kind="level", unit="mmt", weight=8),
)
TELECOM = (
    extra("subscribers", "Subscribers", ("subscriber", "customers"), kind="level", unit="", importance="critical", weight=14),
    extra("arpu", "ARPU", ("arpu",), kind="level", unit="₹", importance="critical", weight=14),
    _GROWTH_SALES, _OPM, _PAT, _CFO, _DEBT,
    extra("churn", "Churn", ("churn",), higher_is_better=False, weight=8),
    extra("spectrum", "Spectrum / licence liability", ("spectrum",), kind="level", unit="₹ cr", higher_is_better=False, importance="supporting", weight=5),
    *_OWN,
)
UTILITIES = _real_economy(
    extra("capacity", "Capacity", ("capacity", "installed capacity"), kind="level", unit="mw", weight=8),
)
POWER_GEN = _real_economy(
    extra("capacity", "Installed capacity", ("installed capacity", "capacity"), kind="level", unit="mw", importance="critical", weight=12),
    extra("plf", "PLF / utilization", ("plf", "plant load factor"), weight=10),
)
POWER_TRANS = _real_economy(
    extra("rab", "Regulated asset base", ("regulated asset", "rab"), kind="level", unit="₹ cr", weight=10),
    extra("transmission", "Transmission growth", ("transmission", "network"), weight=8),
)
REALTY = _real_economy(
    extra("presales", "Pre-sales / bookings", ("pre-sales", "presales", "bookings"), kind="level", unit="₹ cr", importance="critical", weight=14),
    extra("collections", "Collections", ("collections",), kind="level", unit="₹ cr", weight=10),
    extra("land_bank", "Land bank", ("land bank",), kind="level", unit="", importance="supporting", weight=4),
    cfo_critical=True,
)
LOGISTICS = _real_economy(
    extra("volumes", "Volumes / tonnage", ("tonnage", "volume handled"), kind="level", unit="", weight=10),
    extra("utilization", "Utilization", ("utilization", "load factor"), weight=8),
)
AIRLINES = (
    extra("ask", "ASK", ("available seat kilometre", "ask"), kind="level", unit="", importance="critical", weight=12),
    extra("load_factor", "Load factor", ("load factor", "passenger load"), importance="critical", weight=12),
    extra("cask", "CASK", ("cask", "cost per ask"), higher_is_better=False, kind="level", unit="₹", weight=8),
    extra("yield", "Passenger yield", ("yield", "rask"), kind="level", unit="₹", weight=8),
    _GROWTH_SALES, _OPM, _PAT, _DEBT, *_OWN,
)
HOTELS = (
    extra("occupancy", "Occupancy", ("occupancy",), importance="critical", weight=14),
    extra("arr", "ARR", ("average room rate", "arr"), kind="level", unit="₹", weight=10),
    extra("revpar", "RevPAR", ("revpar", "revenue per available room"), kind="level", unit="₹", importance="critical", weight=12),
    _GROWTH_SALES, _OPM, _PAT, _CFO, _DEBT, *_OWN,
)
LIFE_INSURANCE = (
    extra("ape", "APE / new business premium", ("ape", "annualized premium"), kind="level", unit="₹ cr", importance="critical", weight=16),
    extra("vnb_margin", "VNB margin", ("vnb margin", "new business margin"), importance="critical", weight=16),
    extra("persistency", "Persistency (13m)", ("persistency",), weight=10),
    extra("solvency", "Solvency ratio", ("solvency",), kind="level", unit="x", weight=10),
    _GROWTH_SALES, _PAT, _ROE, *_OWN,
)
GENERAL_INSURANCE = (
    extra("gwp", "Gross written premium", ("gross written", "gwp"), kind="level", unit="₹ cr", importance="critical", weight=16),
    extra("combined", "Combined ratio", ("combined ratio",), higher_is_better=False, importance="critical", weight=16),
    extra("loss_ratio", "Loss ratio", ("loss ratio",), higher_is_better=False, weight=8),
    extra("solvency", "Solvency ratio", ("solvency",), kind="level", unit="x", weight=8),
    _GROWTH_SALES, _PAT, *_OWN,
)
EXCHANGE = (
    extra("volumes", "Trading volumes", ("trading volume", "adtv", "turnover"), kind="level", unit="", importance="critical", weight=16),
    _GROWTH_SALES, _OPM, _PAT, _CFO, _ROE, *_OWN,
)
BROKER = (
    extra("active_clients", "Active clients", ("active client", "client base"), kind="level", unit="", importance="critical", weight=14),
    extra("market_share", "Market share", ("market share",), weight=8),
    _GROWTH_SALES, _OPM, _PAT, _CFO, *_OWN,
)
AMC = (
    extra("aum", "AUM", ("aum", "assets under management"), kind="level", unit="₹ cr", importance="critical", weight=16),
    extra("net_flows", "Net flows", ("net inflow", "net flow"), kind="level", unit="₹ cr", weight=10),
    extra("equity_mix", "Equity AUM mix", ("equity aum",), weight=8),
    _GROWTH_SALES, _OPM, _PAT, _ROE, *_OWN,
)
MEDIA = _real_economy(
    extra("subscribers", "Subscribers / users", ("subscriber", "viewership"), kind="level", unit="", weight=8),
)
TEXTILES = _real_economy(
    extra("capacity_util", "Capacity utilization", ("capacity utilisation", "capacity utilization"), weight=8),
)
AGRI = _real_economy(
    extra("volume_growth", "Volume growth", ("volume growth",), weight=8),
)
GENERIC = (_GROWTH_SALES, _OPM, _PAT, _CFO, *_OWN)
