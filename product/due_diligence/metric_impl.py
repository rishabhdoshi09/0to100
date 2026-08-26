"""A framework KPI is not implemented merely by being listed.

Implemented = validated acquisition path + canonical definition + period
handling + provenance + tests. Missing stays missing. No LLM.
"""
from __future__ import annotations

from dataclasses import dataclass


RELIABLE = frozenset({"obtainable", "derivable"})
PRIORITY_FRAMEWORKS = ("bank", "it", "pharma", "capital_goods", "retail")


@dataclass(frozen=True)
class MetricImpl:
    kpi_id: str
    reliability: str  # obtainable | derivable | inconsistent | not_reliably_available
    paths: tuple[str, ...]
    definition: str
    period_policy: str
    false_positive_guard: str
    tests: tuple[str, ...] = ()

    @property
    def implemented(self) -> bool:
        return self.reliability in RELIABLE and bool(self.tests)


def _m(
    kpi_id: str,
    reliability: str,
    paths: tuple[str, ...],
    definition: str,
    *,
    period: str = "Prefer the issuer's latest comparable quarterly print; never mix quarterly with annual or TTM.",
    guard: str = "Require an explicit label; drop out-of-bounds and colliding wording.",
    tests: tuple[str, ...] = (),
) -> MetricImpl:
    return MetricImpl(
        kpi_id=kpi_id,
        reliability=reliability,
        paths=paths,
        definition=definition,
        period_policy=period,
        false_positive_guard=guard,
        tests=tests,
    )


_TABLE = "results_table"
_OVERLAY = "table_overlay"
_TEXT = "filing_text"
_KEY = "key_ratio_snapshot"
_DERIVED = "derived"

_Q = "Latest dated column on the quarterly results table; YoY uses four steps back only when both prints are quarterly."
_A = "Annual P&L / cash-flow / balance-sheet table; YoY is one step back. Do not trend against a quarterly print."
_S = "Key-ratio snapshot only — no QoQ/YoY unless a dated series exists."
_T = "Period taken from nearby Qx FYxx / month-year wording; stamp standalone vs consolidated when present. No trend from a single extracted print."

# tests= names of pytest functions that lock the extractor. Empty tests ⇒ not implemented.
METRIC_IMPL: dict[str, MetricImpl] = {
    "sales": _m("sales", "obtainable", (_TABLE,), "Reported quarterly sales / revenue.", period=_Q, tests=("test_it_framework_does_not_ask_for_gnpa", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "opm": _m("opm", "obtainable", (_TABLE,), "Operating profit margin %.", period=_Q, tests=("test_it_framework_does_not_ask_for_gnpa", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "pat": _m("pat", "obtainable", (_TABLE,), "Reported net profit / PAT.", period=_Q, tests=("test_bank_report_supports_improving_asset_quality", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "eps": _m("eps", "obtainable", (_TABLE,), "EPS as reported on the results table.", period=_Q, tests=("test_it_framework_does_not_ask_for_gnpa",)),
    "cfo": _m("cfo", "obtainable", (_TABLE,), "Cash from operating activities.", period=_A, tests=("test_it_framework_does_not_ask_for_gnpa", "test_industrials_need_sales_margin_or_cash")),
    "promoter": _m("promoter", "obtainable", (_TABLE,), "Promoter holding %.", period=_Q, tests=("test_bank_report_supports_improving_asset_quality",)),
    "pledge": _m("pledge", "obtainable", (_TABLE, _TEXT), "Promoter pledge %.", period=_Q, guard="Ignore charity/pledge-to-give wording.", tests=("test_extract_gnpa_and_guidance_from_filing_text",)),
    "fii": _m("fii", "obtainable", (_TABLE,), "FII holding %.", period=_Q, tests=("test_priority_framework_full_sources_reach_high_decision_coverage",)),
    "dii": _m("dii", "obtainable", (_TABLE,), "DII holding %.", period=_Q, tests=("test_priority_framework_full_sources_reach_high_decision_coverage",)),
    "public": _m("public", "obtainable", (_TABLE,), "Public holding %.", period=_Q, tests=("test_priority_framework_full_sources_reach_high_decision_coverage",)),
    "borrowings": _m("borrowings", "obtainable", (_TABLE,), "Reported borrowings.", period=_A, tests=("test_industrials_need_sales_margin_or_cash", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "roa": _m("roa", "obtainable", (_KEY, _OVERLAY, _TEXT), "Return on assets %.", period=_S, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "roe": _m("roe", "obtainable", (_TABLE, _KEY, _TEXT), "Return on equity %.", period=_S, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "roce": _m("roce", "obtainable", (_KEY,), "Return on capital employed %.", period=_S, tests=("test_priority_framework_full_sources_reach_high_decision_coverage",)),
    "nii": _m("nii", "obtainable", (_TABLE,), "Net interest / financing income. Banks use the revenue/financing-profit row, not generic sales.", period=_Q, tests=("test_bank_report_supports_improving_asset_quality", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "nim": _m("nim", "obtainable", (_TABLE, _OVERLAY, _TEXT), "Net interest margin %. Not financing margin, OPM, or investment yield.", period=_Q, guard="Needles are net interest margin / NIM only; yield and operating margin are rejected.", tests=("test_extract_bank_kpis_from_results_text", "test_nim_pressure_is_a_concern_not_invented")),
    "gnpa": _m("gnpa", "obtainable", (_TABLE, _OVERLAY, _TEXT), "Gross NPA %.", period=_Q, tests=("test_extract_gnpa_and_guidance_from_filing_text", "test_bank_report_supports_improving_asset_quality")),
    "nnpa": _m("nnpa", "obtainable", (_TABLE, _OVERLAY, _TEXT), "Net NPA %.", period=_Q, tests=("test_extract_gnpa_and_guidance_from_filing_text",)),
    "casa": _m("casa", "obtainable", (_OVERLAY, _TEXT), "CASA ratio %.", period=_Q, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "cet1": _m("cet1", "obtainable", (_OVERLAY, _TEXT, _KEY), "CET1 ratio %.", period=_Q, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "crar": _m("crar", "obtainable", (_OVERLAY, _TEXT, _KEY), "CRAR / capital adequacy %.", period=_Q, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "advances": _m("advances", "obtainable", (_TABLE, _OVERLAY, _TEXT), "Gross/net/total advances.", period=_A, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "deposits": _m("deposits", "obtainable", (_TABLE, _OVERLAY, _TEXT), "Total / customer deposits.", period=_A, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "pcr": _m("pcr", "obtainable", (_OVERLAY, _TEXT), "Provision coverage ratio %.", period=_Q, tests=("test_extract_bank_kpis_from_results_text",)),
    "slippages": _m("slippages", "obtainable", (_OVERLAY, _TEXT), "Slippage ratio %.", period=_Q, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "credit_cost": _m("credit_cost", "obtainable", (_OVERLAY, _TEXT), "Credit cost %.", period=_Q, tests=("test_extract_bank_kpis_from_results_text", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "loan_deposit": _m("loan_deposit", "obtainable", (_OVERLAY, _TEXT), "Credit/deposit or loan/deposit ratio %.", period=_Q, tests=("test_extract_bank_kpis_from_results_text",)),
    "aum": _m("aum", "obtainable", (_OVERLAY,), "Assets under management, when a labeled AUM row exists.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "tcv": _m("tcv", "obtainable", (_OVERLAY, _TEXT), "Total contract value / large-deal bookings in ₹ cr.", period=_T, guard="Require TCV or total contract value next to a crore/billion amount.", tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "attrition": _m("attrition", "obtainable", (_OVERLAY, _TEXT), "Employee attrition %.", period=_Q, guard="Reject NPA/deposit/slippage attrition wording.", tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "cc_growth": _m("cc_growth", "obtainable", (_TABLE, _TEXT), "Constant-currency revenue growth %.", period=_Q, guard="Require the phrase constant currency — never a bare CC token.", tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "rnd": _m("rnd", "obtainable", (_TABLE, _OVERLAY, _TEXT), "R&D / sales % when labeled as a rate.", period=_A, guard="Require R&D or research and development next to a percent.", tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "us_sales": _m("us_sales", "obtainable", (_OVERLAY, _TEXT), "US / regulated-market revenue mix %.", period=_Q, guard="Require US plus sales/revenue/mix/formulations.", tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "order_book": _m("order_book", "obtainable", (_OVERLAY, _TEXT), "Order book in ₹ cr.", period=_T, tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "order_inflow": _m("order_inflow", "obtainable", (_OVERLAY, _TEXT), "Order inflow / intake in ₹ cr.", period=_T, tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "sss": _m("sss", "obtainable", (_OVERLAY, _TEXT), "Same-store / like-for-like sales growth %. Negative prints are valid.", period=_Q, guard="Require same-store, SSS, or like-for-like. Allow negative rates.", tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "store_count": _m("store_count", "obtainable", (_OVERLAY, _TEXT), "Store count.", period=_Q, guard="Require store count / stores; reject restored/restore.", tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "inventory_days": _m("inventory_days", "obtainable", (_OVERLAY, _TEXT), "Inventory days (level, not a 0–100 rate).", period=_Q, tests=("test_priority_extractors_reject_false_positives", "test_priority_framework_full_sources_reach_high_decision_coverage")),
    "occupancy": _m("occupancy", "obtainable", (_OVERLAY, _TEXT), "Bed / hotel occupancy %.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "utilization": _m("utilization", "inconsistent", (_OVERLAY,), "Utilization % — labels collide with capacity utilization, PLF and load factor.", period=_Q, tests=()),
    "client_concentration": _m("client_concentration", "inconsistent", (), "Top-client concentration is rarely a stable labeled row.", period=_Q, tests=()),
    "subscription": _m("subscription", "not_reliably_available", (), "Recurring/ARR mix is not in standard NSE result tables.", period=_Q, tests=()),
    "retention": _m("retention", "not_reliably_available", (), "Net retention is a SaaS print, not a standard filing row.", period=_Q, tests=()),
    "ltv": _m("ltv", "not_reliably_available", (), "Gold-loan LTV is presentation-only and not in the standard results table.", period=_Q, tests=()),
    "gold_price": _m("gold_price", "not_reliably_available", (), "Gold-price sensitivity is not a company-reported line.", period=_Q, tests=()),
    "ticket_size": _m("ticket_size", "not_reliably_available", (), "Average ticket size is presentation-only.", period=_Q, tests=()),
    "affordable": _m("affordable", "not_reliably_available", (), "Affordable-housing mix is presentation-only.", period=_Q, tests=()),
    "arpob": _m("arpob", "obtainable", (_OVERLAY,), "ARPOB when a labeled row exists.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "beds": _m("beds", "inconsistent", (_OVERLAY,), "Operational beds vs capacity beds are often unlabeled.", period=_Q, tests=()),
    "test_volumes": _m("test_volumes", "inconsistent", (_OVERLAY,), "Test-volume units and labels vary by issuer.", period=_Q, tests=()),
    "realization": _m("realization", "inconsistent", (_OVERLAY,), "Realization collides across auto, cement, metals and diagnostics.", period=_Q, tests=()),
    "centres": _m("centres", "not_reliably_available", (), "Lab / collection-centre counts are presentation-only.", period=_Q, tests=()),
    "volumes": _m("volumes", "inconsistent", (_OVERLAY,), "Unit volumes vs rupee volumes vs tonnage share the same word.", period=_Q, tests=()),
    "oem_concentration": _m("oem_concentration", "not_reliably_available", (), "OEM concentration is rarely a labeled series.", period=_Q, tests=()),
    "content_per_vehicle": _m("content_per_vehicle", "not_reliably_available", (), "Content per vehicle is presentation-only.", period=_Q, tests=()),
    "volume_growth": _m("volume_growth", "inconsistent", (_OVERLAY, _TEXT), "Volume growth wording collides with revenue growth.", period=_Q, tests=()),
    "gross_margin": _m("gross_margin", "obtainable", (_TABLE,), "Gross margin % when an OPM-style gross-margin row exists.", period=_Q, tests=("test_priority_framework_full_sources_reach_high_decision_coverage",)),
    "book_to_bill": _m("book_to_bill", "inconsistent", (_OVERLAY,), "Book-to-bill is sporadically disclosed.", period=_Q, tests=()),
    "debtor_days": _m("debtor_days", "inconsistent", (_OVERLAY,), "Debtor / receivable days labels vary.", period=_Q, tests=()),
    "defence_orders": _m("defence_orders", "not_reliably_available", (), "Defence-only order split is not a standard table.", period=_Q, tests=()),
    "cement_volume": _m("cement_volume", "inconsistent", (_OVERLAY,), "Cement volume units (mt vs lakh tonnes) are not normalized yet.", period=_Q, tests=()),
    "ebitda_t": _m("ebitda_t", "inconsistent", (_OVERLAY,), "EBITDA/tonne units and currency mix are not normalized yet.", period=_Q, tests=()),
    "production": _m("production", "inconsistent", (_OVERLAY,), "Production volume units vary by commodity.", period=_Q, tests=()),
    "export_mix": _m("export_mix", "inconsistent", (_OVERLAY,), "Export share is a segment mix, not a stable KPI row.", period=_Q, tests=()),
    "spreads": _m("spreads", "not_reliably_available", (), "Chemical spreads are not a standard filing row.", period=_Q, tests=()),
    "grm": _m("grm", "inconsistent", (_OVERLAY,), "GRM units ($/bbl vs ₹) are not normalized yet.", period=_Q, tests=()),
    "throughput": _m("throughput", "inconsistent", (_OVERLAY,), "Refining throughput units vary.", period=_Q, tests=()),
    "subscribers": _m("subscribers", "inconsistent", (_OVERLAY,), "Subscriber vs customer labels and units vary.", period=_Q, tests=()),
    "arpu": _m("arpu", "inconsistent", (_OVERLAY,), "ARPU reported as ₹ vs $/month without a canonical unit.", period=_Q, tests=()),
    "churn": _m("churn", "obtainable", (_OVERLAY, _TEXT), "Churn % when explicitly labeled.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "spectrum": _m("spectrum", "not_reliably_available", (), "Spectrum liability is a note, not a results row.", period=_A, tests=()),
    "capacity": _m("capacity", "inconsistent", (_OVERLAY,), "MW vs MT vs rooms share the capacity label.", period=_Q, tests=()),
    "plf": _m("plf", "obtainable", (_OVERLAY, _TEXT), "Plant load factor %.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "rab": _m("rab", "not_reliably_available", (), "Regulated asset base is a tariff filing, not a results table.", period=_A, tests=()),
    "transmission": _m("transmission", "not_reliably_available", (), "Transmission growth is not a standard KPI row.", period=_Q, tests=()),
    "presales": _m("presales", "obtainable", (_OVERLAY,), "Pre-sales / bookings when labeled.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "collections": _m("collections", "inconsistent", (_OVERLAY,), "Collections collides with trade collections and tax collections.", period=_Q, tests=()),
    "land_bank": _m("land_bank", "not_reliably_available", (), "Land bank is a note, not a dated series.", period=_A, tests=()),
    "ask": _m("ask", "inconsistent", (_OVERLAY,), "ASK vs other aviation units are not normalized yet.", period=_Q, tests=()),
    "load_factor": _m("load_factor", "obtainable", (_OVERLAY, _TEXT), "Passenger load factor %.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "cask": _m("cask", "not_reliably_available", (), "CASK units are not canonical yet.", period=_Q, tests=()),
    "yield": _m("yield", "inconsistent", (_OVERLAY,), "Yield collides with NIM, dividend yield and passenger yield.", period=_Q, tests=()),
    "arr": _m("arr", "inconsistent", (_OVERLAY,), "ARR collides with hotel ARR and SaaS ARR.", period=_Q, tests=()),
    "revpar": _m("revpar", "obtainable", (_OVERLAY,), "RevPAR when labeled.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "ape": _m("ape", "obtainable", (_OVERLAY,), "APE / annualized premium when labeled.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "vnb_margin": _m("vnb_margin", "obtainable", (_OVERLAY, _TEXT), "VNB margin %.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "persistency": _m("persistency", "obtainable", (_OVERLAY, _TEXT), "13-month persistency %.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "solvency": _m("solvency", "inconsistent", (_OVERLAY,), "Solvency as % vs times is not normalized.", period=_Q, tests=()),
    "gwp": _m("gwp", "obtainable", (_OVERLAY,), "Gross written premium when labeled.", period=_Q, tests=("test_priority_extractors_reject_false_positives",)),
    "combined": _m("combined", "inconsistent", (_OVERLAY,), "Combined ratio often exceeds 100% and needs a dedicated bound path.", period=_Q, tests=()),
    "loss_ratio": _m("loss_ratio", "inconsistent", (_OVERLAY,), "Loss ratio labels vary (net vs gross).", period=_Q, tests=()),
    "active_clients": _m("active_clients", "inconsistent", (_OVERLAY,), "Active-client definitions differ by broker.", period=_Q, tests=()),
    "market_share": _m("market_share", "inconsistent", (_OVERLAY,), "Market share is rarely a comparable dated series.", period=_Q, tests=()),
    "net_flows": _m("net_flows", "not_reliably_available", (), "AMC net flows are presentation-only.", period=_Q, tests=()),
    "equity_mix": _m("equity_mix", "not_reliably_available", (), "Equity AUM mix is presentation-only.", period=_Q, tests=()),
    "capacity_util": _m("capacity_util", "inconsistent", (_OVERLAY,), "Capacity utilization collides with PLF and plant utilization.", period=_Q, tests=()),
}

_UNSPECIFIED = MetricImpl(
    kpi_id="",
    reliability="not_reliably_available",
    paths=(),
    definition="No validated acquisition path is registered for this metric.",
    period_policy="Not implemented.",
    false_positive_guard="Do not extract.",
    tests=(),
)


def get_impl(kpi_id: str) -> MetricImpl:
    found = METRIC_IMPL.get(str(kpi_id or "").strip())
    if found is None:
        return MetricImpl(
            kpi_id=str(kpi_id or ""),
            reliability=_UNSPECIFIED.reliability,
            paths=_UNSPECIFIED.paths,
            definition=_UNSPECIFIED.definition,
            period_policy=_UNSPECIFIED.period_policy,
            false_positive_guard=_UNSPECIFIED.false_positive_guard,
            tests=_UNSPECIFIED.tests,
        )
    return found


def reliability_label(value: str) -> str:
    return {
        "obtainable": "obtainable",
        "derivable": "derivable",
        "inconsistent": "inconsistent",
        "not_reliably_available": "not reliably available",
    }.get(str(value or ""), "not reliably available")
