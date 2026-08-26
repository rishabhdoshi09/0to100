"""Deterministic company → framework mapping.

Order:
  1. About-text / business description (most specific)
  2. Known-issuer map (when about-text is thin)
  3. Ticker suffix (BANK / BNK)
  4. NSE comment-group sector map
  5. Broader keyword blob
  6. Generic fallback

Never guess a bank. A more specific framework always beats a broader one.
"""

from __future__ import annotations

import re
from typing import Any

# symbol → (framework_id, sub_sector, business_model)
KNOWN_ISSUERS: dict[str, tuple[str, str, str]] = {
    "ICICIBANK": ("bank", "Private bank", "Private sector bank"),
    "HDFCBANK": ("bank", "Private bank", "Private sector bank"),
    "KOTAKBANK": ("bank", "Private bank", "Private sector bank"),
    "AXISBANK": ("bank", "Private bank", "Private sector bank"),
    "INDUSINDBK": ("bank", "Private bank", "Private sector bank"),
    "FEDERALBNK": ("bank", "Private bank", "Private sector bank"),
    "KARURVYSYA": ("bank", "Private bank", "Private sector bank"),
    "AUBANK": ("bank", "Small finance bank", "Small finance bank"),
    "EQUITASBNK": ("bank", "Small finance bank", "Small finance bank"),
    "UJJIVANSFB": ("bank", "Small finance bank", "Small finance bank"),
    "IDFCFIRSTB": ("bank", "Private bank", "Private sector bank"),
    "SBIN": ("bank", "PSU bank", "Public sector bank"),
    "BANKBARODA": ("bank", "PSU bank", "Public sector bank"),
    "PNB": ("bank", "PSU bank", "Public sector bank"),
    "CANBK": ("bank", "PSU bank", "Public sector bank"),
    "UNIONBANK": ("bank", "PSU bank", "Public sector bank"),
    "BAJFINANCE": ("nbfc", "Diversified NBFC", "Diversified NBFC"),
    "SHRIRAMFIN": ("nbfc", "Vehicle finance", "Vehicle-finance NBFC"),
    "CHOLAFIN": ("nbfc", "Vehicle finance", "Vehicle-finance NBFC"),
    "MUTHOOTFIN": ("nbfc_gold", "Gold loan", "Gold-loan NBFC"),
    "MANAPPURAM": ("nbfc_gold", "Gold loan", "Gold-loan NBFC"),
    "AAVAS": ("nbfc_housing", "Housing finance", "Housing finance company"),
    "HOMEFIRST": ("nbfc_housing", "Housing finance", "Housing finance company"),
    "LICHSGFIN": ("nbfc_housing", "Housing finance", "Housing finance company"),
    "CANFINHOME": ("nbfc_housing", "Housing finance", "Housing finance company"),
    "PNBHOUSING": ("nbfc_housing", "Housing finance", "Housing finance company"),
    "APTUS": ("nbfc_housing", "Housing finance", "Housing finance company"),
    "PFC": ("nbfc", "Infrastructure finance", "Infrastructure-finance NBFC"),
    "IRFC": ("nbfc", "Infrastructure finance", "Infrastructure-finance NBFC"),
    "RECLTD": ("nbfc", "Infrastructure finance", "Infrastructure-finance NBFC"),
    "INFY": ("it", "IT services", "IT outsourcing"),
    "TCS": ("it", "IT services", "IT outsourcing"),
    "WIPRO": ("it", "IT services", "IT outsourcing"),
    "HCLTECH": ("it", "IT services", "IT outsourcing"),
    "TECHM": ("it", "IT services", "IT outsourcing"),
    "LTIM": ("it", "IT services", "IT outsourcing"),
    "LTTS": ("it", "IT services", "IT outsourcing / ER&D"),
    "PERSISTENT": ("it", "IT services", "IT outsourcing"),
    "COFORGE": ("it", "IT services", "IT outsourcing"),
    "MPHASIS": ("it", "IT services", "IT outsourcing"),
    "OFSS": ("it", "IT services", "Financial software / IT services"),
    "NAUKRI": ("software_product", "Online classifieds", "Product / platform"),
    "SUNPHARMA": ("pharma", "Pharmaceuticals", "Pharma manufacturer"),
    "CIPLA": ("pharma", "Pharmaceuticals", "Pharma manufacturer"),
    "DRREDDY": ("pharma", "Pharmaceuticals", "Pharma manufacturer"),
    "DIVISLAB": ("pharma", "Pharmaceuticals", "Pharma manufacturer"),
    "AJANTPHARM": ("pharma", "Pharmaceuticals", "Pharma manufacturer"),
    "APOLLOHOSP": ("hospitals", "Hospitals", "Hospital chain"),
    "MAXHEALTH": ("hospitals", "Hospitals", "Hospital chain"),
    "FORTIS": ("hospitals", "Hospitals", "Hospital chain"),
    "METROPOLIS": ("diagnostics", "Diagnostics", "Diagnostics network"),
    "THYROCARE": ("diagnostics", "Diagnostics", "Diagnostics network"),
    "LT": ("capital_goods", "Capital goods / EPC", "Engineering / capital goods"),
    "SIEMENS": ("capital_goods", "Capital goods", "Engineering / capital goods"),
    "ABB": ("capital_goods", "Capital goods", "Engineering / capital goods"),
    "BHEL": ("capital_goods", "Capital goods", "Engineering / capital goods"),
    "KEI": ("industrials", "Wires & cables", "Industrial manufacturer"),
    "POLYCAB": ("industrials", "Wires & cables", "Industrial manufacturer"),
    "HAL": ("defence", "Defence", "Defence OEM"),
    "BEL": ("defence", "Defence", "Defence OEM"),
    "BDL": ("defence", "Defence", "Defence OEM"),
    "M&M": ("auto", "Automobile OEM", "PV / tractor / UV OEM"),
    "MARUTI": ("auto", "Automobile OEM", "Passenger-vehicle OEM"),
    "TATAMOTORS": ("auto", "Automobile OEM", "PV / CV OEM"),
    "BAJAJ-AUTO": ("auto", "Automobile OEM", "Two-wheeler OEM"),
    "HEROMOTOCO": ("auto", "Automobile OEM", "Two-wheeler OEM"),
    "EICHERMOT": ("auto", "Automobile OEM", "CV / two-wheeler OEM"),
    "TVSMOTOR": ("auto", "Automobile OEM", "Two-wheeler OEM"),
    "MOTHERSON": ("auto_ancillary", "Auto ancillary", "Component supplier"),
    "BOSCHLTD": ("auto_ancillary", "Auto ancillary", "Component supplier"),
    "BHARATFORG": ("auto_ancillary", "Auto ancillary", "Component supplier"),
    "HINDUNILVR": ("fmcg", "FMCG", "Consumer staples"),
    "ITC": ("fmcg", "FMCG", "Consumer staples"),
    "NESTLEIND": ("fmcg", "FMCG", "Consumer staples"),
    "BRITANNIA": ("fmcg", "FMCG", "Consumer staples"),
    "DABUR": ("fmcg", "FMCG", "Consumer staples"),
    "TRENT": ("retail", "Retail", "Retail chain"),
    "DMART": ("retail", "Retail", "Retail chain"),
    "SHOPERSTOP": ("retail", "Retail", "Retail chain"),
    "ABFRL": ("consumer_discretionary", "Apparel", "Brand / lifestyle"),
    "TITAN": ("consumer_discretionary", "Jewellery / watches", "Brand / lifestyle"),
    "DLF": ("realty", "Real estate", "Developer"),
    "GODREJPROP": ("realty", "Real estate", "Developer"),
    "OBEROIRLTY": ("realty", "Real estate", "Developer"),
    "TATASTEEL": ("metals", "Steel", "Metal producer"),
    "JSWSTEEL": ("metals", "Steel", "Metal producer"),
    "HINDALCO": ("metals", "Aluminium", "Metal producer"),
    "VEDL": ("metals", "Diversified metals", "Metal producer"),
    "COALINDIA": ("mining", "Mining", "Miner"),
    "NMDC": ("mining", "Mining", "Miner"),
    "HINDZINC": ("mining", "Mining", "Miner"),
    "BHARTIARTL": ("telecom", "Telecom", "Telecom operator"),
    "IDEA": ("telecom", "Telecom", "Telecom operator"),
    "NTPC": ("power_generation", "Power generation", "Generator"),
    "TATAPOWER": ("power_generation", "Power generation", "Generator"),
    "NHPC": ("power_generation", "Power generation", "Generator"),
    "POWERGRID": ("power_transmission", "Power transmission", "Transmission utility"),
    "HDFCLIFE": ("life_insurance", "Life insurance", "Life insurer"),
    "SBILIFE": ("life_insurance", "Life insurance", "Life insurer"),
    "ICICIPRULI": ("life_insurance", "Life insurance", "Life insurer"),
    "LICI": ("life_insurance", "Life insurance", "Life insurer"),
    "ICICIGI": ("general_insurance", "General insurance", "General insurer"),
    "STARHEALTH": ("general_insurance", "General insurance", "Health / general insurer"),
    "NIACL": ("general_insurance", "General insurance", "General insurer"),
    "CDSL": ("exchange", "Depository / FMI", "Market infrastructure"),
    "BSE": ("exchange", "Exchange", "Market infrastructure"),
    "MCX": ("exchange", "Exchange", "Market infrastructure"),
    "CAMS": ("exchange", "Registrar / FMI", "Market infrastructure"),
    "ANGELONE": ("broker", "Broking", "Broker"),
    "MOTILALOFS": ("broker", "Broking", "Broker"),
    "IIFL": ("broker", "Broking", "Broker"),
    "HDFCAMC": ("amc", "AMC", "Asset manager"),
    "NAM-INDIA": ("amc", "AMC", "Asset manager"),
    "INDIGO": ("airlines", "Airlines", "Airline"),
    "INDHOTEL": ("hotels", "Hotels", "Hotel operator"),
    "LEMONTREE": ("hotels", "Hotels", "Hotel operator"),
    "EIH": ("hotels", "Hotels", "Hotel operator"),
    "ULTRACEMCO": ("cement", "Cement", "Cement producer"),
    "SHREECEM": ("cement", "Cement", "Cement producer"),
    "AMBUJACEM": ("cement", "Cement", "Cement producer"),
    "ONGC": ("oil_gas", "Oil & gas", "Upstream"),
    "OIL": ("oil_gas", "Oil & gas", "Upstream"),
    "BPCL": ("refining", "Refining", "Refiner"),
    "IOC": ("refining", "Refining", "Refiner"),
    "HINDPETRO": ("refining", "Refining", "Refiner"),
    "PIDILITIND": ("specialty_chemicals", "Specialty chemicals", "Specialty chemical"),
    "SRF": ("specialty_chemicals", "Specialty chemicals", "Specialty chemical"),
    "DEEPAKNTR": ("specialty_chemicals", "Specialty chemicals", "Specialty chemical"),
    "PIIND": ("agri", "Agri-inputs", "Agri-inputs"),
    "COROMANDEL": ("agri", "Agri-inputs", "Agri-inputs"),
    "BLUEDART": ("logistics", "Logistics", "Logistics"),
    "DELHIVERY": ("logistics", "Logistics", "Logistics"),
    "CONCOR": ("logistics", "Logistics", "Logistics"),
    "ADANIPORTS": ("logistics", "Ports / logistics", "Logistics"),
    "ZEEL": ("media", "Media", "Media"),
    "PVRINOX": ("media", "Media", "Media"),
    "TRIDENT": ("textiles", "Textiles", "Textile manufacturer"),
    "PAGEIND": ("textiles", "Innerwear / textiles", "Textile manufacturer"),
}

# NSE comment-group names from data/nse_universe.py → default framework.
SECTOR_TO_FRAMEWORK: dict[str, str] = {
    "banking & finance": "nbfc",  # refined by about-text / known issuers / NPA rows
    "it / software": "it",
    "pharma & healthcare": "pharma",
    "manufacturing & capital goods": "capital_goods",
    "infrastructure & construction": "infrastructure",
    "metals & mining": "metals",
    "auto": "auto",
    "auto & auto ancillary": "auto",
    "fmcg": "fmcg",
    "consumer & retail / apparel": "consumer_discretionary",
    "real estate": "realty",
    "energy & power": "power_generation",
    "gas & energy distribution": "oil_gas",
    "specialty chemicals": "specialty_chemicals",
    "cement": "cement",
    "telecom & media": "telecom",
    "logistics & transport": "logistics",
    "hospitality": "hotels",
    "agri & fertilizers": "agri",
    "insurance": "life_insurance",
    "capital markets": "exchange",
    "defence": "defence",
    "textiles": "textiles",
    "housing finance": "nbfc_housing",
    "nbfc": "nbfc",
    "microfinance": "nbfc",
    "capital goods": "capital_goods",
    "engineering": "industrials",
    "wires & cables": "industrials",
    "white goods / consumer durables": "consumer_discretionary",
    "tyres": "auto_ancillary",
    "paints": "consumer_discretionary",
    "digital / new economy": "software_product",
    "staffing & services": "generic",
    "railways": "infrastructure",
    "ceramics & building materials": "industrials",
    "paper & packaging": "industrials",
    "gems & jewellery": "consumer_discretionary",
    "footwear": "consumer_discretionary",
    "diversified / conglomerates": "generic",
}

_NBFC_RE = re.compile(r"nbfc|non[\s-]?banking|housing finance|vehicle financ|finance company")
_BANK_RE = re.compile(
    r"\bsmall finance bank\b|\bprivate sector bank\b|\bpublic sector bank\b|"
    r"\bcommercial bank|\bbanks?\b"
)
_BANKING_RE = re.compile(r"(?<!non-)(?<!non )banking")


def _mentions_nbfc(about: str) -> bool:
    return bool(_NBFC_RE.search(about))


def _mentions_bank(about: str) -> bool:
    if _mentions_nbfc(about) and not _BANK_RE.search(about):
        return False
    return bool(_BANK_RE.search(about) or _BANKING_RE.search(about))


def _hit(blob: str, *needles: str) -> bool:
    return any(n and n in blob for n in needles)


def match_about(about: str) -> tuple[str, str, str, str] | None:
    """Most-specific business description wins. Returns None if about is too thin."""
    text = str(about or "").strip().lower()
    if len(text) < 8:
        return None

    if _hit(text, "gold loan", "gold-loan", "loan against gold"):
        return "nbfc_gold", "Gold loan", "Gold-loan NBFC", "About-text identifies a gold-loan NBFC."
    if _hit(text, "housing finance", "home loan", "housing loan", "affordable housing finance"):
        return "nbfc_housing", "Housing finance", "Housing finance company", "About-text identifies a housing-finance company."
    if re.search(r"\bhospitals?\b", text) and "hospitality" not in text:
        return "hospitals", "Hospitals", "Hospital chain", "About-text identifies a hospital operator."
    if _hit(text, "hospital chain", "multi-speciality hospital", "multi-specialty hospital",
            "hospital operator", "operates hospitals", "hospital network"):
        return "hospitals", "Hospitals", "Hospital chain", "About-text identifies a hospital operator."
    if _hit(text, "diagnostic", "pathology", "clinical lab", "radiology chain"):
        return "diagnostics", "Diagnostics", "Diagnostics network", "About-text identifies a diagnostics company."
    if _hit(text, "life insurance", "life insurer", "life assurance"):
        return "life_insurance", "Life insurance", "Life insurer", "About-text identifies a life insurer."
    if _hit(text, "general insurance", "non-life insurance", "non life insurance",
            "health insurance", "general insurer"):
        return "general_insurance", "General insurance", "General insurer", "About-text identifies a general insurer."
    if _hit(text, "stock exchange", "commodity exchange", "depository", "clearing corporation",
            "depository participant infrastructure"):
        return "exchange", "Exchange / FMI", "Market infrastructure", "About-text identifies an exchange / FMI."
    if _hit(text, "stock broker", "discount broker", "broking", "share broker"):
        return "broker", "Broking", "Broker", "About-text identifies a broker."
    if _hit(text, "asset management", "mutual fund", "amc "):
        return "amc", "AMC", "Asset manager", "About-text identifies an AMC."
    if _hit(text, "airline", "airlines", "passenger airline", "aviation"):
        if not _hit(text, "airport", "defence", "aerospace component"):
            return "airlines", "Airlines", "Airline", "About-text identifies an airline."
    if re.search(r"\bhotels?\b", text) or ("hospitality" in text and not re.search(r"\bhospital\b", text)):
        return "hotels", "Hotels", "Hotel operator", "About-text identifies a hotel operator."
    if _hit(text, "defence", "defense", "ordnance", "aerospace and defence", "aerospace & defence"):
        return "defence", "Defence", "Defence OEM / supplier", "About-text identifies a defence business."
    if _hit(text, "cement manufacturer", "cement producer", "cement company", "cement and clinker"):
        return "cement", "Cement", "Cement producer", "About-text identifies a cement producer."
    if _hit(text, "coal mining", "iron ore mining", "miner ", "mining company", "mines and"):
        return "mining", "Mining", "Miner", "About-text identifies a miner."
    if _hit(text, "lsaw", "pipes", "pipe manufacturer", "welded pipe", "seamless pipe"):
        return "industrials", "Pipes / industrials", "Industrial manufacturer", "About-text identifies an industrial manufacturer (pipes/engineering), not a miner."
    if _hit(text, "cable", "wires and", "wire & cable", "power cable"):
        return "industrials", "Wires & cables", "Industrial manufacturer", "About-text identifies a cable/industrial manufacturer."
    if _hit(text, "power transmission", "transmission utility", "interstate transmission"):
        return "power_transmission", "Power transmission", "Transmission / distribution", "About-text identifies a transmission utility."
    if _hit(text, "power generation", "thermal power", "hydro power", "electricity generation"):
        return "power_generation", "Power generation", "Generator", "About-text identifies a power generator."
    if _hit(text, "refiner", "refining", "gross refining margin"):
        return "refining", "Refining", "Refiner", "About-text identifies a refiner."
    if _hit(text, "upstream oil", "oil exploration", "oil and gas exploration", "crude production"):
        return "oil_gas", "Oil & gas", "Upstream", "About-text identifies upstream oil & gas."
    if _hit(text, "telecom operator", "wireless", "mobile subscriber", "cellular"):
        return "telecom", "Telecom", "Telecom operator", "About-text identifies a telecom operator."
    if _hit(text, "real estate developer", "residential project", "rera", "pre-sales"):
        return "realty", "Real estate", "Developer", "About-text identifies a real-estate developer."
    if _hit(text, "epc", "infrastructure contractor", "construction contractor"):
        return "infrastructure", "Infrastructure", "EPC / infra", "About-text identifies an infra / EPC contractor."
    if _hit(text, "auto ancillary", "auto component", "oem supplier", "tyre"):
        return "auto_ancillary", "Auto ancillary", "Component supplier", "About-text identifies an auto ancillary."
    if _hit(text, "passenger vehicle", "two wheeler", "two-wheeler", "commercial vehicle",
            "tractor", "automobile manufacturer", "vehicle manufacturer"):
        return "auto", "Automobile OEM", "OEM", "About-text identifies an auto OEM."
    if _hit(text, "supermarket", "hypermarket", "retail chain", "same store", "department store"):
        return "retail", "Retail", "Retail chain", "About-text identifies a retailer."
    if _hit(text, "fmcg", "packaged food", "personal care", "consumer staple", "household product"):
        return "fmcg", "FMCG", "Consumer staples", "About-text identifies an FMCG company."
    if _hit(text, "saas", "software product", "product company", "subscription software") and not _hit(
        text, "it services", "bpo", "outsourcing"
    ):
        return "software_product", "Software product", "Product / SaaS", "About-text identifies a software-product company."
    if _hit(text, "it services", "information technology", "bpo", "outsourcing", "software"):
        return "it", "IT services", "IT outsourcing", "About-text identifies IT services / software."
    if _hit(text, "pharma", "pharmaceutical", "formulation", "api manufacturer", "drug"):
        return "pharma", "Pharmaceuticals", "Pharma manufacturer", "About-text identifies a pharma manufacturer."
    if _hit(text, "specialty chemical"):
        return "specialty_chemicals", "Specialty chemicals", "Specialty chemical", "About-text identifies specialty chemicals."
    if _mentions_bank(text):
        model = "Public sector bank" if _hit(text, "public sector") else (
            "Small finance bank" if _hit(text, "small finance") else "Bank"
        )
        if _hit(text, "private sector"):
            model = "Private sector bank"
        return "bank", "Banking", model, "Company description classifies this as a bank."
    if _mentions_nbfc(text):
        return "nbfc", "NBFC", "Diversified NBFC", "Finance company with lending metrics — NBFC framework."
    if _hit(text, "capital goods", "engineering company", "heavy electrical"):
        return "capital_goods", "Capital goods", "Engineering / capital goods", "About-text identifies capital goods."
    if _hit(text, "industrial", "order book"):
        return "industrials", "Industrials", "Industrial manufacturer", "About-text identifies industrials."
    return None


def match_sector_map(sector_name: str) -> str | None:
    from product.due_diligence.series import normalize_label

    return SECTOR_TO_FRAMEWORK.get(normalize_label(sector_name))


def known_issuer(symbol: str) -> tuple[str, str, str] | None:
    return KNOWN_ISSUERS.get(str(symbol or "").upper().strip())


def classify_business(
    symbol: str,
    *,
    sector: str = "",
    about: str = "",
    has_npa: bool = False,
) -> dict[str, Any]:
    """Pick the most specific valid framework. Unknown stays generic."""
    sector_name = str(sector or "").strip()
    about_l = str(about or "").lower()
    ticker = str(symbol or "").upper().strip()

    about_hit = match_about(about)
    if about_hit:
        framework, sub, model, reason = about_hit
        return _result(ticker, sector_name, about, framework, reason, sub, model)

    known = known_issuer(ticker)
    if known:
        framework, sub, model = known
        return _result(
            ticker, sector_name, about, framework,
            "Known-issuer map for this NSE name.",
            sub, model,
        )

    if ticker.endswith(("BANK", "BNK")) and not _mentions_nbfc(about_l):
        return _result(
            ticker, sector_name, about, "bank",
            "Ticker suffix identifies a bank.",
            "Banking", "Bank",
        )

    mapped = match_sector_map(sector_name)
    if mapped == "nbfc" or (has_npa and not _mentions_bank(about_l)):
        if has_npa and mapped not in {None, "nbfc", "bank", "nbfc_gold", "nbfc_housing"}:
            mapped = "nbfc"
        if mapped in {"nbfc", "nbfc_gold", "nbfc_housing"} or has_npa:
            framework = mapped if mapped in {"nbfc", "nbfc_gold", "nbfc_housing"} else "nbfc"
            return _result(
                ticker, sector_name, about, framework,
                "Finance company with lending metrics — NBFC framework.",
                "NBFC", "Diversified NBFC",
            )

    if mapped:
        from product.due_diligence.sector_frameworks.catalog import FRAMEWORKS

        fw = FRAMEWORKS.get(mapped) or FRAMEWORKS["generic"]
        return _result(
            ticker, sector_name, about, mapped,
            f"Sector map '{sector_name}'.",
            fw.default_sub_sector, fw.default_business_model,
        )

    blob = f"{about_l} {sector_name.lower()}"
    if _hit(blob, "steel", "aluminium", "copper", "mining"):
        return _result(ticker, sector_name, about, "metals", "Metals / mining keyword.", "Metals", "Metal producer")
    if sector_name:
        return _result(
            ticker, sector_name, about, "generic",
            f"Sector '{sector_name}' has no dedicated framework yet — generic checks only.",
            "", "",
        )
    return _result(
        ticker, sector_name, about, "generic",
        "No sector-specific table match; using the generic quality framework.",
        "", "",
    )


def _result(
    symbol: str,
    sector_name: str,
    about: str,
    framework: str,
    reason: str,
    sub_sector: str,
    business_model: str,
) -> dict[str, Any]:
    about_text = str(about or "").strip()
    return {
        "symbol": symbol,
        "sector": sector_name or "Unclassified",
        "industry": sector_name or "Unclassified",
        "framework_id": framework,
        "framework_reason": reason,
        "sub_sector": sub_sector or "",
        "business_model": business_model or "Data unavailable",
        "about": about_text,
        "revenue_drivers": "Data unavailable — no segment table on file.",
    }


def framework_for_peer_name(name: str) -> str | None:
    """Best-effort map from a peer-table display name onto a known issuer."""
    blob = str(name or "").upper()
    if not blob.strip():
        return None
    compact = re.sub(r"[^A-Z0-9]", "", blob)
    for symbol, spec in KNOWN_ISSUERS.items():
        token = re.sub(r"[^A-Z0-9]", "", symbol)
        if token and token in compact:
            return spec[0]
        if symbol.replace("&", "AND") in blob.replace("&", "AND"):
            return spec[0]
    return None
