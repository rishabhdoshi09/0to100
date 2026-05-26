"""
NSE Full Universe Provider.

Returns all actively traded NSE equity stocks from the best available source:
  Tier 1: Kite instruments cache (logs/instruments_cache.csv) — up to ~2000 stocks
  Tier 2: NSE website direct (bhavcopy / equity list) — ~1800+ stocks
  Tier 3: Local fallback CSV (data/nse_symbols_fallback.csv) — 320 stocks
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── Predefined index constants ─────────────────────────────────────────────────

NIFTY50: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "KOTAKBANK", "LT", "SBIN", "BHARTIARTL", "AXISBANK", "ASIANPAINT",
    "MARUTI", "NESTLEIND", "ULTRACEMCO", "BAJFINANCE", "WIPRO",
    "HCLTECH", "TECHM", "SUNPHARMA", "TITAN", "ADANIENT", "ADANIPORTS",
    "BAJAJFINSV", "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "GRASIM", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "INDUSINDBK", "ITC", "JSWSTEEL", "M&M", "NTPC",
    "ONGC", "POWERGRID", "SBILIFE", "SHRIRAMFIN", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TRENT", "VEDL", "WIPRO",
]

NIFTY100: List[str] = NIFTY50 + [
    "ABB", "ADANIGREEN", "ADANITRANS", "AMBUJACEM", "AUROPHARMA",
    "BAJAJ-AUTO", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BEL",
    "BERGEPAINT", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL",
    "DMART", "GAIL", "GODREJCP", "HAVELLS", "ICICIPRULI",
    "INDIGO", "IOC", "JUBLFOOD", "LICI", "LUPIN",
    "MARICO", "MCDOWELL-N", "MUTHOOTFIN", "NAUKRI", "OBEROIRLTY",
    "PAGEIND", "PIDILITIND", "PNB", "RECLTD", "SAIL",
    "SIEMENS", "SRF", "TORNTPHARM", "TVSMOTOR", "UBL",
    "UNIONBANK", "UPL", "VOLTAS", "WHIRLPOOL", "ZOMATO",
]

NIFTY500: List[str] = list(dict.fromkeys(NIFTY100 + [
    # FMCG / Consumer Staples
    "DABUR", "EMAMI", "GODREJIND", "HINDPETRO", "VBL",
    "RADICO", "JYOTHYLAB", "BAJAJCON", "GILLETTE", "PGHH",
    "VMART", "VSTIND", "ZYDUSWELL", "BRITANNIA", "COLPAL",
    # Auto & Auto Ancillary
    "ASHOKLEY", "BHARATFORG", "EXIDEIND", "MOTHERSON", "MRF",
    "APOLLOTYRE", "CEATLTD", "TIINDIA", "ENDURANCE", "SUNDRMFAST",
    "SUPRAJIT", "WABCO", "GABRIEL", "SUBROS", "MINDAIND",
    "MAHINDCIE", "SCHAEFFLER", "SKFINDIA", "TIMKEN", "FIEM",
    # IT / Software
    "COFORGE", "LTIM", "MPHASIS", "OFSS", "PERSISTENT",
    "KPITTECH", "CYIENT", "HEXAWARE", "LTTS",
    "TATAELXSI", "ZENSAR", "BIRLASOFT", "INTELLECT",
    "MASTEK", "SONATSOFTW", "HAPPSTMNDS", "ROUTE", "ECLERX",
    # Banking & Finance
    "FEDERALBNK", "IDFCFIRSTB", "KARURVYSYA", "RBLBANK", "CUB",
    "DCBBANK", "UJJIVANSFB", "EQUITASBNK",
    "AUBANK", "ESAFSFB", "JKBANK", "SOUTHBANK", "TMB",
    "CSBBANK", "KTKBANK",
    "MANAPPURAM", "AAVAS", "APTUS", "HOMEFIRST", "CREDITACC",
    "SPANDANA", "FUSION", "ARMANFIN", "SBFC", "UGROCAP",
    "MASFIN", "CHOLAFIN",
    # Pharma & Healthcare
    "ALKEM", "APOLLOHOSP", "GLENMARK", "IPCA", "NATCOPHARM",
    "ABBOTINDIA", "PFIZER", "SANOFI", "GLAXO",
    "LAURUSLABS", "GRANULES", "AJANTPHARM", "BIOCON", "CADILAHC",
    "ZYDUSLIFE", "SOLARA", "GLAND", "STRIDES",
    "POLYMED", "METROPOLIS", "THYROCARE",
    "FORTIS", "MAXHEALTH", "SYNGENE", "DIVISLAB",
    # Real Estate
    "DLF", "GODREJPROP", "PRESTIGE", "SOBHA", "BRIGADE",
    "MAHLIFE", "KOLTEPATIL", "PHOENIXLTD", "LODHA", "MACROTECH",
    "NESCO", "SUNTECK", "KEYSTONE",
    # Infrastructure & Construction
    "GMRAIRPORT", "IRB", "SADBHAV", "PNC", "KNRCON",
    "HGINFRA", "PSPPROJECT", "AHLUCONT", "CAPACITE", "GPPL",
    # Gas & Energy Distribution
    "ATGL", "MGL", "IGL", "GUJGASLTD", "GSPL", "AEGISLOG", "PETRONET",
    # Energy & Power
    "CESC", "JSPL", "TATAPOWER", "TORNTPOWER", "JSWENERGY",
    "NHPC", "SJVN",
    "SUZLON", "INOXWIND",
    "POWERMECH", "KEC", "KALPATPOWR", "PFC", "IRFC",
    "ADANIENSOL",
    # Consumer & Retail / Apparel
    "ABFRL", "BATAINDIA", "RELAXO", "METROBRAND", "CAMPUS",
    "SHOPERSTOP", "VEDANT",
    "MANYAVAR", "RAYMOND",
    "KPRMILL", "RUPA", "DOLLAR",
    # Manufacturing & Capital Goods
    "CUMMINSIND", "ESCORTS", "KFINTECH", "POLYCAB",
    "FINOLEX", "KEI", "APAR", "BHEL",
    "BEML", "BDL", "HAL", "GRINDWELL", "CARBORUNIV",
    "ASTRAL", "PRINCEPIPE", "SUPREMEIND", "NILKAMAL",
    # Specialty Chemicals
    "ATUL", "NOCIL", "NAVINFLUOR", "FLUOROCHEM", "DEEPAKNTR",
    "TATACHEM", "GNFC", "GSFC", "COROMANDEL", "CHAMBAL",
    "CLEAN", "FINEORG", "VINATI", "ROSSARI", "AARTI",
    "GALAXYSURF", "SUDARSCHEM", "DCMSHRIRAM",
    # Cement
    "SHREECEM", "RAMCOCEM", "JKCEMENT", "HEIDELBERG", "JKLAKSHMI",
    "BIRLACORPN", "NUVOCO", "SANGHIIND",
    # Metals & Mining
    "HINDZINC", "HINDCOPPER", "MOIL", "NMDC", "NATIONALUM",
    "RATNAMANI", "WELSPUNIND", "SARDA", "GPIL",
    "JINDALPOLY", "JINDALSAW", "MAHSEAMLES",
    # Telecom & Media
    "TATACOMM", "HFCL", "VINDHYATEL",
    "RAILTEL", "PVRINOX", "INOXLEISUR",
    "SAREGAMA", "NAZARA",
    # Logistics & Transport
    "BLUEDART", "DELHIVERY", "GATI", "TCI",
    "CONCOR", "GATEWAY", "ALLCARGO", "VRLLOG",
    # Hospitality
    "INDHOTEL", "TAJGVK", "LEMONTREE", "CHALET", "EIH",
    "MAHINDHOLIDAY",
    # Ceramics & Building Materials
    "ORIENTBELL", "KAJARIACER", "SOMANYCER",
    # Agri & Fertilizers
    "PIIND", "ASTEC", "DHANUKA", "RALLIS",
    "BAYER", "SHARDACROP", "KSCL", "INSECTICID",
    # Insurance
    "ICICIGI", "NIACL", "STARHEALTH", "GODIGIT", "POLICYBZR",
    # Digital / New Economy
    "PAYTM", "ZOMATO", "NYKAA", "EASEMYTRIP", "IXIGO",
    "RATEGAIN", "CARTRADE", "MAPMYINDIA",
    # Diversified / Conglomerates
    "RPGLIFE", "WOCKHARDT", "ALEMBICPHARM",
    "JBCHEPHARM", "NATCOPHAR", "SEQUENT",
    # Capital Markets
    "CDSL", "CAMS", "MCX", "BSEINDIA", "ANGELONE",
    "MOTILALOFS", "IIFL", "5PAISA", "GEOJITFSL",
    # Staffing & Services
    "QUESS", "TEAMLEASE", "SIS",
    # Paints
    "INDIGOPNTS", "AKZOINDIA", "KANSAINER",
    # Engineering
    "ELGIEQUIP", "THERMAX", "TRIVENI", "GRINDMASTER",
    "ABB", "CUMMINSIND", "VOLTAMPERE",
    # Wires & Cables
    "POLYCAB", "KEI", "FINOLEX", "APAR", "RR",
    # White Goods / Consumer Durables
    "VOLTAS", "BLUESTARCO", "AMBER", "DIXON", "PG",
    "ORIENTELEC", "BAJAJELEC", "CROMPTON", "VGUARD",
    # Tyres
    "MRF", "APOLLOTYRE", "CEATLTD", "BALKRISIND", "TVSSRICHAK",
    # Paper & Packaging
    "TNPL", "WESTERNINDIA", "TAMILNADUPAP", "JKPAPER", "CENTURYPLY",
    "GREENPANEL", "AICA",
    # Footwear
    "BATAINDIA", "RELAXO", "METROBRAND", "CAMPUS", "MIRZA",
    # Gems & Jewellery
    "TITAN", "RAJESHEXPO", "PCJEWELLER",
    # Railways
    "RVNL", "IRCON", "RITES", "IRCTC", "RAILTEL",
    "HBLPOWER", "TITAGARH",
    # Defence
    "HAL", "BDL", "BEML", "MIDHANI", "PARAS",
    # Textiles
    "TRIDENT", "RAYMOND", "VARDHMAN", "ALOKIND", "HIMATSEIDE",
    "WELSPUNIND", "GRASIM",
    # Housing Finance
    "HDFC", "LICHSGFIN", "CANFINHOME", "GRUH", "REPCO",
    "PNBHOUSING", "APTUS", "HOMEFIRST", "AAVAS",
    # NBFC
    "BAJFINANCE", "BAJAJFINSV", "MANAPPURAM", "MUTHOOTFIN",
    "CHOLAFIN", "SHRIRAMFIN", "MAHINDCIE", "SUNDARMFIN",
    "M&MFIN", "SRTRANSFIN",
    # Microfinance
    "CREDITACC", "SPANDANA", "FUSION", "ARMANFIN", "SBFC",
    # Miscellaneous
    "TANLA", "NETWEB", "ONMOBILE", "XELPMOC",
]))

# ── Internal helpers ───────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent.parent  # repo root


def _is_valid_symbol(sym: str) -> bool:
    """
    Keep only clean NSE equity symbols — regular stocks that trade on yfinance.
    Rejects bonds/NCDs, suspended, SME, rights, settlement securities.
    Valid examples: RELIANCE, BAJAJ-AUTO, MCDOWELL-N, M&M, L&TFH
    """
    if not sym:
        return False
    # Must start with a letter
    if not sym[0].isalpha():
        return False
    # Reject index/market symbols with spaces (e.g. "INDIA VIX")
    if " " in sym:
        return False
    # Max 15 chars (MCDOWELL-N = 10, BAJAJ-AUTO = 10, BAJAJFINSV = 10)
    if len(sym) > 15:
        return False
    # Reject specific bad suffixes that indicate non-equity instruments
    # -ST=suspended, -BZ=trade-to-trade, -SF=settlement, -IT/-IL=institutional
    # -BE/-BL=rights/book, -SM=SME, -N0..-N9=bonds/NCDs, -NL=non-listed
    _BAD_SUFFIXES = (
        "-ST", "-BZ", "-SF", "-IT", "-IL", "-BE", "-BL", "-SM",
        "-N0", "-N1", "-N2", "-N3", "-N4", "-N5", "-N6", "-N7", "-N8", "-N9",
        "-NL", "-NI", "-NA", "-NB", "-NC", "-ND", "-NE", "-NF", "-NG",
    )
    for suf in _BAD_SUFFIXES:
        if sym.endswith(suf):
            return False
    # Reject if symbol ends in a digit after a hyphen (bond series like AAFS29A-N0)
    if "-" in sym:
        after_hyphen = sym.split("-")[-1]
        # Keep BAJAJ-AUTO, MCDOWELL-N; reject -N0, -N3, -BZ, -ST etc.
        if after_hyphen and after_hyphen[-1].isdigit():
            return False
        # Also reject 2-char suffixes that are all letters but in bad set
        if len(after_hyphen) == 2 and after_hyphen.isalpha() and after_hyphen not in ("AU", "MO"):
            return False
    # Reject ETF iNAV tickers (indicative NAV — not tradeable stocks)
    if sym.endswith("INAV"):
        return False
    # Reject symbols longer than 10 chars unless they're known valid patterns
    # (most real equity symbols are ≤10 chars; longer ones are usually ETF/index)
    if len(sym) > 10 and not any(c in sym for c in ("-", "&")):
        return False
    return True


def _dedupe_sorted(symbols: List[str]) -> List[str]:
    return sorted(set(s.strip().upper() for s in symbols if _is_valid_symbol(s.strip().upper())))


# ── Tier 1: Kite instruments cache ────────────────────────────────────────────

def _load_from_kite_cache() -> tuple:
    cache_path = _BASE_DIR / "logs" / "instruments_cache.csv"
    if not cache_path.exists():
        return [], {}
    try:
        import pandas as pd
        df = pd.read_csv(cache_path, dtype=str, low_memory=False)
        required = {"tradingsymbol", "exchange", "instrument_type"}
        if not required.issubset(df.columns):
            return [], {}
        df = df[
            (df["exchange"].str.upper() == "NSE") &
            (df["instrument_type"].str.upper() == "EQ")
        ]
        symbols = df["tradingsymbol"].dropna().str.strip().str.upper().tolist()
        names: Dict[str, str] = {}
        if "name" in df.columns:
            for _, row in df.iterrows():
                sym = str(row["tradingsymbol"]).strip().upper()
                names[sym] = str(row.get("name", "")).strip()
        valid = [s for s in symbols if _is_valid_symbol(s)]
        logger.info("Tier 1 (Kite cache): loaded %d symbols", len(valid))
        return valid, names
    except Exception as exc:
        logger.warning("Tier 1 failed: %s", exc)
        return [], {}


# ── Tier 2: NSE equity list ───────────────────────────────────────────────────

def _load_from_nse_website() -> tuple:
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        import io
        import requests
        import pandas as pd
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), dtype=str)
        # Normalise column names
        df.columns = [c.strip().upper() for c in df.columns]
        if "SYMBOL" not in df.columns:
            return [], {}
        name_col = None
        for candidate in ("NAME OF COMPANY", "COMPANY NAME", "NAME"):
            if candidate in df.columns:
                name_col = candidate
                break
        symbols = df["SYMBOL"].dropna().str.strip().str.upper().tolist()
        names: Dict[str, str] = {}
        if name_col:
            for _, row in df.iterrows():
                sym = str(row["SYMBOL"]).strip().upper()
                names[sym] = str(row.get(name_col, "")).strip()
        valid = [s for s in symbols if _is_valid_symbol(s)]
        logger.info("Tier 2 (NSE website): loaded %d symbols", len(valid))
        return valid, names
    except Exception as exc:
        logger.warning("Tier 2 failed: %s", exc)
        return [], {}


# ── Tier 3: local fallback CSV ────────────────────────────────────────────────

def _load_from_fallback_csv() -> tuple:
    csv_path = _BASE_DIR / "data" / "nse_symbols_fallback.csv"
    if not csv_path.exists():
        logger.warning("Tier 3: fallback CSV not found at %s", csv_path)
        return [], {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = [c.strip().upper() for c in df.columns]
        if "SYMBOL" not in df.columns:
            return [], {}
        name_col = "NAME" if "NAME" in df.columns else None
        symbols = df["SYMBOL"].dropna().str.strip().str.upper().tolist()
        names: Dict[str, str] = {}
        if name_col:
            for _, row in df.iterrows():
                sym = str(row["SYMBOL"]).strip().upper()
                names[sym] = str(row.get(name_col, "")).strip()
        valid = [s for s in symbols if _is_valid_symbol(s)]
        logger.info("Tier 3 (fallback CSV): loaded %d symbols", len(valid))
        return valid, names
    except Exception as exc:
        logger.warning("Tier 3 failed: %s", exc)
        return [], {}


# ── Process-lifetime cache ────────────────────────────────────────────────────

_cached_universe: List[str] = []
_cached_names: Dict[str, str] = {}
_universe_loaded: bool = False


def _load_universe() -> tuple:
    global _cached_universe, _cached_names, _universe_loaded
    if _universe_loaded:
        return _cached_universe, _cached_names

    symbols: List[str] = []
    names: Dict[str, str] = {}

    # Tier 1
    t1_syms, t1_names = _load_from_kite_cache()
    if t1_syms:
        symbols = t1_syms
        names.update(t1_names)

    # Tier 2
    if not symbols:
        t2_syms, t2_names = _load_from_nse_website()
        if t2_syms:
            symbols = t2_syms
            names.update(t2_names)

    # Tier 3
    if not symbols:
        t3_syms, t3_names = _load_from_fallback_csv()
        symbols = t3_syms
        names.update(t3_names)

    # Ultimate fallback — use the built-in NIFTY500 constant
    if not symbols:
        logger.warning("All tiers failed — using built-in NIFTY500 list")
        symbols = list(NIFTY500)

    _cached_universe = _dedupe_sorted(symbols)
    _cached_names = {k: v for k, v in names.items() if _is_valid_symbol(k)}
    _universe_loaded = True
    return _cached_universe, _cached_names


# ── Public API ────────────────────────────────────────────────────────────────

def get_nse_universe() -> List[str]:
    """
    Returns list of NSE equity symbols (no .NS suffix), deduplicated, sorted,
    filtered to EQ segment instruments only. Result is cached for process lifetime.
    """
    syms, _ = _load_universe()
    return syms


def get_nifty500_universe() -> List[str]:
    """Returns the NIFTY 500 list — fast default for scanning (~500 stocks)."""
    return sorted(NIFTY500)


def get_nse_universe_with_names() -> Dict[str, str]:
    """Returns {symbol: company_name} dict for all NSE equities."""
    _, names = _load_universe()
    return dict(names)


def get_nse_universe_by_sector() -> Dict[str, List[str]]:
    """
    Returns {sector_name: [symbols]} if sector data is available, else empty dict.
    Sector data is only present when loaded from Kite instruments cache
    (which includes a 'sector' or 'segment' column).
    """
    cache_path = _BASE_DIR / "logs" / "instruments_cache.csv"
    if not cache_path.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(cache_path, dtype=str, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"tradingsymbol", "exchange", "instrument_type"}
        if not required.issubset(df.columns):
            return {}
        df = df[
            (df["exchange"].str.upper() == "NSE") &
            (df["instrument_type"].str.upper() == "EQ")
        ]
        sector_col = None
        for candidate in ("sector", "industry", "segment"):
            if candidate in df.columns:
                sector_col = candidate
                break
        if sector_col is None:
            return {}
        result: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            sym = str(row["tradingsymbol"]).strip().upper()
            sec = str(row.get(sector_col, "UNKNOWN")).strip() or "UNKNOWN"
            if not _is_valid_symbol(sym):
                continue
            result.setdefault(sec, []).append(sym)
        # Sort symbols within each sector
        return {sec: sorted(set(syms)) for sec, syms in result.items()}
    except Exception as exc:
        logger.warning("get_nse_universe_by_sector failed: %s", exc)
        return {}
