"""
NSE Full Universe Provider.

Returns all actively traded NSE equity stocks by *unioning* available sources:
  Tier 1: Kite instruments cache (logs/instruments_cache.csv) — ~2000 EQ stocks
  Tier 2: NSE website EQUITY_L.csv — when local coverage is thin (<1500)
  Tier 3: Local fallback CSV (data/nse_symbols_fallback.csv) — 320 stocks
  Plus:  local bhavcopy store symbols (official EQ history already on disk)
"""
from __future__ import annotations

import json
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
    "ABB", "ADANIGREEN", "ADANIENSOL", "AMBUJACEM", "AUROPHARMA",
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
    "DABUR", "EMAMILTD", "GODREJIND", "HINDPETRO", "VBL",
    "RADICO", "JYOTHYLAB", "BAJAJCON", "GILLETTE", "PGHH",
    "VMART", "VSTIND", "ZYDUSWELL", "BRITANNIA", "COLPAL",
    # Auto & Auto Ancillary
    "ASHOKLEY", "BHARATFORG", "EXIDEIND", "MOTHERSON", "MRF",
    "APOLLOTYRE", "CEATLTD", "TIINDIA", "ENDURANCE", "SUNDRMFAST",
    "SUPRAJIT", "WABCO", "GABRIEL", "SUBROS", "MINDAIND",
    "MAHINDCIE", "SCHAEFFLER", "SKFINDIA", "TIMKEN", "FIEMIND",
    # IT / Software
    "COFORGE", "LTIM", "MPHASIS", "OFSS", "PERSISTENT",
    "KPITTECH", "CYIENT", "LTTS",
    "TATAELXSI", "ZENSAR", "BSOFT", "INTELLECT",
    "MASTEK", "SONATSOFTW", "HAPPSTMNDS", "ROUTE", "ECLERX",
    # Banking & Finance
    "FEDERALBNK", "IDFCFIRSTB", "KARURVYSYA", "RBLBANK", "CUB",
    "DCBBANK", "UJJIVANSFB", "EQUITASBNK",
    "AUBANK", "ESAFSFB", "J&KBANK", "SOUTHBANK", "TMB",
    "CSBBANK", "KTKBANK",
    "MANAPPURAM", "AAVAS", "APTUS", "HOMEFIRST", "CREDITACC",
    "SPANDANA", "FUSION", "ARMANFIN", "SBFC", "UGROCAP",
    "MASFIN", "CHOLAFIN",
    # Pharma & Healthcare
    "ALKEM", "APOLLOHOSP", "GLENMARK", "IPCALAB", "NATCOPHARM",
    "ABBOTINDIA", "PFIZER", "SANOFI", "GLAXO",
    "LAURUSLABS", "GRANULES", "AJANTPHARM", "BIOCON", "ZYDUSLIFE",
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
    "CESC", "JINDALSTEL", "TATAPOWER", "TORNTPOWER", "JSWENERGY",
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
    "FINCABLES", "KEI", "APARINDS", "BHEL",
    "BEML", "BDL", "HAL", "GRINDWELL", "CARBORUNIV",
    "ASTRAL", "PRINCEPIPE", "SUPREMEIND", "NILKAMAL",
    # Specialty Chemicals
    "ATUL", "NOCIL", "NAVINFLUOR", "FLUOROCHEM", "DEEPAKNTR",
    "TATACHEM", "GNFC", "GSFC", "COROMANDEL", "CHAMBLFERT",
    "CLEAN", "FINEORG", "VINATI", "ROSSARI", "AARTIIND",
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
    "RAILTEL", "PVRINOX", "SAREGAMA", "NAZARA",
    # Logistics & Transport
    "BLUEDART", "DELHIVERY", "ACLGATI", "TCI",
    "CONCOR", "GATEWAY", "ALLCARGO", "VRLLOG",
    # Hospitality
    "INDHOTEL", "TAJGVK", "LEMONTREE", "CHALET", "EIH",
    "MAHINDHOLIDAY",
    # Ceramics & Building Materials
    "ORIENTBELL", "KAJARIACER", "SOMANYCER",
    # Agri & Fertilizers
    "PIIND", "ASTEC", "DHANUKA", "RALLIS",
    "BAYERCROP", "SHARDACROP", "KSCL", "INSECTICID",
    # Insurance
    "ICICIGI", "NIACL", "STARHEALTH", "GODIGIT", "POLICYBZR",
    # Digital / New Economy
    "PAYTM", "ZOMATO", "NYKAA", "EASEMYTRIP", "IXIGO",
    "RATEGAIN", "CARTRADE", "MAPMYINDIA",
    # Diversified / Conglomerates
    "RPGLIFE", "WOCKHARDT", "APLLTD",
    "JBCHEPHARM", "NATCOPHAR", "SEQUENT",
    # Capital Markets
    "CDSL", "CAMS", "MCX", "BSE", "ANGELONE",
    "MOTILALOFS", "IIFL", "5PAISA", "GEOJITFSL",
    # Staffing & Services
    "QUESS", "TEAMLEASE", "SIS",
    # Paints
    "INDIGOPNTS", "AKZOINDIA", "KANSAINER",
    # Engineering
    "ELGIEQUIP", "THERMAX", "TRIVENI", "GRINDWELL",
    "ABB", "CUMMINSIND", "VOLTAMPERE",
    # Wires & Cables
    "POLYCAB", "KEI", "FINCABLES", "APARINDS", "RR",
    # White Goods / Consumer Durables
    "VOLTAS", "BLUESTARCO", "AMBER", "DIXON", "PG",
    "ORIENTELEC", "BAJAJELEC", "CROMPTON", "VGUARD",
    # Tyres
    "MRF", "APOLLOTYRE", "CEATLTD", "BALKRISIND", "TVSSRICHAK",
    # Paper & Packaging
    "TNPL", "WESTERNINDIA", "TAMILNADUPAP", "JKPAPER", "CENTURYPLY",
    "GREENPANEL", "AIAENG",
    # Footwear
    "BATAINDIA", "RELAXO", "METROBRAND", "CAMPUS", "MIRZA",
    # Gems & Jewellery
    "TITAN", "RAJESHEXPO", "PCJEWELLER",
    # Railways
    "RVNL", "IRCON", "RITES", "IRCTC", "RAILTEL",
    "HBLENGINE", "TITAGARH",
    # Defence
    "HAL", "BDL", "BEML", "MIDHANI", "PARAS",
    # Textiles
    "TRIDENT", "RAYMOND", "VARDHMAN", "ALOKINDS", "HIMATSEIDE",
    "WELSPUNIND", "GRASIM",
    # Housing Finance
    "LICHSGFIN", "CANFINHOME", "REPCO",
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
    # Reject specific bad suffixes that indicate non-equity / non-holdable junk.
    # -BE is KEEP: NSE Trade-for-Trade / book-entry series — retail often holds these
    # (e.g. GAUDIUMIVF-BE). -BZ stays out of the research scan universe (GSM/T2T junk
    # filter) but holdings import still accepts the raw broker symbol separately.
    # -ST=suspended, -SF=settlement, -IT/-IL=institutional, -BL=rights/book,
    # -SM=SME, -N0..-N9=bonds/NCDs, -NL=non-listed
    _BAD_SUFFIXES = (
        "-ST", "-BZ", "-SF", "-IT", "-IL", "-BL", "-SM",
        "-N0", "-N1", "-N2", "-N3", "-N4", "-N5", "-N6", "-N7", "-N8", "-N9",
        "-NL", "-NI", "-NA", "-NB", "-NC", "-ND", "-NE", "-NF", "-NG",
    )
    for suf in _BAD_SUFFIXES:
        if sym.endswith(suf):
            return False
    # Reject if symbol ends in a digit after a hyphen (bond series like AAFS29A-N0)
    if "-" in sym:
        after_hyphen = sym.split("-")[-1]
        # Keep BAJAJ-AUTO, MCDOWELL-N, GAUDIUMIVF-BE; reject -N0, -N3, -BZ, -ST etc.
        if after_hyphen and after_hyphen[-1].isdigit():
            return False
        # Also reject 2-char suffixes that are all letters but in bad set
        if len(after_hyphen) == 2 and after_hyphen.isalpha() and after_hyphen not in ("AU", "MO", "BE"):
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


def _filter_to_instruments(symbols: List[str], token_map: dict) -> List[str]:
    """Drop symbols not in the Kite instrument token map — the AUTHORITY on
    what's actually tradeable/quotable on NSE. The pattern-based junk filter
    (_is_valid_symbol) can't catch a symbol that LOOKS clean but doesn't exist
    as a Kite instrument (e.g. AMIRCHAND — a stale NSE listing); such a symbol
    fails a Kite fetch + a Yahoo 404 every single scan cycle, forever.

    Two guards make this fail-SAFE (dropping a real stock only means it isn't
    scanned — never a wrong trade — but we still avoid it):
      • no-op if the map isn't substantially loaded (≤1000) — a failed/absent
        instrument load must never nuke the universe.
      • no-op if it would drop an implausibly large chunk (>15%) — that means
        the map is incomplete, not that 15% of NSE is junk."""
    if not token_map or len(token_map) <= 1000:
        return symbols
    kept = [s for s in symbols if s.upper() in token_map]
    dropped = len(symbols) - len(kept)
    if dropped == 0:
        return symbols
    if dropped > len(symbols) * 0.15:
        logger.warning("universe: instrument cross-check would drop %d/%d (>15%%) "
                       "— skipping (instrument map likely incomplete)",
                       dropped, len(symbols))
        return symbols
    logger.info("universe: dropped %d stale/junk symbols not in Kite instruments "
                "(e.g. AMIRCHAND) — no more per-cycle fetch misses", dropped)
    return kept


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

# Prefer network / secondary sources when local coverage is thinner than this.
_THIN_UNIVERSE = 1500


def reset_universe_cache() -> None:
    """Test helper — force the next get_nse_universe() call to reload tiers."""
    global _cached_universe, _cached_names, _universe_loaded
    _cached_universe = []
    _cached_names = {}
    _universe_loaded = False


def _merge_names(dest: Dict[str, str], src: Dict[str, str]) -> None:
    for key, value in src.items():
        sym = str(key or "").strip().upper()
        if not _is_valid_symbol(sym):
            continue
        label = str(value or "").strip()
        if label and not dest.get(sym):
            dest[sym] = label
        else:
            dest.setdefault(sym, label)


def _bhav_symbols() -> List[str]:
    """Official EQ names present in the local bhavcopy store (may be empty)."""
    try:
        from data.bhavcopy_store import store_symbols
        return [str(s).strip().upper() for s in (store_symbols() or []) if str(s).strip()]
    except Exception as exc:
        logger.debug("universe bhav merge skipped: %s", exc)
        return []


def _load_universe() -> tuple:
    """Build the live equity universe by *unioning* local sources.

    Older logic took the first non-empty tier only. That left the product on a
    320-name fallback CSV (or a partial Kite dump) even when bhav history already
    covered the full EQ market. Search, scan, and Stock Intelligence then looked
    like “shares are missing”.
    """
    global _cached_universe, _cached_names, _universe_loaded
    if _universe_loaded:
        return _cached_universe, _cached_names

    pool: set[str] = set()
    names: Dict[str, str] = {}

    t1_syms, t1_names = _load_from_kite_cache()
    pool.update(s.strip().upper() for s in t1_syms if s)
    _merge_names(names, t1_names)

    t3_syms, t3_names = _load_from_fallback_csv()
    pool.update(s.strip().upper() for s in t3_syms if s)
    _merge_names(names, t3_names)

    for sym in _bhav_symbols():
        if _is_valid_symbol(sym):
            pool.add(sym)
            names.setdefault(sym, "")

    # Pull the official NSE equity list when local coverage is thin or empty.
    if len(pool) < _THIN_UNIVERSE:
        t2_syms, t2_names = _load_from_nse_website()
        pool.update(s.strip().upper() for s in t2_syms if s)
        _merge_names(names, t2_names)

    if not pool:
        logger.warning("All tiers failed — using built-in NIFTY500 list")
        pool.update(str(s).strip().upper() for s in NIFTY500)

    symbols = _dedupe_sorted(list(pool))

    # Cross-check against the live Kite instrument map — removes stale NSE
    # listings that would otherwise miss on every fetch. Guarded + fail-safe.
    try:
        from data.instruments import InstrumentManager
        symbols = _filter_to_instruments(symbols, InstrumentManager()._token_map)
    except Exception as exc:
        logger.debug("universe instrument cross-check skipped: %s", exc)

    # Every universe member must appear in the names map so scan services that
    # iterate dict keys never see an empty universe after a constant fallback.
    for sym in symbols:
        names.setdefault(sym, "")

    _cached_universe = symbols
    _cached_names = {k: v for k, v in names.items() if k in set(symbols)}
    _universe_loaded = True
    logger.info(
        "universe_loaded count=%d named=%d",
        len(_cached_universe),
        sum(1 for v in _cached_names.values() if v),
    )
    return _cached_universe, _cached_names


# ── Public API ────────────────────────────────────────────────────────────────

def get_nse_universe() -> List[str]:
    """
    Returns list of NSE equity symbols (no .NS suffix), deduplicated, sorted,
    filtered to EQ segment instruments only. Result is cached for process lifetime.
    """
    syms, _ = _load_universe()
    return list(syms)


def get_nifty500_universe() -> List[str]:
    """Returns the NIFTY 500 list — fast default for scanning (~500 stocks)."""
    return sorted(NIFTY500)


def get_nse_universe_with_names() -> Dict[str, str]:
    """Returns {symbol: company_name} for *every* equity in get_nse_universe().

    Missing display names are empty strings — never drop symbols from the map.
    """
    syms, names = _load_universe()
    return {sym: str(names.get(sym) or "") for sym in syms}


def search_nse_symbols(query: str = "", *, limit: int = 50) -> List[Dict[str, str]]:
    """Prefix/substring search over the full live universe for UI autocomplete."""
    q = str(query or "").strip().upper()
    lim = max(1, min(int(limit or 50), 5000))
    names = get_nse_universe_with_names()
    rows: List[Dict[str, str]] = []
    if not q:
        for sym in list(names)[:lim]:
            rows.append({"symbol": sym, "name": names.get(sym) or ""})
        return rows
    # Exact + prefix first, then name / substring matches.
    exact = []
    prefix = []
    fuzzy = []
    for sym, name in names.items():
        label = (name or "").upper()
        if sym == q:
            exact.append({"symbol": sym, "name": name or ""})
        elif sym.startswith(q):
            prefix.append({"symbol": sym, "name": name or ""})
        elif q in sym or (label and q in label):
            fuzzy.append({"symbol": sym, "name": name or ""})
    for bucket in (exact, sorted(prefix, key=lambda r: r["symbol"]), sorted(fuzzy, key=lambda r: r["symbol"])):
        for row in bucket:
            rows.append(row)
            if len(rows) >= lim:
                return rows
    return rows


def point_in_time_universe(as_of, path=None) -> dict:
    """Survivorship-aware membership: symbols TRADEABLE on `as_of`.

    Reads ``logs/universe_history.json`` (or QT_UNIVERSE_HISTORY_FILE / ``path``).
    Supports both a bare JSON list and the versioned object form written by
    ``data.universe_history``.

    HONEST FALLBACK: when no history file exists we return today's survivors with
    ``survivorship_complete=False``. Callers MUST treat that as biased.
    """
    import pandas as pd
    from data.universe_history import history_path, ledger_status, _coerce_payload

    p = Path(path) if path else history_path()
    asof = pd.Timestamp(as_of)
    if not p.exists():
        return {
            "as_of": str(asof.date()),
            "symbols": get_nse_universe(),
            "survivorship_complete": False,
            "source": "",
            "research_grade": False,
            "note": (
                "no membership history on file — returning TODAY's survivors; "
                "results built on this are survivorship-biased. Supply "
                "logs/universe_history.json from NSE archives, or run the "
                "universe_history autonomy job to bootstrap from local bhav."
            ),
        }
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "as_of": str(asof.date()),
            "symbols": get_nse_universe(),
            "survivorship_complete": False,
            "source": "",
            "research_grade": False,
            "note": f"membership history unreadable ({exc}); using survivors.",
        }

    rows, meta = _coerce_payload(raw)
    status = ledger_status(p)
    live: List[str] = []
    for r in rows:
        try:
            sym = str(r["symbol"]).strip().upper()
            listed = pd.Timestamp(r["listed"])
            delisted = pd.Timestamp(r["delisted"]) if r.get("delisted") else None
            if listed <= asof and (delisted is None or delisted > asof):
                live.append(sym)
        except Exception:
            continue
    note = status.get("note") or ""
    return {
        "as_of": str(asof.date()),
        "symbols": sorted(set(live)),
        "survivorship_complete": True,
        "source": status.get("source") or str(meta.get("source") or "operator"),
        "research_grade": bool(status.get("research_grade")),
        "note": note,
        "rows": int(status.get("rows") or 0),
    }


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
