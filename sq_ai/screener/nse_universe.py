"""
Curated NSE equity universe, tagged by market-cap bucket.

yfinance has no fast bulk market-cap endpoint, so screening 2000+ listed
NSE stocks live would mean 2000+ individual API calls — too slow for an
interactive screener. Instead we ship a curated cross-section of liquid,
well-covered NSE names spanning Large/Mid/Small cap, which is what most
momentum/breakout scans actually trade.

bucket: "Large" (>₹50,000cr), "Mid" (₹10,000–50,000cr), "Small" (<₹10,000cr)
— approximate, as of compile time. Refresh periodically.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple


class UniverseStock(NamedTuple):
    symbol: str
    name: str
    bucket: str
    sector: str


NSE_UNIVERSE: List[UniverseStock] = [
    # ── Large cap ──────────────────────────────────────────────────────────
    UniverseStock("RELIANCE", "Reliance Industries", "Large", "Energy"),
    UniverseStock("TCS", "Tata Consultancy Services", "Large", "IT"),
    UniverseStock("HDFCBANK", "HDFC Bank", "Large", "Banking"),
    UniverseStock("ICICIBANK", "ICICI Bank", "Large", "Banking"),
    UniverseStock("INFY", "Infosys", "Large", "IT"),
    UniverseStock("SBIN", "State Bank of India", "Large", "Banking"),
    UniverseStock("BHARTIARTL", "Bharti Airtel", "Large", "Telecom"),
    UniverseStock("ITC", "ITC", "Large", "FMCG"),
    UniverseStock("HINDUNILVR", "Hindustan Unilever", "Large", "FMCG"),
    UniverseStock("LT", "Larsen & Toubro", "Large", "Infra"),
    UniverseStock("KOTAKBANK", "Kotak Mahindra Bank", "Large", "Banking"),
    UniverseStock("AXISBANK", "Axis Bank", "Large", "Banking"),
    UniverseStock("BAJFINANCE", "Bajaj Finance", "Large", "NBFC"),
    UniverseStock("MARUTI", "Maruti Suzuki", "Large", "Auto"),
    UniverseStock("ASIANPAINT", "Asian Paints", "Large", "Consumer"),
    UniverseStock("SUNPHARMA", "Sun Pharma", "Large", "Pharma"),
    UniverseStock("TITAN", "Titan Company", "Large", "Consumer"),
    UniverseStock("ULTRACEMCO", "UltraTech Cement", "Large", "Cement"),
    UniverseStock("WIPRO", "Wipro", "Large", "IT"),
    UniverseStock("NTPC", "NTPC", "Large", "Power"),
    UniverseStock("ADANIENT", "Adani Enterprises", "Large", "Diversified"),
    UniverseStock("ADANIPORTS", "Adani Ports", "Large", "Infra"),
    UniverseStock("POWERGRID", "Power Grid Corp", "Large", "Power"),
    UniverseStock("M&M", "Mahindra & Mahindra", "Large", "Auto"),
    UniverseStock("TATAMOTORS", "Tata Motors", "Large", "Auto"),
    UniverseStock("TATASTEEL", "Tata Steel", "Large", "Metals"),
    UniverseStock("JSWSTEEL", "JSW Steel", "Large", "Metals"),
    UniverseStock("HCLTECH", "HCL Technologies", "Large", "IT"),
    UniverseStock("BAJAJFINSV", "Bajaj Finserv", "Large", "NBFC"),
    UniverseStock("NESTLEIND", "Nestle India", "Large", "FMCG"),
    UniverseStock("ONGC", "Oil & Natural Gas Corp", "Large", "Energy"),
    UniverseStock("COALINDIA", "Coal India", "Large", "Mining"),
    UniverseStock("DRREDDY", "Dr Reddy's Labs", "Large", "Pharma"),
    UniverseStock("GRASIM", "Grasim Industries", "Large", "Cement"),
    UniverseStock("HINDALCO", "Hindalco Industries", "Large", "Metals"),

    # ── Mid cap ────────────────────────────────────────────────────────────
    UniverseStock("INDIGO", "InterGlobe Aviation", "Mid", "Aviation"),
    UniverseStock("PIIND", "PI Industries", "Mid", "Chemicals"),
    UniverseStock("PERSISTENT", "Persistent Systems", "Mid", "IT"),
    UniverseStock("COFORGE", "Coforge", "Mid", "IT"),
    UniverseStock("MPHASIS", "Mphasis", "Mid", "IT"),
    UniverseStock("LUPIN", "Lupin", "Mid", "Pharma"),
    UniverseStock("AUROPHARMA", "Aurobindo Pharma", "Mid", "Pharma"),
    UniverseStock("ALKEM", "Alkem Laboratories", "Mid", "Pharma"),
    UniverseStock("BIOCON", "Biocon", "Mid", "Pharma"),
    UniverseStock("FEDERALBNK", "Federal Bank", "Mid", "Banking"),
    UniverseStock("IDFCFIRSTB", "IDFC First Bank", "Mid", "Banking"),
    UniverseStock("BANKBARODA", "Bank of Baroda", "Mid", "Banking"),
    UniverseStock("PNB", "Punjab National Bank", "Mid", "Banking"),
    UniverseStock("CANBK", "Canara Bank", "Mid", "Banking"),
    UniverseStock("AUBANK", "AU Small Finance Bank", "Mid", "Banking"),
    UniverseStock("CHOLAFIN", "Cholamandalam Investment", "Mid", "NBFC"),
    UniverseStock("MUTHOOTFIN", "Muthoot Finance", "Mid", "NBFC"),
    UniverseStock("ESCORTS", "Escorts Kubota", "Mid", "Auto"),
    UniverseStock("ASHOKLEY", "Ashok Leyland", "Mid", "Auto"),
    UniverseStock("BHARATFORG", "Bharat Forge", "Mid", "Auto Ancillary"),
    UniverseStock("MOTHERSON", "Samvardhana Motherson", "Mid", "Auto Ancillary"),
    UniverseStock("TVSMOTOR", "TVS Motor", "Mid", "Auto"),
    UniverseStock("VOLTAS", "Voltas", "Mid", "Consumer Durables"),
    UniverseStock("HAVELLS", "Havells India", "Mid", "Consumer Durables"),
    UniverseStock("DIXON", "Dixon Technologies", "Mid", "Electronics"),
    UniverseStock("POLYCAB", "Polycab India", "Mid", "Electricals"),
    UniverseStock("TRENT", "Trent", "Mid", "Retail"),
    UniverseStock("PAGEIND", "Page Industries", "Mid", "Apparel"),
    UniverseStock("JUBLFOOD", "Jubilant FoodWorks", "Mid", "QSR"),
    UniverseStock("INDHOTEL", "Indian Hotels Co", "Mid", "Hospitality"),
    UniverseStock("SRF", "SRF Limited", "Mid", "Chemicals"),
    UniverseStock("DEEPAKNTR", "Deepak Nitrite", "Mid", "Chemicals"),
    UniverseStock("APLAPOLLO", "APL Apollo Tubes", "Mid", "Metals"),
    UniverseStock("JINDALSTEL", "Jindal Steel & Power", "Mid", "Metals"),
    UniverseStock("NMDC", "NMDC", "Mid", "Mining"),
    UniverseStock("CUMMINSIND", "Cummins India", "Mid", "Capital Goods"),
    UniverseStock("ABB", "ABB India", "Mid", "Capital Goods"),
    UniverseStock("SIEMENS", "Siemens", "Mid", "Capital Goods"),
    UniverseStock("LODHA", "Macrotech Developers", "Mid", "Realty"),
    UniverseStock("OBEROIRLTY", "Oberoi Realty", "Mid", "Realty"),
    UniverseStock("PHOENIXLTD", "Phoenix Mills", "Mid", "Realty"),
    UniverseStock("PERSISTENT", "Persistent Systems", "Mid", "IT"),
    UniverseStock("ZYDUSLIFE", "Zydus Lifesciences", "Mid", "Pharma"),
    UniverseStock("TORNTPHARM", "Torrent Pharma", "Mid", "Pharma"),
    UniverseStock("MAXHEALTH", "Max Healthcare", "Mid", "Healthcare"),
    UniverseStock("APOLLOHOSP", "Apollo Hospitals", "Mid", "Healthcare"),
    UniverseStock("LTIM", "LTIMindtree", "Mid", "IT"),

    # ── Small cap ──────────────────────────────────────────────────────────
    UniverseStock("RAINBOW", "Rainbow Children's Medicare", "Small", "Healthcare"),
    UniverseStock("ROUTE", "Route Mobile", "Small", "IT"),
    UniverseStock("HAPPSTMNDS", "Happiest Minds", "Small", "IT"),
    UniverseStock("LATENTVIEW", "Latent View Analytics", "Small", "IT"),
    UniverseStock("KPITTECH", "KPIT Technologies", "Small", "IT"),
    UniverseStock("CYIENT", "Cyient", "Small", "IT"),
    UniverseStock("RATNAMANI", "Ratnamani Metals", "Small", "Metals"),
    UniverseStock("GRAVITA", "Gravita India", "Small", "Metals"),
    UniverseStock("CLEAN", "Clean Science & Technology", "Small", "Chemicals"),
    UniverseStock("FINEORG", "Fine Organic Industries", "Small", "Chemicals"),
    UniverseStock("AARTIIND", "Aarti Industries", "Small", "Chemicals"),
    UniverseStock("NAVINFLUOR", "Navin Fluorine", "Small", "Chemicals"),
    UniverseStock("ANGELONE", "Angel One", "Small", "Broking"),
    UniverseStock("CDSL", "Central Depository Services", "Small", "Financial Services"),
    UniverseStock("CAMS", "Computer Age Mgmt Services", "Small", "Financial Services"),
    UniverseStock("IEX", "Indian Energy Exchange", "Small", "Power"),
    UniverseStock("KPRMILL", "K.P.R. Mill", "Small", "Textiles"),
    UniverseStock("VGUARD", "V-Guard Industries", "Small", "Consumer Durables"),
    UniverseStock("BLUESTARCO", "Blue Star", "Small", "Consumer Durables"),
    UniverseStock("CGPOWER", "CG Power & Industrial", "Small", "Capital Goods"),
    UniverseStock("TIINDIA", "Tube Investments of India", "Small", "Auto Ancillary"),
    UniverseStock("SCHAEFFLER", "Schaeffler India", "Small", "Auto Ancillary"),
    UniverseStock("ZFCVINDIA", "ZF Commercial Vehicle Control", "Small", "Auto Ancillary"),
    UniverseStock("KEI", "KEI Industries", "Small", "Electricals"),
    UniverseStock("FINPIPE", "Finolex Industries", "Small", "Pipes"),
    UniverseStock("SUPREMEIND", "Supreme Industries", "Small", "Plastics"),
    UniverseStock("CERA", "Cera Sanitaryware", "Small", "Consumer Durables"),
    UniverseStock("AFFLE", "Affle India", "Small", "AdTech"),
    UniverseStock("MAPMYINDIA", "MapmyIndia", "Small", "Tech"),
    UniverseStock("NAZARA", "Nazara Technologies", "Small", "Gaming"),
]


def by_bucket(buckets: List[str]) -> List[UniverseStock]:
    wanted = {b.lower() for b in buckets}
    return [s for s in NSE_UNIVERSE if s.bucket.lower() in wanted]


def symbols_only(stocks: List[UniverseStock]) -> List[str]:
    return [s.symbol for s in stocks]
