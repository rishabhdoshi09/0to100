"""
US universe — a curated list of liquid US names (Nasdaq-100 + large-cap
S&P leaders). Kept deliberately liquid: US penny/illiquid stocks are a
different game, and the scanner's edge lives in names that trade.

Symbols are bare Yahoo tickers (AAPL, MSFT) — no suffix.
"""
from __future__ import annotations

# Nasdaq-100 heavyweights + high-liquidity S&P leaders across sectors.
# Trimmed to names that reliably return clean daily data.
_US = {
    # Mega-cap tech
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
    "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms",
    "TSLA": "Tesla", "AVGO": "Broadcom", "ORCL": "Oracle",
    "ADBE": "Adobe", "CRM": "Salesforce", "AMD": "Advanced Micro Devices",
    "INTC": "Intel", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "MU": "Micron", "AMAT": "Applied Materials", "LRCX": "Lam Research",
    "ADI": "Analog Devices", "KLAC": "KLA Corp", "SNPS": "Synopsys",
    "CDNS": "Cadence", "MRVL": "Marvell", "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike", "FTNT": "Fortinet", "NOW": "ServiceNow",
    "INTU": "Intuit", "SNOW": "Snowflake", "NET": "Cloudflare",
    "DDOG": "Datadog", "SMCI": "Super Micro", "ARM": "Arm Holdings",
    "PLTR": "Palantir", "MSTR": "MicroStrategy",
    # Communication / internet
    "NFLX": "Netflix", "DIS": "Disney", "CMCSA": "Comcast",
    "T": "AT&T", "VZ": "Verizon", "TMUS": "T-Mobile",
    "UBER": "Uber", "ABNB": "Airbnb", "SHOP": "Shopify",
    "SPOT": "Spotify", "PYPL": "PayPal", "XYZ": "Block",
    "COIN": "Coinbase", "HOOD": "Robinhood",
    # Consumer
    "COST": "Costco", "WMT": "Walmart", "HD": "Home Depot",
    "NKE": "Nike", "MCD": "McDonald's", "SBUX": "Starbucks",
    "PEP": "PepsiCo", "KO": "Coca-Cola", "PG": "Procter & Gamble",
    "LULU": "Lululemon", "TGT": "Target", "LOW": "Lowe's",
    # Financials
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "V": "Visa", "MA": "Mastercard",
    "AXP": "American Express", "WFC": "Wells Fargo", "SCHW": "Charles Schwab",
    "BLK": "BlackRock", "C": "Citigroup",
    # Health
    "LLY": "Eli Lilly", "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson",
    "ABBV": "AbbVie", "MRK": "Merck", "PFE": "Pfizer",
    "TMO": "Thermo Fisher", "ABT": "Abbott", "ISRG": "Intuitive Surgical",
    "VRTX": "Vertex", "REGN": "Regeneron", "AMGN": "Amgen",
    # Industrial / energy / auto
    "CAT": "Caterpillar", "BA": "Boeing", "GE": "GE Aerospace",
    "HON": "Honeywell", "UPS": "UPS", "RTX": "RTX Corp",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "F": "Ford", "GM": "General Motors", "DE": "Deere",
    # Momentum / growth favourites
    "AMZN2": "", "RIVN": "Rivian", "LCID": "Lucid", "SOFI": "SoFi",
    "DKNG": "DraftKings", "RBLX": "Roblox", "U": "Unity",
    "ROKU": "Roku", "PINS": "Pinterest", "SNAP": "Snap",
}
_US.pop("AMZN2", None)


def get_us_universe() -> list[str]:
    return sorted(_US.keys())


def get_us_universe_with_names() -> dict[str, str]:
    return dict(_US)
