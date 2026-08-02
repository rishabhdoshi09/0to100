"""
Peer Strength Ranker — given any stock, rank it vs its sector peers.

"HDFCBANK looks good" is incomplete. The right question is:
"Is HDFCBANK the BEST banking stock to buy right now?"

This module answers that by scoring all stocks in the same sector
on momentum, accumulation, relative strength, and volume — then ranking them.
A trader should always buy the leader within the sector, not just any stock.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional


# Sector → peer stocks mapping (NSE, top FNO-liquid names per sector)
_SECTOR_PEERS: dict[str, list[str]] = {
    "BANKING": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
        "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    ],
    "IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "LTIM", "MPHASIS", "PERSISTENT", "COFORGE", "KPITTECH",
    ],
    "AUTO": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
        "EICHERMOT", "ASHOKLEY", "TVSMOTOR", "MRF", "MOTHERSON",
    ],
    "PHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
        "TORNTPHARM", "ALKEM", "BIOCON", "LUPIN", "IPCALAB",
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "MARICO", "COLPAL", "EMAMILTD", "GODREJCP", "TATACONSUM",
    ],
    "METALS": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL",
        "NATIONALUM", "NMDC", "MOIL", "APLAPOLLO", "WELCORP",
    ],
    "ENERGY": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL",
        "POWERGRID", "NTPC", "TATAPOWER", "ADANIGREEN", "TORNTPOWER",
    ],
    "REALTY": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD",
        "BRIGADE", "SOBHA", "MAHLIFE", "SUNTECK", "KOLTEPATIL",
    ],
    "CEMENT": [
        "ULTRACEMCO", "GRASIM", "AMBUJACEM", "ACC", "SHREECEM",
        "DALMIACMT", "JKCEMENT", "RAMCOCEM", "HEIDELBERG", "BIRLACORPN",
    ],
    "INSURANCE_FINANCE": [
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULI",
        "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM", "PIRAMALENT", "L&TFH",
    ],
    "CONSUMER_DURABLES": [
        "TITAN", "HAVELLS", "VOLTAS", "CROMPTON", "VGUARD",
        "BLUESTAR", "WHIRLPOOL", "CESC", "INOXWIND", "POLYCAB",
    ],
    "CHEMICALS": [
        "PIDILITIND", "ASIANPAINT", "BERGERPAINTS", "AKZO", "KANSAINER",
        "AAPL", "DEEPAKNTR", "ATUL", "SRF", "NAVIN",
    ],
}


@dataclass
class PeerRankResult:
    symbol: str
    rank: int                   # 1 = best in sector
    total_peers: int
    sector: str
    score: float                # composite peer score 0-100
    momentum_rank: int
    rs_rank: int
    accum_rank: int
    price: float
    momentum_20d: float
    rs_vs_nifty: float
    verdict: str                # "SECTOR_LEADER" | "TOP_3" | "MID_PACK" | "LAGGARD"
    peers_ranked: list[dict]    # full ranked list for display


def _find_sector(symbol: str) -> Optional[str]:
    sym = symbol.upper()
    for sector, peers in _SECTOR_PEERS.items():
        if sym in peers:
            return sector
    return None


def _score_peer(symbol: str) -> dict:
    """Score a single stock for peer ranking. Returns lightweight score dict."""
    try:
        from agents.tools import get_technical_indicators
        d = get_technical_indicators(symbol, days=60)
        if "error" in d:
            return {"symbol": symbol, "score": 0, "error": True}

        score = 0.0
        price    = float(d.get("ltp") or d.get("price") or 0)
        mom20    = float(d.get("momentum_20d_pct") or 0)
        mom5     = float(d.get("momentum_5d_pct") or 0)
        vr       = float(d.get("volume_ratio") or 1.0)
        rsi      = float(d.get("rsi_14") or 50)
        gc       = bool(d.get("golden_cross", False))
        ema20    = d.get("ema_20")
        above20  = float(d.get("pct_above_sma20") or 0)
        above50  = float(d.get("pct_above_sma50") or 0)

        # Momentum component (40 pts)
        if mom20 > 15:   score += 40
        elif mom20 > 8:  score += 30
        elif mom20 > 3:  score += 20
        elif mom20 > 0:  score += 10

        # Relative strength proxy (30 pts) — price vs moving averages
        if above50 > 5 and gc:  score += 30
        elif above50 > 0 and gc: score += 22
        elif above50 > 0:        score += 14
        elif above20 > 0:        score += 7

        # Volume / accumulation (20 pts)
        if vr > 2.0:    score += 20
        elif vr > 1.5:  score += 14
        elif vr > 1.0:  score += 7

        # RSI quality zone (10 pts)
        if 55 <= rsi <= 70:     score += 10
        elif 45 <= rsi < 55:    score += 5

        # Rough RS vs Nifty (use 20d momentum as proxy since Nifty ~12% pa = ~1% per week)
        rs_proxy = mom20 - 1.0  # > 0 = outperforming Nifty

        return {
            "symbol": symbol, "score": round(score, 1), "price": price,
            "momentum_20d": mom20, "momentum_5d": mom5,
            "volume_ratio": vr, "rsi": rsi,
            "rs_proxy": round(rs_proxy, 2), "above50": above50,
        }
    except Exception:
        return {"symbol": symbol, "score": 0, "error": True}


def rank_vs_peers(symbol: str) -> Optional[PeerRankResult]:
    """
    Rank `symbol` against all stocks in its sector.
    Returns None if the sector is not found.
    """
    sector = _find_sector(symbol.upper())
    if sector is None:
        return None

    peers = _SECTOR_PEERS[sector]

    # Score all peers in parallel
    scores: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_score_peer, p): p for p in peers}
        for f in as_completed(futs):
            try:
                r = f.result()
                if not r.get("error"):
                    scores.append(r)
            except Exception:
                pass

    if not scores:
        return None

    # Sort by composite score
    scores.sort(key=lambda x: x["score"], reverse=True)

    # Find target symbol's rank
    target_rank = next(
        (i + 1 for i, s in enumerate(scores) if s["symbol"] == symbol.upper()),
        len(scores),
    )
    target_data = next((s for s in scores if s["symbol"] == symbol.upper()), {})

    # Assign verdict
    n = len(scores)
    if target_rank == 1:                          verdict = "SECTOR_LEADER"
    elif target_rank <= max(3, int(n * 0.3)):     verdict = "TOP_3"
    elif target_rank <= max(5, int(n * 0.6)):     verdict = "MID_PACK"
    else:                                         verdict = "LAGGARD"

    # Compute sub-ranks for momentum and RS
    mom_sorted = sorted(scores, key=lambda x: x.get("momentum_20d", 0), reverse=True)
    rs_sorted  = sorted(scores, key=lambda x: x.get("rs_proxy", 0), reverse=True)
    acc_sorted = sorted(scores, key=lambda x: x.get("volume_ratio", 1), reverse=True)

    mom_rank = next((i + 1 for i, s in enumerate(mom_sorted) if s["symbol"] == symbol.upper()), n)
    rs_rank  = next((i + 1 for i, s in enumerate(rs_sorted)  if s["symbol"] == symbol.upper()), n)
    acc_rank = next((i + 1 for i, s in enumerate(acc_sorted) if s["symbol"] == symbol.upper()), n)

    peers_ranked = [
        {
            "rank": i + 1, "symbol": s["symbol"],
            "score": s["score"], "price": s.get("price", 0),
            "momentum_20d": s.get("momentum_20d", 0),
            "rs_proxy": s.get("rs_proxy", 0),
            "volume_ratio": s.get("volume_ratio", 1),
        }
        for i, s in enumerate(scores)
    ]

    return PeerRankResult(
        symbol=symbol.upper(),
        rank=target_rank,
        total_peers=n,
        sector=sector,
        score=target_data.get("score", 0),
        momentum_rank=mom_rank,
        rs_rank=rs_rank,
        accum_rank=acc_rank,
        price=target_data.get("price", 0),
        momentum_20d=target_data.get("momentum_20d", 0),
        rs_vs_nifty=target_data.get("rs_proxy", 0),
        verdict=verdict,
        peers_ranked=peers_ranked,
    )
