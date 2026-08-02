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
    """Score a single stock for peer ranking from local official history only.

    Must not call Kite/yfinance — Stock Intelligence embeds this for every peer and
    network round-trips made symbol pages (e.g. GAIL) hang for tens of seconds.
    """
    try:
        from data.bhavcopy_runtime import get_ohlcv

        frame = get_ohlcv(symbol)
        if frame is None or len(frame) < 25:
            return {"symbol": symbol, "score": 0, "error": True}

        data = frame.sort_index().copy()
        close = data["close"].astype(float).dropna()
        if len(close) < 25:
            return {"symbol": symbol, "score": 0, "error": True}

        price = float(close.iloc[-1])
        mom20 = ((price / float(close.iloc[-21]) - 1.0) * 100.0) if len(close) > 21 else 0.0
        mom5 = ((price / float(close.iloc[-6]) - 1.0) * 100.0) if len(close) > 6 else 0.0

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        last_loss = float(loss.iloc[-1]) or 1e-9
        rsi = 100.0 - (100.0 / (1.0 + float(gain.iloc[-1]) / last_loss))

        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else ema50
        above20 = ((price / ema20) - 1.0) * 100.0 if ema20 else 0.0
        above50 = ((price / ema50) - 1.0) * 100.0 if ema50 else 0.0
        gc = ema50 > ema200

        vr = 1.0
        if "volume" in data.columns:
            volume = float(data["volume"].astype(float).iloc[-1])
            avg_volume = float(data["volume"].astype(float).tail(20).mean())
            if avg_volume > 0:
                vr = volume / avg_volume

        score = 0.0
        if mom20 > 15:
            score += 40
        elif mom20 > 8:
            score += 30
        elif mom20 > 3:
            score += 20
        elif mom20 > 0:
            score += 10

        if above50 > 5 and gc:
            score += 30
        elif above50 > 0 and gc:
            score += 22
        elif above50 > 0:
            score += 14
        elif above20 > 0:
            score += 7

        if vr > 2.0:
            score += 20
        elif vr > 1.5:
            score += 14
        elif vr > 1.0:
            score += 7

        if 55 <= rsi <= 70:
            score += 10
        elif 45 <= rsi < 55:
            score += 5

        rs_proxy = mom20 - 1.0

        return {
            "symbol": symbol,
            "score": round(score, 1),
            "price": round(price, 2),
            "momentum_20d": round(mom20, 2),
            "momentum_5d": round(mom5, 2),
            "volume_ratio": round(vr, 2),
            "rsi": round(rsi, 2),
            "rs_proxy": round(rs_proxy, 2),
            "above50": round(above50, 2),
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
