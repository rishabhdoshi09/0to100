"""
Plain-language explanations for every chart indicator and element.

Every explanation is written so that a 13-year-old can understand it.
Add new indicators by inserting a new entry in EXPLANATIONS below.
"""

from __future__ import annotations

EXPLANATIONS: dict[str, dict[str, str]] = {
    "vwap": {
        "name": "VWAP – Volume Weighted Average Price",
        "short": "The average price weighted by trade size — the market's true 'fair price' for today.",
        "full": (
            "VWAP stands for Volume-Weighted Average Price. Think of it like the average price "
            "everyone paid for the stock today, but bigger trades count MORE than tiny ones. "
            "If the current price is ABOVE VWAP, the stock is trading more expensive than the "
            "average buyer paid — big institutions often start selling into this. "
            "If price is BELOW VWAP, the stock is cheaper than average — often a buying opportunity. "
            "Day traders use VWAP as a 'fair value' line: they buy dips below it and sell rallies "
            "above it. Professional traders almost always check VWAP before placing a trade."
        ),
    },
    "bollinger_bands": {
        "name": "Bollinger Bands (20, 2)",
        "short": "Elastic bands around price — touching the edge signals over-extension; a squeeze signals a big move coming.",
        "full": (
            "Bollinger Bands are like elastic rubber bands wrapped around the price chart. "
            "The MIDDLE line is the average price over 20 candles (the SMA-20). "
            "The UPPER band is 2 standard deviations above — statistically, price is here only 5% of the time. "
            "The LOWER band is 2 standard deviations below — same rarity. "
            "When price TOUCHES the upper band, it has moved unusually high — it might fall back. "
            "When price TOUCHES the lower band, it has moved unusually low — it might bounce up. "
            "When the bands SQUEEZE TOGETHER (get very narrow), the stock has been calm for too long — "
            "a big explosive move is building and is about to happen. Traders call this the 'Bollinger Squeeze'. "
            "The direction of the breakout after a squeeze tells you where price is going."
        ),
    },
    "donchian_channels": {
        "name": "Donchian Channels (20)",
        "short": "The highest high and lowest low over 20 candles — breakouts signal new trends.",
        "full": (
            "Donchian Channels draw two lines: the HIGHEST price and the LOWEST price seen in the "
            "last 20 candles. Think of it as the recent trading range. "
            "If price breaks ABOVE the upper line — it's hitting new highs. That's a bullish breakout. "
            "If price breaks BELOW the lower line — it's hitting new lows. That's a bearish breakdown. "
            "Famous trend trader Richard Donchian invented this in the 1970s. "
            "The Turtle Traders (who turned $1 million into $100 million) used this exact rule: "
            "buy every new 20-day high, sell every new 20-day low. Simple but powerful."
        ),
    },
    "rsi": {
        "name": "RSI – Relative Strength Index (14)",
        "short": "A speedometer for the stock: above 70 means moving too fast (overbought), below 30 means oversold.",
        "full": (
            "RSI (Relative Strength Index) is like a speedometer for a stock's price. "
            "It measures how fast and how MUCH the price has been rising or falling over the last 14 candles. "
            "The score goes from 0 to 100. "
            "ABOVE 70 → the stock has been rising too fast and is 'overbought'. It might cool down soon. "
            "BELOW 30 → the stock has been falling too fast and is 'oversold'. It might bounce back up. "
            "Between 30 and 70 is normal territory — no extreme signal. "
            "DIVERGENCE is the advanced trick: if price makes a NEW HIGH but RSI makes a LOWER HIGH, "
            "the uptrend is running out of steam — a reversal might be coming soon. "
            "Same works in reverse for downtrends."
        ),
    },
    "macd": {
        "name": "MACD – Moving Average Convergence Divergence (12, 26, 9)",
        "short": "Shows when two speed lines cross — crossovers signal trend changes.",
        "full": (
            "MACD stands for Moving Average Convergence Divergence. Don't be scared by the name! "
            "It shows you two moving averages and whether they're getting closer or farther apart. "
            "MACD Line = fast moving average (12 periods) MINUS slow moving average (26 periods). "
            "Signal Line = a smoothed version (9-period EMA) of the MACD line. "
            "Histogram = MACD minus Signal. Green bars = momentum growing. Red bars = momentum fading. "
            "THE KEY SIGNAL: When MACD crosses ABOVE the Signal line → bullish (potential buy). "
            "When MACD crosses BELOW the Signal line → bearish (potential sell). "
            "When histogram bars switch from negative to positive, momentum is shifting up — "
            "this often happens before the actual price breakout."
        ),
    },
    "atr": {
        "name": "ATR – Average True Range (14)",
        "short": "Measures how wild or calm the stock is — high ATR means big daily moves, use wider stop losses.",
        "full": (
            "ATR stands for Average True Range and it measures VOLATILITY — how much the stock moves daily. "
            "It looks at each candle and measures the TRUE RANGE, which is the biggest of: "
            "(1) Today's High minus Today's Low, "
            "(2) How far today's High is from yesterday's Close, "
            "(3) How far today's Low is from yesterday's Close. "
            "These extra checks handle gaps (when stock opens much higher or lower than yesterday's close). "
            "Then it averages this over 14 candles. "
            "HIGH ATR → the stock is very volatile, moving big amounts each day — risky but high opportunity. "
            "LOW ATR → the stock is calm and quiet — lower risk but smaller profits. "
            "PRACTICAL USE: Set your stop loss at 1.5x or 2x ATR below your entry. "
            "This way, normal day-to-day noise won't stop you out, but a real reversal will."
        ),
    },
    "volume_profile": {
        "name": "Volume Profile – VPVR (Visible Range)",
        "short": "Shows WHERE the most trading happened — busy price levels matter most.",
        "full": (
            "Volume Profile shows you WHERE the most trading happened, not just WHEN. "
            "Imagine all the trades stacked sideways against the price axis as horizontal bars. "
            "The LONGEST bar is called the POC (Point of Control) — the price where the most shares "
            "traded hands. Price is magnetically attracted to POC and often returns to it. "
            "The VALUE AREA (coloured differently) covers 70% of all trading volume — this is 'fair value'. "
            "HIGH VOLUME NODES (HVN) are prices with lots of trading — price moves slowly here because "
            "buyers and sellers are balanced and fighting. "
            "LOW VOLUME NODES (LVN) are price levels with almost no trading — price moves FAST through "
            "these like an empty highway, because no one is there to slow it down. "
            "When price breaks through an LVN, it can travel quickly to the next HVN."
        ),
    },
    "poc": {
        "name": "POC – Point of Control",
        "short": "The single price where the most shares traded — acts like a magnet, price always returns here.",
        "full": (
            "POC stands for Point of Control — the single price level where the MOST shares traded "
            "during the entire visible period. "
            "Think of it as the 'fairest' price in the market's eyes — where the most agreement happened. "
            "After price moves away from the POC, it very often RETURNS to it — like a rubber band snapping back. "
            "If current price is ABOVE POC → market is bullish (buyers are in control). "
            "If current price is BELOW POC → market is bearish (sellers are in control). "
            "A trade back toward POC after a big move is called 'returning to value' — "
            "this is one of the most reliable patterns in professional market profile trading."
        ),
    },
    "value_area": {
        "name": "Value Area (VA)",
        "short": "The price range containing 70% of volume — inside is 'fair value', outside is 'extreme'.",
        "full": (
            "The Value Area is the price range that contains 70% of all the trading volume. "
            "Think of it as the market's comfort zone — where most participants agreed the price was fair. "
            "VALUE AREA HIGH (VAH): The top of this zone. When price pushes above VAH, "
            "it is trading expensively relative to where most people traded. "
            "VALUE AREA LOW (VAL): The bottom of this zone. When price drops below VAL, "
            "it is trading cheaply relative to most participants. "
            "Classic strategy: if price opens OUTSIDE the Value Area and quickly pops back INSIDE, "
            "it often travels all the way to the OTHER SIDE of the VA — traders call this the '80% rule'."
        ),
    },
    "footprint": {
        "name": "Footprint Chart (Order Flow)",
        "short": "An X-ray of each candle showing buyers (green) vs sellers (red) at every price level.",
        "full": (
            "A Footprint Chart is like an X-ray of each candle — it shows EXACTLY what happened "
            "at every price level inside that candle. "
            "GREEN number = how many shares were BOUGHT at that price level (ask volume — buyers were aggressive). "
            "RED number = how many shares were SOLD at that price level (bid volume — sellers were aggressive). "
            "If RED is much bigger than GREEN, big sellers are active — the price might drop. "
            "A STAR (⭐) marks an IMBALANCE — where one side is more than 3 times the other. "
            "That is where the smart money (big institutions) is showing its hand. "
            "NOTE: This chart simulates bid/ask volumes from OHLCV data because real tick data "
            "requires a live Zerodha WebSocket connection. The simulation is based on where the "
            "candle closes relative to its high and low."
        ),
    },
    "cumulative_delta": {
        "name": "Cumulative Delta",
        "short": "Running total of buy vs sell aggression — divergence from price is a powerful warning.",
        "full": (
            "Cumulative Delta is the running total of (ask volume − bid volume) over time. "
            "When buyers HIT the ask price (buying aggressively), delta goes UP. "
            "When sellers HIT the bid price (selling aggressively), delta goes DOWN. "
            "HEALTHY uptrend: price goes up AND delta goes up → buyers are genuinely fueling the move. "
            "FAKE uptrend (DIVERGENCE): price goes up BUT delta goes down → "
            "sellers are actually more aggressive even as price rises. This is a WARNING — "
            "the uptrend might collapse soon as the buyers run out. "
            "Same logic in reverse for downtrends: if price falls but delta rises, "
            "the selling pressure is drying up — bounce incoming. "
            "This is one of the most powerful signals professional traders use."
        ),
    },
    "liquidity_heatmap": {
        "name": "Liquidity Heatmap (Order Book Depth)",
        "short": "Shows where big buy/sell orders are stacked — brighter colour = more orders = stronger support/resistance.",
        "full": (
            "The Liquidity Heatmap shows the ORDER BOOK — all pending buy and sell orders "
            "at different price levels, waiting to be executed. "
            "GREEN bars = big BUY orders (bids) — these act like a floor under the price. "
            "Price often bounces UP from large green clusters. "
            "RED bars = big SELL orders (asks) — these act like a ceiling above the price. "
            "Price often gets rejected DOWN from large red clusters. "
            "The BRIGHTER and LONGER the bar, the more orders are sitting there — "
            "stronger support or resistance. "
            "When a big level gets CONSUMED (eaten through by the opposite side), "
            "price can move very fast to the next liquidity cluster because there are no more orders "
            "standing in the way. "
            "NOTE: This uses a simulated order book. Live data requires Zerodha WebSocket market depth."
        ),
    },
    "imbalance": {
        "name": "Order Flow Imbalance (⭐)",
        "short": "One side is 3× more active than the other — the dominant side is controlling price at this level.",
        "full": (
            "An imbalance occurs when buyers are MORE THAN 3 TIMES more active than sellers "
            "at a price level (or vice versa). "
            "This is extremely important because it reveals where the SMART MONEY is. "
            "A BUYING IMBALANCE (green star ⭐ above candle): at this price level, "
            "buyers were so aggressive that they outnumbered sellers 3:1 or more. "
            "This level often becomes SUPPORT — if price comes back here, buyers will defend it. "
            "A SELLING IMBALANCE (red star ⭐ below candle): sellers dominated here 3:1+. "
            "This level often becomes RESISTANCE — if price rallies back, sellers will reappear. "
            "Traders hunt for these imbalance zones and trade them like magnets."
        ),
    },
}


def explain(indicator: str) -> str:
    """Return a formatted plain-language explanation for the given indicator."""
    key = indicator.lower().replace(" ", "_").replace("-", "_")
    entry = EXPLANATIONS.get(key)
    if entry is None:
        available = ", ".join(sorted(EXPLANATIONS.keys()))
        return (
            f"\nNo explanation found for '{indicator}'.\n"
            f"Available indicators: {available}\n"
        )
    border = "=" * 62
    return (
        f"\n{border}\n"
        f"  {entry['name']}\n"
        f"{border}\n\n"
        f"{entry['full']}\n\n"
        f"ONE-LINE SUMMARY: {entry['short']}\n"
        f"{border}\n"
    )


def list_all() -> list[str]:
    """Return sorted list of all available indicator keys."""
    return sorted(EXPLANATIONS.keys())
