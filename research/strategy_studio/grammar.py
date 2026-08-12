"""
🔤 Approved feature grammar — the ONLY building blocks autonomous discovery may use.

Every block is point-in-time-safe by construction (it names a computation the existing
PIT primitives already provide, or one clearly available at signal time). Discovery may
combine blocks into readable rules; it may NOT invent arbitrary features. Fundamentals
appear only where genuinely point-in-time (result-publication-dated); current
fundamentals are excluded on purpose.

Pure data + tiny helpers. No streamlit, no network, no order path.
"""
from __future__ import annotations

# group -> list of (block_id, plain-language label, point_in_time_available)
PRICE_BEHAVIOUR = [
    ("ret_3m", "3-month return", True),
    ("ret_6m", "6-month return", True),
    ("ret_12m1m", "12-month return (skip last month)", True),
    ("trend_above_200dma", "above a rising 200-day average", True),
    ("rel_strength", "stronger than the market", True),
    ("breakout_pivot", "breaks above a prior resistance", True),
    ("pullback_to_ema", "pulls back to a moving average", True),
    ("mean_reversion_low", "bounces from an oversold low", True),
    ("vol_contraction", "volatility has tightened", True),
    ("vol_expansion", "volatility is expanding", True),
    ("gap_up", "opens with an up-gap", True),
    ("dist_from_50dma", "distance from the 50-day average", True),
    ("dist_from_52w_high", "distance from the 52-week high", True),
    ("support_resistance", "near support/resistance", True),
    ("drawdown_recovery", "recovering from a drawdown", True),
]

PARTICIPATION = [
    ("volume", "volume", True),
    ("rel_volume", "relative volume", True),
    ("delivery_pct", "delivery %", "when_available"),
    ("turnover", "turnover", True),
    ("accumulation", "accumulation footprint", True),
    ("volume_dryup", "volume dry-up in the base", True),
    ("volume_expansion", "volume expansion on the move", True),
    ("liquidity", "liquidity (tradable size)", True),
]

MARKET_CONTEXT = [
    ("nifty_trend", "Nifty trend", True),
    ("breadth", "market breadth", True),
    ("sector_strength", "sector strength", True),
    ("sector_breadth", "sector breadth", True),
    ("vol_regime", "volatility regime", True),
    ("market_state", "bull / bear / sideways / stressed", True),
]

# fundamentals — ONLY where point-in-time (publication-dated) data genuinely exists
FUNDAMENTALS = [
    ("sales_growth", "sales growth", "pit_only"),
    ("earnings_growth", "earnings growth", "pit_only"),
    ("profitability", "profitability", "pit_only"),
    ("valuation", "valuation", "pit_only"),
    ("balance_sheet_quality", "balance-sheet quality", "pit_only"),
    ("result_event", "just published results", "pit_only"),
]

GROUPS = {
    "price": PRICE_BEHAVIOUR, "participation": PARTICIPATION,
    "context": MARKET_CONTEXT, "fundamentals": FUNDAMENTALS,
}

# which building blocks each family naturally draws on (keeps generated strategies
# readable and family-appropriate — not a soup of every feature)
FAMILY_BLOCKS: dict[str, list[str]] = {
    "cross_sectional_momentum": ["ret_6m", "ret_12m1m", "rel_strength", "liquidity"],
    "trend_following": ["trend_above_200dma", "dist_from_50dma", "nifty_trend"],
    "breakout": ["breakout_pivot", "vol_contraction", "rel_volume", "sector_strength"],
    "pullback": ["pullback_to_ema", "rel_strength", "volume_dryup"],
    "mean_reversion": ["mean_reversion_low", "drawdown_recovery", "vol_regime"],
    "sector_rotation": ["sector_strength", "sector_breadth", "rel_strength"],
    "volatility_contraction": ["vol_contraction", "volume_dryup", "breakout_pivot"],
    "quality_plus_momentum": ["ret_6m", "profitability", "earnings_growth"],
    "event_driven": ["result_event", "gap_up", "rel_volume"],
    "regime_specific": ["market_state", "breadth", "vol_regime"],
    "rule_combo": ["rel_strength", "breakout_pivot", "volume_dryup", "sector_strength"],
}

PLAIN = {bid: label for grp in GROUPS.values() for bid, label, _ in grp}


def block_label(block_id: str) -> str:
    return PLAIN.get(block_id, block_id)


def is_pit_available(block_id: str, has_fundamentals_pit: bool = False,
                     has_delivery: bool = False) -> bool:
    """Whether a block can be used given what data is genuinely point-in-time available.
    Fundamentals need a real publication-dated source; delivery needs the field present."""
    for grp in GROUPS.values():
        for bid, _label, avail in grp:
            if bid == block_id:
                if avail is True:
                    return True
                if avail == "when_available":
                    return has_delivery
                if avail == "pit_only":
                    return has_fundamentals_pit
    return False
