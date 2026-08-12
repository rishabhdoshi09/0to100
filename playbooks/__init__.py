"""
playbooks/__init__.py
Institutional playbook registry for NSE India trading system.

Professionals trade playbooks, not indicators. Each playbook encodes a
complete behavioral pattern — setup, context, execution, and historical
expectancy — so the system reasons about what the market is doing, not
just what a single indicator says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Playbook:
    id: str
    name: str
    category: str  # BREAKOUT | MOMENTUM | REVERSAL | POSITIONAL | CONTINUATION
    emoji: str

    # Market context requirements
    ideal_market_regime: list[str]      # e.g. ["TRENDING_BULL", "EXPANSION"]
    avoid_market_regime: list[str]      # e.g. ["TRENDING_BEAR", "DISTRIBUTION"]
    ideal_volatility_regime: list[str]  # e.g. ["LOW_VOL_COMPRESSION", "NORMAL"]
    ideal_breadth: list[str]            # e.g. ["STRONG", "NEUTRAL"]
    ideal_institutional_activity: list[str]  # e.g. ["ACCUMULATION", "RISK_ON"]

    # Setup detection criteria
    behavioral_conditions: list[str]    # human-readable behavior descriptions
    setup_conditions: list[str]         # specific quantitative conditions
    failure_characteristics: list[str]  # how/why this setup fails

    # Execution
    entry_trigger: str   # specific entry condition
    stop_logic: str      # how to set stop
    target_logic: str    # how to set target
    trail_logic: str     # how to trail

    # Historical expectancy (static baseline — overridden by live ExpectancyEngine)
    baseline_win_rate: float        # 0–1
    baseline_avg_win_pct: float
    baseline_avg_loss_pct: float
    baseline_expectancy: float      # win_rate × avg_win - (1-wr) × avg_loss
    baseline_avg_hold_days: int
    baseline_risk_reward: float

    # Qualitative
    liquidity_requirement: str
    ideal_sector_conditions: str
    notes: str

    # Runtime flag — set externally by RegimeEngine before scoring
    _regime_aligned: bool = field(default=False, repr=False, compare=False)

    @property
    def regime_aligned(self) -> bool:
        return self._regime_aligned

    @regime_aligned.setter
    def regime_aligned(self, value: bool) -> None:
        self._regime_aligned = value

    def __str__(self) -> str:
        return f"{self.emoji} {self.name} [{self.category}] WR={self.baseline_win_rate:.0%} E={self.baseline_expectancy:.2f}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Playbook] = {}


def _register(pb: Playbook) -> Playbook:
    REGISTRY[pb.id] = pb
    return pb


# 1 ─ VCP_BREAKOUT ──────────────────────────────────────────────────────────

_vcp_wr = 0.58
_vcp_aw = 0.18
_vcp_al = 0.06

VCP_BREAKOUT = _register(Playbook(
    id="VCP_BREAKOUT",
    name="Volatility Contraction Pattern Breakout",
    category="BREAKOUT",
    emoji="🎯",

    ideal_market_regime=["TRENDING_BULL", "EXPANSION"],
    avoid_market_regime=["TRENDING_BEAR", "DISTRIBUTION", "CHOPPY_BEAR"],
    ideal_volatility_regime=["LOW_VOL_COMPRESSION", "NORMAL"],
    ideal_breadth=["STRONG", "NEUTRAL"],
    ideal_institutional_activity=["ACCUMULATION", "RISK_ON"],

    behavioral_conditions=[
        "Stock contracts volatility in successive tighter price ranges after a strong prior uptrend",
        "Volume dries to its lowest point during the final contraction — supply exhausted",
        "Price holds above SMA200, confirming Stage 2 uptrend intact",
        "Explosive breakout above pivot on surge in volume signals institutional re-entry",
    ],
    setup_conditions=[
        "3 or more successive contractions, each < 75% of the prior contraction's width",
        "Volume declining on each contraction leg (supply leaving the stock)",
        "Price > SMA200 (Stage 2 confirmation)",
        "Breakout day volume > 1.5× 50-day average volume",
        "Pivot is the highest intraday high of the final base",
    ],
    failure_characteristics=[
        "Volume fails to expand meaningfully on the breakout day",
        "Broad market turns choppy or rolls over within the first week",
        "Stock closes back below the pivot within 3 sessions (false breakout)",
        "Sector rotates out — leadership changes mid-move",
    ],

    entry_trigger="Buy on close above pivot OR on first pullback to pivot that holds with declining volume",
    stop_logic="Below the lowest point of the final contraction; typically 5–8% below entry",
    target_logic="Measure the depth of the deepest contraction and project 2–3× from pivot",
    trail_logic="Trail with 10EMA; tighten to 21EMA once +15% unrealised gain",

    baseline_win_rate=_vcp_wr,
    baseline_avg_win_pct=_vcp_aw,
    baseline_avg_loss_pct=_vcp_al,
    baseline_expectancy=round(_vcp_wr * _vcp_aw - (1 - _vcp_wr) * _vcp_al, 4),
    baseline_avg_hold_days=28,
    baseline_risk_reward=3.0,

    liquidity_requirement="Min 50Cr daily turnover (NSE EQ segment)",
    ideal_sector_conditions="Sector in Stage 2 uptrend; at least 2 other leaders setting up simultaneously",
    notes="Minervini's VCP is most reliable when the stock has already run 50–100%+ before the base. "
          "On NSE, requires F&O eligibility screening to avoid illiquid traps.",
))

# 2 ─ MOMENTUM_EXPANSION ────────────────────────────────────────────────────

_me_wr = 0.62
_me_aw = 0.14
_me_al = 0.05

MOMENTUM_EXPANSION = _register(Playbook(
    id="MOMENTUM_EXPANSION",
    name="Momentum Expansion Breakout",
    category="MOMENTUM",
    emoji="🚀",

    ideal_market_regime=["TRENDING_BULL", "EXPANSION"],
    avoid_market_regime=["TRENDING_BEAR", "DISTRIBUTION"],
    ideal_volatility_regime=["NORMAL", "EXPANDING"],
    ideal_breadth=["STRONG"],
    ideal_institutional_activity=["RISK_ON", "ACCUMULATION"],

    behavioral_conditions=[
        "Stock breaking out of a multi-week base on above-average volume",
        "Relative strength vs Nifty50 visibly improving over the past 4–6 weeks",
        "Sector showing broad participation — not a lone-wolf breakout",
        "Price action is smooth and orderly; no wild intraday reversals",
    ],
    setup_conditions=[
        "RSI between 60 and 80 (momentum present, not yet extended)",
        "Price > SMA20 > SMA50 > SMA200 (full bullish stack)",
        "Volume on breakout day > 1.5× 20-day average",
        "5-day price change > 8% (expansion of range)",
        "RS line vs Nifty at or near 52-week highs",
    ],
    failure_characteristics=[
        "Broad market breadth deteriorates within first week of entry",
        "Volume spike is one-day wonder — no follow-through buying",
        "Stock stalls at prior resistance or round number level",
        "Sector peer stocks failing to confirm the move",
    ],

    entry_trigger="Buy on volume-confirmed break of multi-week base; avoid chasing > 5% above pivot",
    stop_logic="Below the 10-day low of the base; maximum 5% from entry",
    target_logic="Target 1.5–2× the base depth projected from the breakout pivot",
    trail_logic="Trail with 21EMA on daily chart; move stop to breakeven after +8% gain",

    baseline_win_rate=_me_wr,
    baseline_avg_win_pct=_me_aw,
    baseline_avg_loss_pct=_me_al,
    baseline_expectancy=round(_me_wr * _me_aw - (1 - _me_wr) * _me_al, 4),
    baseline_avg_hold_days=12,
    baseline_risk_reward=round(_me_aw / _me_al, 2),

    liquidity_requirement="Min 100Cr daily turnover; F&O preferred for hedging",
    ideal_sector_conditions="IT, financials, or consumer discretionary in strong FII inflow cycle",
    notes="Works best in the first 6 weeks of a new market upleg. "
          "After week 8 of a bull run, selectivity must increase significantly.",
))

# 3 ─ EARLY_LEADER ──────────────────────────────────────────────────────────

_el_wr = 0.54
_el_aw = 0.22
_el_al = 0.08

EARLY_LEADER = _register(Playbook(
    id="EARLY_LEADER",
    name="Early Leader (First to Break)",
    category="POSITIONAL",
    emoji="🌅",

    ideal_market_regime=["CHOPPY", "COMPRESSION", "EARLY_BULL"],
    avoid_market_regime=["TRENDING_BEAR", "DISTRIBUTION"],
    ideal_volatility_regime=["LOW_VOL_COMPRESSION", "NORMAL"],
    ideal_breadth=["NEUTRAL", "RECOVERING"],
    ideal_institutional_activity=["ACCUMULATION", "STEALTH_BUYING"],

    behavioral_conditions=[
        "Stock breaks out while the broad market (Nifty) is still sideways or recovering",
        "Displays positive relative strength vs Nifty for 20+ consecutive trading days",
        "Accumulation pattern visible on weekly chart — tight closes in upper half of range",
        "No significant distribution weeks in the prior base",
    ],
    setup_conditions=[
        "RS line vs Nifty positive for 20+ trading days and making new highs",
        "Volume drying up during base formation, then a pocket pivot (volume spike > 1.5× on up day)",
        "Stage 1-to-2 transition: price crossing SMA200 on volume",
        "Base formed during market weakness — stock refuses to make new lows",
    ],
    failure_characteristics=[
        "Broad market fails to follow and re-enters downtrend",
        "Stock is a sector outlier with no peer confirmation",
        "Earnings catalyst is weak or upcoming — narrative gap",
        "Thin liquidity amplifies volatility, triggering stops prematurely",
    ],

    entry_trigger="Pocket pivot (vol > 1.5× 10-day avg on up day) OR close above Stage 1 range ceiling",
    stop_logic="Below the lowest point of the base; accept wider 8–10% stop for positional trade",
    target_logic="Project 3–4× risk; hold for 30–90 days as market confirms new uptrend",
    trail_logic="Weekly chart 10-week MA trail; do not use tight daily stop on positional entries",

    baseline_win_rate=_el_wr,
    baseline_avg_win_pct=_el_aw,
    baseline_avg_loss_pct=_el_al,
    baseline_expectancy=round(_el_wr * _el_aw - (1 - _el_wr) * _el_al, 4),
    baseline_avg_hold_days=45,
    baseline_risk_reward=round(_el_aw / _el_al, 2),

    liquidity_requirement="Min 25Cr daily turnover acceptable for positional sizing",
    ideal_sector_conditions="Sector ignored by analysts; potential re-rating catalyst on the horizon",
    notes="Highest conviction plays come when 3+ early leaders emerge simultaneously "
          "from different sectors, signalling a broad market inflection point.",
))

# 4 ─ ACCUMULATION_BREAKOUT ─────────────────────────────────────────────────

_ab_wr = 0.60
_ab_aw = 0.15
_ab_al = 0.05

ACCUMULATION_BREAKOUT = _register(Playbook(
    id="ACCUMULATION_BREAKOUT",
    name="Accumulation Breakout (Wyckoff)",
    category="BREAKOUT",
    emoji="📦",

    ideal_market_regime=["TRENDING_BULL", "CHOPPY"],
    avoid_market_regime=["TRENDING_BEAR", "DISTRIBUTION"],
    ideal_volatility_regime=["LOW_VOL_COMPRESSION", "NORMAL"],
    ideal_breadth=["STRONG", "NEUTRAL"],
    ideal_institutional_activity=["ACCUMULATION", "QUIET_BUYING"],

    behavioral_conditions=[
        "Tight sideways range for 6+ weeks with volume declining — smart money absorbing supply",
        "No impulsive down days; pullbacks recover quickly with narrow range closes",
        "Strong close above the range ceiling on 2×+ base average volume",
        "Clear absence of overhead resistance for at least 15% above the breakout level",
    ],
    setup_conditions=[
        "Price range during base < 12% (high-to-low of entire base)",
        "Volume declining consistently over the 6-week base period",
        "Breakout week volume > 2× average volume of the base period",
        "No significant overhead supply within 15% of pivot",
        "Institutional ownership changes positive in most recent quarterly data",
    ],
    failure_characteristics=[
        "Breakout occurs on single-day volume spike with no follow-through",
        "Price quickly retreats into the range (false breakout or shakeout)",
        "Volume expansion was event-driven (block deal, news) rather than organic",
        "Macro event reverses sentiment before the stock can gain altitude",
    ],

    entry_trigger="Buy on first close above range ceiling with volume > 2× base average",
    stop_logic="Below the midpoint of the base; hard stop at base low",
    target_logic="1.5–2× base range projected from breakout level",
    trail_logic="Trail with 10EMA; hold through minor pullbacks that close above EMA",

    baseline_win_rate=_ab_wr,
    baseline_avg_win_pct=_ab_aw,
    baseline_avg_loss_pct=_ab_al,
    baseline_expectancy=round(_ab_wr * _ab_aw - (1 - _ab_wr) * _ab_al, 4),
    baseline_avg_hold_days=21,
    baseline_risk_reward=round(_ab_aw / _ab_al, 2),

    liquidity_requirement="Min 50Cr daily turnover",
    ideal_sector_conditions="Sector just starting its outperformance cycle vs Nifty",
    notes="Wyckoff accumulation is most reliable when the stock is in a sector with a "
          "clear fundamental catalyst (policy change, commodity cycle, rate cycle).",
))

# 5 ─ EARNINGS_CONTINUATION ─────────────────────────────────────────────────

_ec_wr = 0.63
_ec_aw = 0.09
_ec_al = 0.04

EARNINGS_CONTINUATION = _register(Playbook(
    id="EARNINGS_CONTINUATION",
    name="Earnings Gap Continuation",
    category="CONTINUATION",
    emoji="📈",

    ideal_market_regime=["TRENDING_BULL", "EXPANSION", "CHOPPY", "COMPRESSION"],
    avoid_market_regime=["TRENDING_BEAR"],
    ideal_volatility_regime=["NORMAL", "EXPANDING"],
    ideal_breadth=["STRONG", "NEUTRAL"],
    ideal_institutional_activity=["RISK_ON", "ACCUMULATION"],

    behavioral_conditions=[
        "Stock gaps up 4%+ on earnings; institutions buying the news, not selling",
        "Gap holds above gap-open level for 3 consecutive sessions — supply absorbed",
        "Pullback to gap top forms a tight flag or consolidation, not a breakdown",
        "Overall market is not in a downtrend that would overwhelm stock-specific strength",
    ],
    setup_conditions=[
        "Earnings gap > 4% on gap-open basis",
        "Stock holds above gap-open price for at least 3 trading sessions",
        "Gap day volume > 3× 50-day average volume",
        "RSI on gap day between 50 and 75 (not entering overbought exhaustion zone)",
        "EPS and revenue both beat estimates; guidance raised if possible",
    ],
    failure_characteristics=[
        "Gap fills within 5 sessions — institutional selling into the gap",
        "Sector-wide rotation out of the group overwhelms stock strength",
        "Second-quarter earnings reveal one-time nature of the beat",
        "Broad market sharp decline coincides with entry period",
    ],

    entry_trigger="Buy on pullback to gap-open price that holds on intraday basis, or on 3rd session close above gap",
    stop_logic="Below gap-open price (gap fill = trade invalidation); typically 3–5% from entry",
    target_logic="1.5–2× the gap size projected from gap-open level",
    trail_logic="Use 5EMA on daily chart for tight earnings-play trail; exit on close below 5EMA",

    baseline_win_rate=_ec_wr,
    baseline_avg_win_pct=_ec_aw,
    baseline_avg_loss_pct=_ec_al,
    baseline_expectancy=round(_ec_wr * _ec_aw - (1 - _ec_wr) * _ec_al, 4),
    baseline_avg_hold_days=8,
    baseline_risk_reward=round(_ec_aw / _ec_al, 2),

    liquidity_requirement="Min 100Cr daily turnover; F&O preferred for gap risk management",
    ideal_sector_conditions="Results season with positive sector-level surprises building",
    notes="NSE earnings are concentrated in quarterly results cycles (Apr–May, Jul–Aug, "
          "Oct–Nov, Jan–Feb). Trade density in this playbook spikes during those windows.",
))

# 6 ─ FAILED_BREAKOUT_REVERSAL ──────────────────────────────────────────────

_fb_wr = 0.55
_fb_aw = 0.07
_fb_al = 0.035

FAILED_BREAKOUT_REVERSAL = _register(Playbook(
    id="FAILED_BREAKOUT_REVERSAL",
    name="Failed Breakout Reversal (Fade)",
    category="REVERSAL",
    emoji="🔄",

    ideal_market_regime=["TRENDING_BEAR", "DISTRIBUTION", "CHOPPY"],
    avoid_market_regime=["TRENDING_BULL", "EXPANSION"],
    ideal_volatility_regime=["NORMAL", "EXPANDING", "HIGH_VOL"],
    ideal_breadth=["WEAK", "DETERIORATING"],
    ideal_institutional_activity=["DISTRIBUTION", "RISK_OFF"],

    behavioral_conditions=[
        "Stock broke out above a key pivot but failed to follow through with volume",
        "Price returns below pivot within 5 sessions — trapped bulls begin to exit",
        "Short sellers pressing as stop-loss selling accelerates",
        "Broad market provides no tailwind; sector is lagging",
    ],
    setup_conditions=[
        "Stock made a valid breakout above pivot within the last 5 trading days",
        "Price closes back below pivot (breakout failure confirmed)",
        "Volume declining on any recovery attempts above pivot",
        "RSI divergence: price made higher high on breakout, RSI did not",
        "Market breadth weak: advance-decline ratio < 1 on breakout day",
    ],
    failure_characteristics=[
        "Broad market reverses and powers the original breakout direction",
        "Fundamental catalyst emerges post-entry supporting the bullish case",
        "Heavy short interest creates a violent short squeeze",
        "Circuit breaker on NSE prevents timely exit",
    ],

    entry_trigger="Short (or exit long) on close below breakout pivot; confirm with intraday rejection at pivot on re-test",
    stop_logic="Above the failed breakout high; typically 3–5% above entry",
    target_logic="Target prior base low; secondary target at SMA200 level",
    trail_logic="Trail with 5EMA on intraday chart; cover on any daily close above 10EMA",

    baseline_win_rate=_fb_wr,
    baseline_avg_win_pct=_fb_aw,
    baseline_avg_loss_pct=_fb_al,
    baseline_expectancy=round(_fb_wr * _fb_aw - (1 - _fb_wr) * _fb_al, 4),
    baseline_avg_hold_days=5,
    baseline_risk_reward=round(_fb_aw / _fb_al, 2),

    liquidity_requirement="Min 100Cr daily turnover; F&O required for short exposure on NSE",
    ideal_sector_conditions="Sector in distribution phase or underperforming Nifty for 4+ weeks",
    notes="This is a short-side or exit playbook. On NSE cash segment, can only be "
          "implemented as an exit-long; F&O required for actual short exposure.",
))

# 7 ─ MEAN_REVERSION ────────────────────────────────────────────────────────

_mr_wr = 0.58
_mr_aw = 0.08
_mr_al = 0.045

MEAN_REVERSION = _register(Playbook(
    id="MEAN_REVERSION",
    name="Mean Reversion from Oversold",
    category="REVERSAL",
    emoji="↩️",

    ideal_market_regime=["CHOPPY", "COMPRESSION"],
    avoid_market_regime=["TRENDING_BEAR", "DISTRIBUTION"],
    ideal_volatility_regime=["HIGH_VOL", "EXPANDING"],
    ideal_breadth=["NEUTRAL", "RECOVERING"],
    ideal_institutional_activity=["NEUTRAL", "ACCUMULATION"],

    behavioral_conditions=[
        "Stock falls 15–25% from its 52-week high in 5–10 trading days",
        "Still in a Stage 2 uptrend on weekly chart — longer-term structure intact",
        "RSI reaches oversold territory; volume surges suggest exhaustion selling",
        "Clear support zone visible (prior base, SMA200, round number level)",
    ],
    setup_conditions=[
        "RSI < 30 on daily chart (oversold)",
        "Price > SMA200 (Stage 2 uptrend intact on higher timeframe)",
        "Price below lower Bollinger Band (2σ)",
        "Volume on down-move > 2× average (exhaustion / panic selling)",
        "Drop from 52-week high between 15% and 30% within 10 trading days",
    ],
    failure_characteristics=[
        "Broad market enters a primary downtrend — mean reversion becomes trend following down",
        "Fundamental deterioration revealed (earnings warning, regulatory issue)",
        "RSI reaches oversold but price continues lower for weeks (Stage 3/4 transition)",
        "Support zone breaks cleanly, accelerating selling",
    ],

    entry_trigger="Buy on first daily close above 5EMA after oversold RSI reading, or hammer/doji at support with volume",
    stop_logic="Below the low of the exhaustion candle or below SMA200; max 5% from entry",
    target_logic="Target SMA20 or SMA50 reversion; typically 6–10% gain",
    trail_logic="Exit at first sign of failure to reclaim SMA20; no complex trail needed for short-duration trade",

    baseline_win_rate=_mr_wr,
    baseline_avg_win_pct=_mr_aw,
    baseline_avg_loss_pct=_mr_al,
    baseline_expectancy=round(_mr_wr * _mr_aw - (1 - _mr_wr) * _mr_al, 4),
    baseline_avg_hold_days=10,
    baseline_risk_reward=round(_mr_aw / _mr_al, 2),

    liquidity_requirement="Any NSE EQ stock with > 20Cr daily turnover",
    ideal_sector_conditions="Sector not in primary downtrend; the decline is stock-specific or sector-rotation-driven",
    notes="NEVER trade mean reversion in a trending bear market — mean reversion becomes "
          "a dead-cat-bounce trap. Confirm Stage 2 on weekly before entry.",
))

# 8 ─ TREND_CONTINUATION ────────────────────────────────────────────────────

_tc_wr = 0.65
_tc_aw = 0.11
_tc_al = 0.04

TREND_CONTINUATION = _register(Playbook(
    id="TREND_CONTINUATION",
    name="Trend Continuation (Pullback to Key MA)",
    category="CONTINUATION",
    emoji="📉➡️📈",

    ideal_market_regime=["TRENDING_BULL"],
    avoid_market_regime=["TRENDING_BEAR", "DISTRIBUTION", "CHOPPY_BEAR"],
    ideal_volatility_regime=["LOW_VOL_COMPRESSION", "NORMAL"],
    ideal_breadth=["STRONG", "NEUTRAL"],
    ideal_institutional_activity=["ACCUMULATION", "RISK_ON"],

    behavioral_conditions=[
        "Healthy uptrending stock in Stage 2 with clean MA structure",
        "Pulls back 5–8% on declining volume — healthy profit-taking, not distribution",
        "Price finds support exactly at 10EMA or 21EMA level",
        "First day of re-acceleration shows volume picking up — buyers returning",
    ],
    setup_conditions=[
        "Price > SMA50 > SMA200 (Stage 2 bullish stack)",
        "Pullback is 5–8% from recent swing high",
        "Price tags the 10EMA or 21EMA on daily chart",
        "Volume below 20-day average during the entire pullback",
        "Pullback lasts 3–10 trading days (not multi-week base — that is a different setup)",
    ],
    failure_characteristics=[
        "Stock undercuts the MA on a high-volume close (distribution, not pullback)",
        "Broad market accelerates lower during the pullback period",
        "Prior uptrend was parabolic — correction turns into base repair",
        "Sector leadership changes; stock is no longer the strongest in its group",
    ],

    entry_trigger="Buy on first up-day after tagging 10EMA/21EMA with volume uptick, or on close back above 10EMA",
    stop_logic="Below the 21EMA or below the pullback low; typically 4–6% from entry",
    target_logic="Prior swing high + 50%; measured move extension of prior leg",
    trail_logic="Trail with 10EMA; accelerate to 5EMA once stock is up 10%+ from entry",

    baseline_win_rate=_tc_wr,
    baseline_avg_win_pct=_tc_aw,
    baseline_avg_loss_pct=_tc_al,
    baseline_expectancy=round(_tc_wr * _tc_aw - (1 - _tc_wr) * _tc_al, 4),
    baseline_avg_hold_days=14,
    baseline_risk_reward=round(_tc_aw / _tc_al, 2),

    liquidity_requirement="Min 50Cr daily turnover",
    ideal_sector_conditions="Sector in confirmed uptrend with institutional inflows documented",
    notes="Highest-probability playbook in a bull market. Works best on Nifty50 and "
          "Nifty Midcap100 constituents with strong institutional ownership.",
))

# 9 ─ HIGH_TIGHT_FLAG ───────────────────────────────────────────────────────

_ht_wr = 0.50
_ht_aw = 0.40
_ht_al = 0.10

HIGH_TIGHT_FLAG = _register(Playbook(
    id="HIGH_TIGHT_FLAG",
    name="High Tight Flag (Most Explosive)",
    category="BREAKOUT",
    emoji="🏴",

    ideal_market_regime=["TRENDING_BULL", "EXPANSION"],
    avoid_market_regime=["TRENDING_BEAR", "DISTRIBUTION", "CHOPPY"],
    ideal_volatility_regime=["LOW_VOL_COMPRESSION", "NORMAL"],
    ideal_breadth=["STRONG"],
    ideal_institutional_activity=["RISK_ON", "AGGRESSIVE_ACCUMULATION"],

    behavioral_conditions=[
        "Stock up 100%+ in fewer than 8 weeks — massive price discovery in progress",
        "Consolidates in a tight flag for 3–5 weeks with < 25% range",
        "Volume contracts sharply in the flag — profit-taking absorbed by new buyers",
        "Relative strength vs all peers: this stock is the #1 performer in its sector",
    ],
    setup_conditions=[
        "Prior move > 100% in < 60 calendar days (4 × the normal momentum threshold)",
        "Consolidation range during flag < 25% of flag high",
        "Volume contraction during flag: each flag week lower volume than the prior",
        "RS rank vs peers: top decile (top 10%) of all NSE EQ stocks",
        "Flag forms in upper half of the prior move — no round-trip of gains",
    ],
    failure_characteristics=[
        "Flag drifts below 50% of prior move — retracement too deep",
        "Volume expands on down days within the flag (distribution)",
        "Fundamental driver behind the move reverses (commodity price, policy)",
        "Broad market enters correction, preventing flag breakout follow-through",
        "Single-theme stock with no institutional coverage — illiquidity risk",
    ],

    entry_trigger="Buy on close above flag upper trendline on volume > 1.5× 10-day average",
    stop_logic="Below the flag midpoint; maximum 10% from entry given the explosive nature",
    target_logic="Flag pole height projected from flag breakout; target 30–50% gain",
    trail_logic="Weekly 10-week MA trail; this is a core position — hold through volatility",

    baseline_win_rate=_ht_wr,
    baseline_avg_win_pct=_ht_aw,
    baseline_avg_loss_pct=_ht_al,
    baseline_expectancy=round(_ht_wr * _ht_aw - (1 - _ht_wr) * _ht_al, 4),
    baseline_avg_hold_days=20,
    baseline_risk_reward=round(_ht_aw / _ht_al, 2),

    liquidity_requirement="Min 200Cr daily turnover; extreme price moves attract circuit breakers on thin stocks",
    ideal_sector_conditions="Emerging theme with catalysts: new policy, commodity super-cycle, technology disruption",
    notes="Rarest setup on NSE — perhaps 5–10 genuine high-tight flags per year across the entire market. "
          "When found, size it as a core position (not a trade). "
          "Expectancy is highest of all playbooks despite lower win rate due to massive avg win.",
))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_playbook(playbook_id: str) -> Playbook:
    """Return a Playbook by its string id.

    Raises KeyError with a helpful message if not found.
    """
    try:
        return REGISTRY[playbook_id]
    except KeyError:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown playbook '{playbook_id}'. Available: {available}"
        ) from None


def get_playbooks_for_regime(
    market_regime: str,
    vol_regime: str,
    breadth: str,
) -> list[Playbook]:
    """Return playbooks sorted by regime-alignment score (descending).

    Scoring:
      +3  market_regime in ideal_market_regime
      -3  market_regime in avoid_market_regime
      +2  vol_regime in ideal_volatility_regime
      +1  breadth in ideal_breadth
    Only playbooks with score > 0 are returned.
    """
    scored: list[tuple[int, Playbook]] = []

    for pb in REGISTRY.values():
        score = 0
        if market_regime in pb.ideal_market_regime:
            score += 3
        if market_regime in pb.avoid_market_regime:
            score -= 3
        if vol_regime in pb.ideal_volatility_regime:
            score += 2
        if breadth in pb.ideal_breadth:
            score += 1
        if score > 0:
            pb.regime_aligned = True
            scored.append((score, pb))
        else:
            pb.regime_aligned = False

    scored.sort(key=lambda x: x[0], reverse=True)
    return [pb for _, pb in scored]


__all__ = [
    "Playbook",
    "REGISTRY",
    "VCP_BREAKOUT",
    "MOMENTUM_EXPANSION",
    "EARLY_LEADER",
    "ACCUMULATION_BREAKOUT",
    "EARNINGS_CONTINUATION",
    "FAILED_BREAKOUT_REVERSAL",
    "MEAN_REVERSION",
    "TREND_CONTINUATION",
    "HIGH_TIGHT_FLAG",
    "get_playbook",
    "get_playbooks_for_regime",
]
