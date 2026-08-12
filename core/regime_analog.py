"""
core/regime_analog.py
Historical regime pattern matcher for QUANTTERM.
Finds past NSE market periods similar to today's conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegimeAnalog:
    period: str                  # e.g. "Jul 2020"
    similarity_score: float      # 0-100, higher = more similar
    regime: str                  # market regime during that period
    vix_then: float
    what_worked: list[str]       # 3 items
    what_failed: list[str]       # 3 items
    outcome_summary: str         # 1-2 sentences


# ---------------------------------------------------------------------------
# Hardcoded historical period library (NSE, 2019-2024)
# ---------------------------------------------------------------------------

_PERIOD_LIBRARY: list[dict[str, Any]] = [
    {
        "period": "Jul-Aug 2019",
        "market_regime": "CHOPPY",
        "volatility_regime": "ELEVATED",
        "breadth_label": "WEAK",
        "vix": 18.0,
        "nifty_change_5d_approx": -1.2,
        "sector_leaders": ["IT"],
        "what_worked": [
            "IT defensive positioning avg +4.1R over 3 weeks",
            "Selling OTM calls on index spikes — premium decay reliable",
            "Short duration bond exposure during equity weakness",
        ],
        "what_failed": [
            "Bank/NBFC longs crushed by liquidity fears",
            "Momentum breakouts failed — no follow-through in choppy tape",
            "Small-cap longs hit by sustained FII selling pressure",
        ],
        "outcome_summary": (
            "Nifty consolidated in a wide range with sharp intra-week swings driven by "
            "global trade war fears and domestic NBFC stress. IT outperformed as a defensive "
            "sector while the broader market drifted lower."
        ),
    },
    {
        "period": "Mar 2020",
        "market_regime": "TRENDING_BEAR",
        "volatility_regime": "PANIC",
        "breadth_label": "WEAK",
        "vix": 80.0,
        "nifty_change_5d_approx": -18.0,
        "sector_leaders": [],
        "what_worked": [
            "Cash and short exposure — everything else was a trap",
            "Buying deep-ITM puts as portfolio hedge avg +15R over 2 weeks",
            "Gold ETF allocation as flight-to-safety trade",
        ],
        "what_failed": [
            "Buying dips too early — bear market bounces were brief and violent",
            "Selling volatility — VIX spikes made short-premium strategies lethal",
            "Sector rotation into defensives — selling was indiscriminate",
        ],
        "outcome_summary": (
            "COVID-19 shock caused one of the fastest bear markets in NSE history, with Nifty "
            "losing ~38% peak-to-trough in weeks. Liquidity evaporated and virtually no long "
            "equity strategy survived without hedging."
        ),
    },
    {
        "period": "May-Jun 2020",
        "market_regime": "EXPANSION",
        "volatility_regime": "NORMAL",
        "breadth_label": "NEUTRAL",
        "vix": 26.0,
        "nifty_change_5d_approx": 3.5,
        "sector_leaders": ["METAL", "AUTO"],
        "what_worked": [
            "Metal and auto sector recovery longs avg +6.8R over 4 weeks",
            "Breakout buying on volume — early recovery leaders ran hard",
            "Covered calls on recovery positions to reduce cost basis",
        ],
        "what_failed": [
            "Re-shorting into the bounce — underestimated central bank support",
            "FMCG and pharma longs lagged as risk-on rotation accelerated",
            "Waiting for all-clear signal — missed first 15% of the move",
        ],
        "outcome_summary": (
            "Markets began recovering sharply from COVID lows driven by RBI liquidity measures "
            "and fiscal stimulus expectations. Cyclicals and beaten-down sectors led while "
            "defensives underperformed the reopening trade."
        ),
    },
    {
        "period": "Jul-Oct 2020",
        "market_regime": "TRENDING_BULL",
        "volatility_regime": "NORMAL",
        "breadth_label": "STRONG",
        "vix": 22.0,
        "nifty_change_5d_approx": 2.1,
        "sector_leaders": ["IT", "PHARMA"],
        "what_worked": [
            "IT and pharma momentum breakouts avg +8.2R over 3 weeks",
            "Trend-following systems — ADX > 25 setups had exceptional hit rate",
            "Buying pullbacks to 20-day EMA in leading sectors",
        ],
        "what_failed": [
            "PSU banks — credit stress kept them range-bound despite broad bull run",
            "Value traps in telecom and utilities — no re-rating catalyst",
            "Shorting overbought leaders — strong trends stayed overbought for months",
        ],
        "outcome_summary": (
            "A powerful post-COVID recovery rally with IT and pharma as secular winners on "
            "WFH and healthcare demand themes. Breadth was strong and trend-following strategies "
            "delivered exceptional returns with minimal drawdown."
        ),
    },
    {
        "period": "Jan-Feb 2021",
        "market_regime": "TRENDING_BULL",
        "volatility_regime": "LOW_VOL_COMPRESSION",
        "breadth_label": "STRONG",
        "vix": 20.0,
        "nifty_change_5d_approx": 1.8,
        "sector_leaders": ["METAL", "BANK"],
        "what_worked": [
            "Bank and metal rotational breakouts avg +7.5R over 4 weeks",
            "Budget anticipation plays in infra and capex themes",
            "Bull call spreads — low vol made options buying cheap",
        ],
        "what_failed": [
            "IT consolidation — sector paused after massive 2020 run",
            "Defensive sector longs — market rewarded risk-on names exclusively",
            "Taking profits too early in cyclicals — the move extended much further",
        ],
        "outcome_summary": (
            "Pre-budget optimism drove a broad bull market with banks and metals rotating into "
            "leadership. Low volatility enabled trend-following entries with tight stops and the "
            "Union Budget provided further fuel for cyclical re-rating."
        ),
    },
    {
        "period": "May 2021",
        "market_regime": "CHOPPY",
        "volatility_regime": "ELEVATED",
        "breadth_label": "NEUTRAL",
        "vix": 23.0,
        "nifty_change_5d_approx": -0.8,
        "sector_leaders": ["PHARMA", "FMCG"],
        "what_worked": [
            "Pharma and FMCG defensive rotation avg +3.2R over 3 weeks",
            "Iron condor strategies on index — range-bound tape suited premium selling",
            "Reducing position size and waiting for clarity",
        ],
        "what_failed": [
            "Cyclical momentum plays — second COVID wave dented recovery optimism",
            "Banking longs — NPA fears resurfaced with lockdown 2.0",
            "Breakout trades — false breakouts dominated the choppy environment",
        ],
        "outcome_summary": (
            "India's deadly second COVID wave created uncertainty and triggered defensive "
            "rotation away from cyclicals. The market held up better than feared at the index "
            "level but breadth deteriorated significantly beneath the surface."
        ),
    },
    {
        "period": "Oct-Nov 2021",
        "market_regime": "TRENDING_BULL",
        "volatility_regime": "NORMAL",
        "breadth_label": "STRONG",
        "vix": 17.0,
        "nifty_change_5d_approx": 1.5,
        "sector_leaders": ["BANK", "REALTY", "METAL"],
        "what_worked": [
            "Broad market momentum — almost every sector participated",
            "Small and mid-cap breakout strategies avg +9.4R over 5 weeks",
            "Sector rotation from IT into cyclicals captured significant alpha",
        ],
        "what_failed": [
            "Hedging with puts — low vol and strong trend made protection expensive",
            "Staying concentrated in IT — sector lagged the broader rally",
            "Waiting for deeper pullbacks — the market rarely gave them",
        ],
        "outcome_summary": (
            "Nifty made all-time highs in a broad-based rally with strong participation across "
            "market caps and sectors. Low volatility and improving earnings created ideal "
            "conditions for trend-following and breakout strategies."
        ),
    },
    {
        "period": "Jan-Feb 2022",
        "market_regime": "DISTRIBUTION",
        "volatility_regime": "ELEVATED",
        "breadth_label": "WEAK",
        "vix": 23.0,
        "nifty_change_5d_approx": -2.5,
        "sector_leaders": ["IT"],
        "what_worked": [
            "Reducing equity exposure and raising cash — capital preservation",
            "Short positions in high-beta midcap names avg +5.1R over 3 weeks",
            "Long volatility via put spreads as distribution accelerated",
        ],
        "what_failed": [
            "Dip-buying — distribution phases punish early buyers repeatedly",
            "FII-heavy names — sustained foreign selling crushed liquidity",
            "Smallcap longs — worst hit as institutions trimmed risk",
        ],
        "outcome_summary": (
            "FII outflows driven by Fed rate hike expectations triggered distribution in "
            "Indian markets. Breadth deteriorated sharply while the index held deceptively "
            "well, masking the damage in broader market."
        ),
    },
    {
        "period": "Jun 2022",
        "market_regime": "TRENDING_BEAR",
        "volatility_regime": "ELEVATED",
        "breadth_label": "WEAK",
        "vix": 22.0,
        "nifty_change_5d_approx": -4.2,
        "sector_leaders": [],
        "what_worked": [
            "Cash and short exposure — trend was decisively down",
            "Defensive sectors (IT services, pharma) held better than cyclicals",
            "Systematic stop-loss adherence prevented catastrophic drawdowns",
        ],
        "what_failed": [
            "Metal and energy longs — commodity supercycle narrative collapsed abruptly",
            "Global cyclical plays — synchronized tightening hit all risk assets",
            "Averaging down into falling positions",
        ],
        "outcome_summary": (
            "Global rate hike fears and commodity price collapse drove a sharp correction in "
            "NSE markets. Metal and energy stocks that had led 2021-22 suffered the steepest "
            "declines as the macro narrative reversed quickly."
        ),
    },
    {
        "period": "Jul-Sep 2022",
        "market_regime": "EXPANSION",
        "volatility_regime": "NORMAL",
        "breadth_label": "NEUTRAL",
        "vix": 18.0,
        "nifty_change_5d_approx": 2.8,
        "sector_leaders": ["AUTO", "BANK"],
        "what_worked": [
            "Auto sector breakouts on improving volume data avg +6.2R over 4 weeks",
            "Banking recovery longs as credit cycle turned positive",
            "Buying breakouts from multi-month bases — clean technical setups",
        ],
        "what_failed": [
            "IT sector re-entry too early — FAANG layoffs dented sentiment",
            "Shorting the recovery — bear market rally became genuine uptrend",
            "Commodity plays — metals remained under pressure",
        ],
        "outcome_summary": (
            "NSE staged a sharp recovery from June lows as macro fears moderated and "
            "domestic consumption data surprised positively. Auto and banking led the "
            "expansion phase as earnings remained resilient."
        ),
    },
    {
        "period": "Nov 2022-Jan 2023",
        "market_regime": "TRENDING_BULL",
        "volatility_regime": "LOW_VOL_COMPRESSION",
        "breadth_label": "STRONG",
        "vix": 13.0,
        "nifty_change_5d_approx": 1.2,
        "sector_leaders": ["CAPEX", "INFRA", "BANK"],
        "what_worked": [
            "Capital goods and infrastructure theme avg +11.3R over 6 weeks",
            "PSU capex plays — government spending acceleration re-rated the sector",
            "Low-vol breakout strategies with tight stops and large targets",
        ],
        "what_failed": [
            "Export-oriented IT longs — global slowdown fears weighed",
            "Consumer discretionary — urban demand signals were mixed",
            "Shorting the bull run — low volatility grinded bears down",
        ],
        "outcome_summary": (
            "A powerful low-volatility bull market driven by India's infrastructure push and "
            "capex cycle. Government spending on railways, defence, and roads re-rated PSU "
            "names while the broader market made steady all-time highs."
        ),
    },
    {
        "period": "Mar-Apr 2023",
        "market_regime": "CHOPPY",
        "volatility_regime": "NORMAL",
        "breadth_label": "NEUTRAL",
        "vix": 13.0,
        "nifty_change_5d_approx": 0.3,
        "sector_leaders": ["PHARMA"],
        "what_worked": [
            "Mean reversion strategies — range-bound tape suited buy-dip sell-rip",
            "Pharma sector plays as sector rotated defensively",
            "Covered call writing on index holdings — theta decay in flat market",
        ],
        "what_failed": [
            "Breakout buying — false breakouts were frequent in sideways tape",
            "High-beta momentum plays — no follow-through on moves",
            "Adani group exposure — corporate governance concerns caused sharp drops",
        ],
        "outcome_summary": (
            "Markets consolidated sideways following the Adani group selloff and global banking "
            "concerns. The index held key support levels but lacked directional conviction, "
            "making momentum strategies ineffective."
        ),
    },
    {
        "period": "Jun-Sep 2023",
        "market_regime": "TRENDING_BULL",
        "volatility_regime": "NORMAL",
        "breadth_label": "STRONG",
        "vix": 11.0,
        "nifty_change_5d_approx": 1.9,
        "sector_leaders": ["AUTO", "REALTY", "CAPEX"],
        "what_worked": [
            "Smallcap and midcap momentum plays avg +12.1R over 5 weeks",
            "Realty and auto breakouts rode strong domestic demand narrative",
            "Trend-following with 50-day EMA as dynamic support",
        ],
        "what_failed": [
            "Hedging via puts — extremely low VIX made protection very costly",
            "IT sector longs — Nasdaq weakness dragged Indian IT",
            "Waiting for corrections to enter — market offered very shallow pullbacks",
        ],
        "outcome_summary": (
            "India outperformed global markets with one of the strongest broad bull markets in "
            "recent history. Low volatility, strong FII inflows, and domestic growth drove "
            "smallcaps and midcaps to significant outperformance."
        ),
    },
    {
        "period": "Dec 2023-Jan 2024",
        "market_regime": "TRENDING_BULL",
        "volatility_regime": "LOW_VOL_COMPRESSION",
        "breadth_label": "STRONG",
        "vix": 12.0,
        "nifty_change_5d_approx": 1.5,
        "sector_leaders": ["SMALLCAP", "MIDCAP", "REALTY"],
        "what_worked": [
            "Smallcap euphoria — aggressive breakout buying averaged +15.2R over 6 weeks",
            "Realty theme with multi-year breakouts in housing demand stocks",
            "Momentum factor performed exceptionally — high RS names ran further",
        ],
        "what_failed": [
            "Defensive positioning — FMCG and pharma severely underperformed",
            "Shorting frothy smallcaps — momentum sustained far longer than expected",
            "Options premium selling — low VIX meant tiny credits with tail risk",
        ],
        "outcome_summary": (
            "Smallcap euphoria reached extreme levels with retail participation surging. "
            "Valuations stretched but momentum sustained, rewarding trend-followers while "
            "punishing value-oriented and defensive strategies."
        ),
    },
    {
        "period": "Mar-Apr 2024",
        "market_regime": "DISTRIBUTION",
        "volatility_regime": "ELEVATED",
        "breadth_label": "WEAK",
        "vix": 16.0,
        "nifty_change_5d_approx": -1.8,
        "sector_leaders": ["IT", "PHARMA"],
        "what_worked": [
            "Trimming high-beta positions ahead of election uncertainty",
            "IT defensive positioning as sector held relative strength",
            "Cash accumulation — patience rewarded with better entries post-correction",
        ],
        "what_failed": [
            "Smallcap and midcap longs — sharp correction from euphoric levels",
            "PSU names — profit-taking after massive run-up hit sector hard",
            "Breakout trades — elevated uncertainty caused multiple false starts",
        ],
        "outcome_summary": (
            "Pre-election uncertainty triggered distribution as FIIs booked profits from the "
            "smallcap and midcap euphoria. Breadth weakened significantly even as large-cap "
            "index held up, masking the damage to broader portfolios."
        ),
    },
    {
        "period": "Jul-Sep 2024",
        "market_regime": "TRENDING_BULL",
        "volatility_regime": "NORMAL",
        "breadth_label": "STRONG",
        "vix": 14.0,
        "nifty_change_5d_approx": 2.2,
        "sector_leaders": ["BANK", "AUTO", "REALTY"],
        "what_worked": [
            "Post-election policy clarity drove banking and cyclical breakouts avg +8.7R",
            "Infrastructure and defence themes re-accelerated on budget clarity",
            "Buying the dip after pre-election selloff — mean reversion worked perfectly",
        ],
        "what_failed": [
            "IT sector long-term bulls — rupee appreciation and deal softness weighed",
            "Global commodity plays — China slowdown kept metals subdued",
            "Excessive hedging post-election — the bull run resumed faster than expected",
        ],
        "outcome_summary": (
            "Post-election clarity triggered a powerful resumption of the bull market with "
            "banking and cyclicals leading. Policy continuity and strong domestic flows drove "
            "Nifty to new all-time highs with improving breadth."
        ),
    },
]


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------

_REGIME_WEIGHTS = {
    "market_regime": 0.40,
    "volatility_regime": 0.20,
    "breadth_label": 0.20,
    "vix_range": 0.20,
}

_VIX_BUCKETS = [
    (0, 13, "very_low"),
    (13, 17, "low"),
    (17, 22, "normal"),
    (22, 30, "elevated"),
    (30, 999, "high"),
]


def _vix_bucket(vix: float) -> str:
    for lo, hi, label in _VIX_BUCKETS:
        if lo <= vix < hi:
            return label
    return "high"


def _compute_similarity(regime_state: dict, period: dict) -> float:
    """
    Compute 0-100 similarity between current regime_state and a historical period.
    Weights: market_regime=40%, volatility_regime=20%, breadth_label=20%, VIX bucket=20%.
    """
    score = 0.0

    # market_regime (40%)
    if regime_state.get("market_regime") == period["market_regime"]:
        score += 40.0

    # volatility_regime (20%)
    if regime_state.get("volatility_regime") == period["volatility_regime"]:
        score += 20.0

    # breadth_label (20%)
    if regime_state.get("breadth_label") == period["breadth_label"]:
        score += 20.0

    # VIX range (20%) — bucket match
    current_vix = regime_state.get("vix", 0.0)
    if isinstance(current_vix, (int, float)):
        if _vix_bucket(current_vix) == _vix_bucket(period["vix"]):
            score += 20.0
        else:
            # Partial credit for close buckets
            diff = abs(current_vix - period["vix"])
            if diff <= 5:
                score += 10.0

    return round(score, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_analogs(regime_state: dict, top_n: int = 3) -> list[RegimeAnalog]:
    """
    Find historical NSE market periods most similar to the current regime_state.

    Parameters
    ----------
    regime_state : dict
        Keys: market_regime, volatility_regime, breadth_label, vix,
              nifty_change_5d, sector_returns
    top_n : int
        Number of analogs to return (default 3)

    Returns
    -------
    list[RegimeAnalog]
        Sorted by similarity_score descending.
    """
    scored: list[tuple[float, dict]] = []
    for period in _PERIOD_LIBRARY:
        sim = _compute_similarity(regime_state, period)
        scored.append((sim, period))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    analogs = []
    for sim_score, p in top:
        analogs.append(
            RegimeAnalog(
                period=p["period"],
                similarity_score=sim_score,
                regime=p["market_regime"],
                vix_then=p["vix"],
                what_worked=p["what_worked"],
                what_failed=p["what_failed"],
                outcome_summary=p["outcome_summary"],
            )
        )
    return analogs


def format_analog_insight(analogs: list[RegimeAnalog]) -> str:
    """
    Format a list of RegimeAnalog objects into a multi-line string
    suitable for JARVIS chat display.
    """
    if not analogs:
        return "No historical analogs found."

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  HISTORICAL REGIME ANALOGS")
    lines.append("=" * 60)

    for i, analog in enumerate(analogs, 1):
        lines.append(f"\n#{i}  {analog.period}   [Similarity: {analog.similarity_score:.0f}/100]")
        lines.append(f"    Regime : {analog.regime}   |   VIX then: {analog.vix_then:.1f}")
        lines.append("")
        lines.append("    What WORKED:")
        for item in analog.what_worked:
            lines.append(f"      + {item}")
        lines.append("")
        lines.append("    What FAILED:")
        for item in analog.what_failed:
            lines.append(f"      - {item}")
        lines.append("")
        lines.append("    Outcome:")
        # Word-wrap outcome_summary at ~70 chars
        words = analog.outcome_summary.split()
        current_line = "      "
        for word in words:
            if len(current_line) + len(word) + 1 > 72:
                lines.append(current_line.rstrip())
                current_line = "      " + word + " "
            else:
                current_line += word + " "
        if current_line.strip():
            lines.append(current_line.rstrip())
        lines.append("-" * 60)

    return "\n".join(lines)
