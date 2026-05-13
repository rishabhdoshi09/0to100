"""
Scan Pipeline — Orchestrator for the 8-stage institutional scan.

Single entry point: ScanPipeline.run(universe) -> list[RankedSetup]

Stages:
  1. Universe Filter    — liquidity hard-filter
  2. Regime Filter      — trend-stage filter based on market regime
  3. Setup Detection    — behavioral archetype detection
  4. Quality Scoring    — 10-factor institutional scoring
  5. Playbook Matching  — regime-aware playbook alignment
  6. Expectancy         — EV computation per setup+regime
  7. Ranking            — final sort by regime-adjusted score
  8. Alert Check        — fire Telegram for Elite setups (optional)
"""
from __future__ import annotations

import time
from typing import Optional

from logger import get_logger

log = get_logger(__name__)


class ScanPipeline:
    """
    The unified regime-aware scan pipeline.
    Replaces the 3 legacy scanners as the single source of setup discovery.
    """

    def __init__(
        self,
        max_workers: int = 16,
        min_quality_score: float = 42.0,
        skip_avoid_tier: bool = True,
        fire_alerts: bool = False,
    ):
        self._max_workers       = max_workers
        self._min_quality       = min_quality_score
        self._skip_avoid        = skip_avoid_tier
        self._fire_alerts       = fire_alerts

    def run(
        self,
        universe: list[str],
        top_n: int = 30,
        regime_state=None,          # RegimeState from core.regime_engine
        skip_liquidity_filter: bool = False,
    ) -> list:                       # list[RankedSetup]

        t0 = time.time()
        log.info("pipeline_start", universe_size=len(universe))

        # ── 0. Get regime ─────────────────────────────────────────────────────
        if regime_state is None:
            regime_state = self._get_regime()

        market_regime  = getattr(regime_state, "market_regime",  "CHOPPY")
        quality_mult   = getattr(regime_state, "quality_multiplier", 1.0)
        leading_sectors = getattr(regime_state, "leading_sectors", [])
        breadth        = getattr(regime_state, "breadth_label", "NEUTRAL")
        inst_activity  = getattr(regime_state, "institutional_activity", "NEUTRAL")
        vol_regime     = getattr(regime_state, "volatility_regime", "NORMAL")

        log.info("regime_loaded", regime=market_regime, quality_mult=quality_mult)

        # ── 1. Universe Filter ────────────────────────────────────────────────
        if skip_liquidity_filter or len(universe) <= 30:
            liquid = universe
            log.info("liquidity_filter_skipped", count=len(liquid))
        else:
            from scan.universe_filter import UniverseFilter
            liquid = UniverseFilter(max_workers=self._max_workers).filter(universe)
            log.info("liquidity_filter_done", passed=len(liquid), rejected=len(universe)-len(liquid))

        if not liquid:
            return []

        # ── 2. Regime Filter ──────────────────────────────────────────────────
        # Only apply regime stage filter for large universes
        if len(liquid) > 50:
            from scan.regime_filter import RegimeFilter
            regime_filtered = RegimeFilter(market_regime, max_workers=self._max_workers).filter(liquid)
            log.info("regime_filter_done", passed=len(regime_filtered))
        else:
            regime_filtered = liquid

        if not regime_filtered:
            return []

        # ── 3. Setup Detection ────────────────────────────────────────────────
        from scan.setup_engine import SetupEngine
        candidates = SetupEngine(max_workers=self._max_workers).detect(regime_filtered)
        log.info("setup_detection_done", candidates=len(candidates))

        if not candidates:
            return []

        # ── 4. Quality Scoring ────────────────────────────────────────────────
        from scan.quality_engine import QualityEngine
        qe = QualityEngine(
            leading_sectors=leading_sectors,
            breadth_label=breadth,
            market_regime=market_regime,
            institutional_activity=inst_activity,
        )
        quality_scores = [qe.score(c) for c in candidates]

        # Filter by minimum score
        if self._skip_avoid:
            candidates    = [c for c, q in zip(candidates, quality_scores) if q.tier != "AVOID"]
            quality_scores = [q for q in quality_scores if q.tier != "AVOID"]

        candidates    = [c for c, q in zip(candidates, quality_scores) if q.score >= self._min_quality]
        quality_scores = [q for q in quality_scores if q.score >= self._min_quality]

        log.info("quality_filter_done", passed=len(candidates))

        if not candidates:
            return []

        # ── 5-7. Playbook match + Expectancy + Ranking ────────────────────────
        from scan.ranking_engine import RankingEngine
        ranked = RankingEngine(
            regime_state=regime_state, quality_multiplier=quality_mult
        ).rank(candidates, quality_scores)

        result = ranked[:top_n]
        elapsed = round(time.time() - t0, 1)
        log.info("pipeline_done", results=len(result), elapsed_s=elapsed)

        # ── 8. Alerts (optional) ──────────────────────────────────────────────
        if self._fire_alerts:
            self._fire_elite_alerts(result)

        return result

    def _get_regime(self):
        try:
            from core.regime_engine import compute_regime
            return compute_regime()
        except Exception as exc:
            log.debug("regime_unavailable", error=str(exc))
            return None

    def _fire_elite_alerts(self, setups: list) -> None:
        try:
            from alerts.telegram_alerts import AlertEngine
            if not AlertEngine().is_configured():
                return
            import requests
            import os
            token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            if not token or not chat_id:
                return
            elite = [s for s in setups if s.quality_tier == "ELITE_A_PLUS"]
            if not elite:
                return
            lines = ["🏛️ <b>QUANTTERM — Elite Setups</b>"]
            for s in elite[:5]:
                ev = f"+{s.expected_value_r:.1f}R" if s.expected_value_r >= 0 else f"{s.expected_value_r:.1f}R"
                lines.append(
                    f"\n⭐ <b>{s.symbol}</b>  {s.archetype.replace('_',' ')}\n"
                    f"  Price: ₹{s.price:,.0f}  Pivot: ₹{s.pivot_level:,.0f}\n"
                    f"  Quality: {s.quality_tier.replace('_',' ')} ({s.quality_score:.0f})  EV: {ev}\n"
                    f"  Regime: {s.regime_alignment}"
                )
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception:
            pass
