"""
Ranking Engine — Final stage of the scan pipeline.

Takes quality-scored setup candidates and produces a final ranked
list with playbook match, expected value, and institutional summary.

Output is a list of RankedSetup — the single object consumed by every
UI panel, alert system, and position sizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RankedSetup:
    # Identity
    symbol: str
    archetype: str              # SetupEngine archetype id
    playbook_id: str            # matched playbook from registry

    # Pricing
    price: float
    pivot_level: float
    stop_level: float
    risk_pct: float             # (pivot - stop) / pivot × 100

    # Scoring
    quality_tier: str           # ELITE_A_PLUS | A | B | WATCHLIST | AVOID
    quality_score: float        # 0-100
    regime_adjusted_score: float

    # Expectancy
    expected_value_r: float     # e.g. +3.4 (in R units)
    historical_win_rate: float
    regime_alignment: str       # HIGH | MEDIUM | LOW
    failure_risk: str           # LOW | MODERATE | HIGH

    # Context
    behavioral_evidence: list[str]
    regime: str
    sector_leader: bool

    # Position sizing hint
    suggested_position_pct: float  # % of portfolio based on Kelly-lite

    # Institutional summary (for right panel)
    institutional_summary: str

    def rank_key(self) -> float:
        """Used for sorting. Higher = better."""
        tier_boost = {
            "ELITE_A_PLUS": 30, "A": 20, "B": 10, "WATCHLIST": 0, "AVOID": -50
        }.get(self.quality_tier, 0)
        return self.regime_adjusted_score + tier_boost + self.expected_value_r * 5


class RankingEngine:
    """
    Combines quality scores + playbook match + expectancy into final RankedSetup list.
    """

    def __init__(self, regime_state=None, quality_multiplier: float = 1.0):
        self._regime        = regime_state
        self._quality_mult  = quality_multiplier

    def rank(
        self,
        candidates,        # list[SetupCandidate]
        quality_scores,    # list[QualityScore]
        playbook_registry = None,
    ) -> list[RankedSetup]:
        from playbooks import get_playbooks_for_regime, REGISTRY

        regime_name = getattr(self._regime, "market_regime", "CHOPPY") if self._regime else "CHOPPY"
        vol_regime  = getattr(self._regime, "volatility_regime", "NORMAL") if self._regime else "NORMAL"
        breadth     = getattr(self._regime, "breadth_label", "NEUTRAL") if self._regime else "NEUTRAL"
        inst_act    = getattr(self._regime, "institutional_activity", "NEUTRAL") if self._regime else "NEUTRAL"

        # Build quality lookup
        quality_map = {qs.symbol: qs for qs in quality_scores}

        # Get regime-aligned playbooks
        regime_playbooks = get_playbooks_for_regime(regime_name, vol_regime, breadth)
        playbook_map     = {pb.id: pb for pb in regime_playbooks}

        # Match archetype → playbook
        _ARCHETYPE_TO_PLAYBOOK = {
            "VCP_BREAKOUT":           "VCP_BREAKOUT",
            "MOMENTUM_EXPANSION":     "MOMENTUM_EXPANSION",
            "EARLY_LEADER":           "EARLY_LEADER",
            "ACCUMULATION_BREAKOUT":  "ACCUMULATION_BREAKOUT",
            "EARNINGS_CONTINUATION":  "EARNINGS_CONTINUATION",
            "FAILED_BREAKOUT":        "FAILED_BREAKOUT_REVERSAL",
            "MEAN_REVERSION":         "MEAN_REVERSION",
            "TREND_CONTINUATION":     "TREND_CONTINUATION",
            "HIGH_TIGHT_FLAG":        "HIGH_TIGHT_FLAG",
        }

        results: list[RankedSetup] = []

        for cand in candidates:
            qs = quality_map.get(cand.symbol)
            if not qs or qs.tier == "AVOID":
                continue

            pb_id = _ARCHETYPE_TO_PLAYBOOK.get(cand.archetype, "VCP_BREAKOUT")
            pb    = REGISTRY.get(pb_id)
            if not pb:
                continue

            # Regime-adjusted score
            adj_score = min(100.0, qs.score * self._quality_mult)

            # Expected value (use live from ExpectancyEngine if available, else playbook baseline)
            ev_r = self._get_ev(pb_id, pb, qs.tier, regime_name)

            # Position sizing: 2% portfolio risk × Kelly fraction
            risk_pct = max(0.5, (cand.pivot_level - cand.stop_level) / cand.pivot_level * 100
                           if cand.pivot_level > 0 else 5.0)
            kelly    = (pb.baseline_win_rate * pb.baseline_risk_reward - (1 - pb.baseline_win_rate))
            half_kelly = max(0, kelly * 0.5)  # half-Kelly conservative sizing
            pos_pct  = min(10.0, round(half_kelly * 100 * (2.0 / max(risk_pct, 0.5)), 1))

            # Regime alignment label
            reg_aligned = pb_id in [r.id for r in regime_playbooks]
            reg_label   = "HIGH" if reg_aligned and adj_score >= 70 else ("MEDIUM" if reg_aligned else "LOW")

            # Failure risk
            fail_risk = "LOW" if qs.score >= 75 else ("MODERATE" if qs.score >= 55 else "HIGH")

            leading = getattr(self._regime, "leading_sectors", []) if self._regime else []
            sec_leader = bool(any(s in cand.symbol.upper() for s in leading))

            summary = self._format_summary(cand, qs, pb, ev_r, reg_label, fail_risk, regime_name)

            results.append(RankedSetup(
                symbol=cand.symbol,
                archetype=cand.archetype,
                playbook_id=pb_id,
                price=cand.price,
                pivot_level=cand.pivot_level,
                stop_level=cand.stop_level,
                risk_pct=round(risk_pct, 1),
                quality_tier=qs.tier,
                quality_score=qs.score,
                regime_adjusted_score=round(adj_score, 1),
                expected_value_r=round(ev_r, 2),
                historical_win_rate=pb.baseline_win_rate,
                regime_alignment=reg_label,
                failure_risk=fail_risk,
                behavioral_evidence=cand.behavioral_evidence,
                regime=regime_name,
                sector_leader=sec_leader,
                suggested_position_pct=pos_pct,
                institutional_summary=summary,
            ))

        return sorted(results, key=lambda x: x.rank_key(), reverse=True)

    def _get_ev(self, pb_id: str, pb, quality_tier: str, regime: str) -> float:
        """Try ExpectancyEngine; fall back to playbook baseline adjusted by quality tier."""
        try:
            from expectancy.expectancy_engine import ExpectancyEngine
            ee     = ExpectancyEngine()
            report = ee.get_expected_value_report(pb_id, quality_tier, regime)
            if report and report.sample_size >= 5:
                return report.expected_value_r
        except Exception:
            pass
        # Baseline adjustment: ELITE_A_PLUS gets 1.3× EV, A: 1.1×, B: 0.9×
        tier_mult = {"ELITE_A_PLUS": 1.3, "A": 1.1, "B": 0.9, "WATCHLIST": 0.7}.get(quality_tier, 1.0)
        baseline_ev = pb.baseline_win_rate * pb.baseline_risk_reward - (1 - pb.baseline_win_rate)
        return round(baseline_ev * tier_mult, 2)

    def _format_summary(self, cand, qs, pb, ev_r, reg_label, fail_risk, regime) -> str:
        tier_display = {
            "ELITE_A_PLUS": "ELITE A+",
            "A": "A",
            "B": "B",
            "WATCHLIST": "WATCHLIST",
        }.get(qs.tier, qs.tier)

        ev_display = f"+{ev_r:.1f}R" if ev_r >= 0 else f"{ev_r:.1f}R"
        wr_display = f"{pb.baseline_win_rate*100:.0f}%"

        evidence_lines = "\n".join(f"• {e}" for e in qs.evidence[:3]) if qs.evidence else "• Setup detected"
        disq_lines     = "\n".join(f"• {d}" for d in qs.disqualifiers[:2]) if qs.disqualifiers else ""
        risk_block     = f"\nRISK:\n{disq_lines}" if disq_lines else ""

        return (
            f"SETUP: {cand.archetype.replace('_', ' ')}\n"
            f"QUALITY: {tier_display} ({qs.score:.0f}/100)\n\n"
            f"WHY IT MATTERS:\n{evidence_lines}\n"
            f"{risk_block}\n"
            f"EXPECTANCY:\n"
            f"• Expected Value: {ev_display} historical avg\n"
            f"• Win Rate: {wr_display} · R:R {pb.baseline_risk_reward:.1f}×\n"
            f"• Regime Alignment: {reg_label}\n"
            f"• Failure Risk: {fail_risk}\n\n"
            f"REGIME: {regime.replace('_', ' ')}"
        ).strip()
