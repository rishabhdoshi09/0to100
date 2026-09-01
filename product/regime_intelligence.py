"""Regime Intelligence 2.0 — measured, interpretable, shadow-first.

Does not replace production RISK_ON / RISK_OFF until an explicit promotion.
Insufficient evidence stays UNKNOWN. One bad week cannot change production.
Hard risk gates remain intact regardless of regime label.
"""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = 1  # previous: production uses RISK_ON / RISK_OFF only.
SHADOW_MODE = True

BROAD_RISK_ON = "BROAD_RISK_ON"
NARROW_LEADERSHIP = "NARROW_LEADERSHIP"
VOLATILE_RISK_ON = "VOLATILE_RISK_ON"
TRANSITION = "TRANSITION"
DISTRIBUTION = "DISTRIBUTION"
RISK_OFF = "RISK_OFF"
UNKNOWN = "UNKNOWN"

PRODUCTION_ON = "RISK_ON"
PRODUCTION_OFF = "RISK_OFF"

REQUIRED_MIN_FIELDS = 3
PROMOTION_MIN_WEEKS = 4


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _measured(evidence: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = evidence.get(key)
    if isinstance(raw, Mapping) and "value" in raw:
        val = _f(raw.get("value"))
        return {
            "name": key,
            "value": val,
            "source": raw.get("source") or "unspecified",
            "confidence": raw.get("confidence") or ("high" if val is not None else "none"),
            "measured": val is not None and not raw.get("estimated"),
            "estimated": bool(raw.get("estimated")) if val is not None else False,
            "missing": val is None,
        }
    val = _f(raw)
    return {
        "name": key,
        "value": val,
        "source": "caller",
        "confidence": "medium" if val is not None else "none",
        "measured": val is not None,
        "estimated": False,
        "missing": val is None,
    }


class RegimeIntelligenceEngine:
    """Shadow classifier. Production mapping is returned separately."""

    def __init__(self, *, shadow_mode: bool = True, promoted: bool = False) -> None:
        self.shadow_mode = bool(shadow_mode) and not promoted
        self.promoted = bool(promoted)

    def classify(
        self,
        evidence: Mapping[str, Any] | None = None,
        *,
        production_regime: str = PRODUCTION_ON,
        weeks_of_agreement: int = 0,
        last_week_only: bool = False,
    ) -> dict[str, Any]:
        evidence = dict(evidence or {})
        fields = [
            _measured(evidence, key)
            for key in (
                "breadth",
                "advance_decline",
                "pct_above_ma",
                "new_highs_lows",
                "breakout_success_rate",
                "breakout_failure_rate",
                "volatility",
                "gap_behavior",
                "sector_participation",
                "leadership_concentration",
                "fii_dii",
                "index_trend",
                "liquidity",
                "recent_strategy_performance",
            )
        ]
        present = [f for f in fields if not f["missing"]]
        if len(present) < REQUIRED_MIN_FIELDS:
            state = UNKNOWN
            reason = f"insufficient evidence ({len(present)}/{REQUIRED_MIN_FIELDS} fields)"
        else:
            state, reason = self._state(present)

        # One bad week cannot change production even after promotion.
        production = str(production_regime or PRODUCTION_ON)
        affects_production = False
        if self.promoted and not self.shadow_mode:
            if last_week_only or weeks_of_agreement < PROMOTION_MIN_WEEKS:
                affects_production = False
                reason = reason + "; one-week move cannot change production"
            else:
                affects_production = True

        policy = self.policy_for(state, production_regime=production)
        return {
            "schema_version": SCHEMA_VERSION,
            "shadow_mode": self.shadow_mode,
            "promoted": self.promoted,
            "affects_production": affects_production,
            "state": state,
            "reason": reason,
            "production_regime": production,
            "fields": fields,
            "n_measured": len(present),
            "policy": policy,
            "hard_gates_intact": True,
            "live_locked": True,
        }

    def _state(self, present: list[dict[str, Any]]) -> tuple[str, str]:
        by = {f["name"]: f["value"] for f in present}
        breadth = by.get("breadth")
        vol = by.get("volatility")
        conc = by.get("leadership_concentration")
        trend = by.get("index_trend")
        fail = by.get("breakout_failure_rate")
        ad = by.get("advance_decline")
        pct_ma = by.get("pct_above_ma")

        if trend is not None and trend < 0 and (breadth is not None and breadth < 40 or ad is not None and ad < 0):
            return RISK_OFF, "negative trend with weak breadth"
        if fail is not None and fail >= 0.55 and (trend is not None and trend <= 0):
            return DISTRIBUTION, "breakout failures elevated while trend is not up"
        if conc is not None and conc >= 0.65 and (breadth is not None and breadth < 50):
            return NARROW_LEADERSHIP, "leadership concentrated; breadth not broad"
        if vol is not None and vol >= 0.7 and (trend is not None and trend > 0):
            return VOLATILE_RISK_ON, "uptrend with elevated volatility"
        if trend is not None and abs(trend) < 0.15:
            return TRANSITION, "index trend near flat"
        if (
            (breadth is not None and breadth >= 60)
            or (pct_ma is not None and pct_ma >= 60)
        ) and (trend is None or trend >= 0):
            return BROAD_RISK_ON, "broad participation with non-negative trend"
        return TRANSITION, "mixed measured fields; no dominant state"

    def policy_for(self, state: str, *, production_regime: str = PRODUCTION_ON) -> dict[str, Any]:
        """Recommended posture. Never disables hard gates."""
        table = {
            BROAD_RISK_ON: {"max_new": 3, "risk_budget_pct": 5.0, "enter_bias": "ENTER_NOW"},
            NARROW_LEADERSHIP: {"max_new": 2, "risk_budget_pct": 3.0, "enter_bias": "WAIT"},
            VOLATILE_RISK_ON: {"max_new": 2, "risk_budget_pct": 3.0, "enter_bias": "WAIT"},
            TRANSITION: {"max_new": 1, "risk_budget_pct": 2.0, "enter_bias": "WAIT"},
            DISTRIBUTION: {"max_new": 0, "risk_budget_pct": 1.0, "enter_bias": "WAIT"},
            RISK_OFF: {"max_new": 0, "risk_budget_pct": 1.0, "enter_bias": "NO_TRADE"},
            UNKNOWN: {"max_new": None, "risk_budget_pct": None, "enter_bias": "UNCHANGED"},
        }
        pol = dict(table.get(state) or table[UNKNOWN])
        pol["hard_gates_intact"] = True
        pol["cannot_disable_dd"] = True
        pol["cannot_disable_stop"] = True
        pol["cannot_disable_liquidity"] = True
        pol["applies_to_production"] = False if self.shadow_mode else (
            production_regime == PRODUCTION_OFF or state == RISK_OFF
        )
        # Shadow policy never overrides production RISK_ON/OFF
        if self.shadow_mode:
            pol["applies_to_production"] = False
        return pol


def shadow_classify(evidence: Mapping[str, Any] | None, *, production_regime: str = PRODUCTION_ON) -> dict[str, Any]:
    return RegimeIntelligenceEngine(shadow_mode=True, promoted=False).classify(
        evidence, production_regime=production_regime
    )
