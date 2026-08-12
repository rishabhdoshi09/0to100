"""
research/challenger_lab — wire incumbent vs challenger through the evidence spine.

Phase A / A3. Research-only by default: never mutates live scanner/ML behaviour.
"""
from research.challenger_lab.models import (
    ModelIdentity,
    NaiveBaseline,
    LogisticChallenger,
    fit_predict_oos,
)
from research.challenger_lab.bakeoff import (
    BakeOffConfig,
    BakeOffResult,
    VERDICT_PROMOTE,
    VERDICT_KEEP_INCUMBENT,
    VERDICT_FAIL,
    VERDICT_INCONCLUSIVE,
    run_bakeoff,
)

__all__ = [
    "ModelIdentity",
    "NaiveBaseline",
    "LogisticChallenger",
    "fit_predict_oos",
    "BakeOffConfig",
    "BakeOffResult",
    "VERDICT_PROMOTE",
    "VERDICT_KEEP_INCUMBENT",
    "VERDICT_FAIL",
    "VERDICT_INCONCLUSIVE",
    "run_bakeoff",
]
