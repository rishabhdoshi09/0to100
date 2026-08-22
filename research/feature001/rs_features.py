"""Canonical RS feature vector (rs_features_v1) wrapping rs_cs_v1.

Does not change the RS methodology. RS >= 70 is a descriptive flag only.
"""
from __future__ import annotations

from typing import Any, Mapping

from research.feature001.constants import RS_DELTA_SESSIONS, RS_SOURCE, RS_VERSION, STRONG_RS, rs_bucket
from research.sepa.rs import lookup_rs


def compute_rs_features(
    symbol: str,
    table: Mapping[str, Any] | None,
    *,
    prev_table: Mapping[str, Any] | None = None,
    bench_rel_63: float | None = None,
) -> dict[str, Any]:
    one = lookup_rs(table, symbol)
    pct = one.get("percentile")
    score = one.get("score")
    comps = dict(one.get("components") or {})
    prev = lookup_rs(prev_table, symbol) if prev_table else {"available": False, "percentile": None}
    prev_pct = prev.get("percentile")
    delta = None
    if pct is not None and prev_pct is not None:
        delta = float(pct) - float(prev_pct)
    return {
        "version": RS_VERSION,
        "source": RS_SOURCE,
        "available": bool(one.get("available")),
        "rs_percentile": None if pct is None else float(pct),
        "rs_score": None if score is None else float(score),
        "r63": comps.get("r63"),
        "r126": comps.get("r126"),
        "r189": comps.get("r189"),
        "r252": comps.get("r252"),
        "rs_ge_70": None if pct is None else bool(float(pct) >= STRONG_RS),
        "rs_bucket": rs_bucket(None if pct is None else float(pct)),
        "rs_pct_chg_21": None if delta is None else round(delta, 4),
        "rs_delta_sessions": RS_DELTA_SESSIONS,
        "bench_rel_63": None if bench_rel_63 is None else float(bench_rel_63),
        "n_ranked": one.get("n_ranked"),
    }
