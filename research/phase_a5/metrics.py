"""Shared evaluation utilities for Phase A.5 (research only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.harness import evaluate, benjamini_hochberg, effective_sample_size


def returns_panel(closes: pd.DataFrame) -> pd.DataFrame:
    return closes.pct_change().dropna(how="all")


def cross_sectional_momentum_scores(closes: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Incumbent-style rule: 60d return rank (transparent baseline, not ML)."""
    rets = closes.pct_change(lookback)
    return rets.rank(axis=1, pct=True)


def long_short_from_scores(scores: pd.DataFrame, fwd: pd.DataFrame, top_q: float = 0.2):
    """Equal-weight long top quintile vs short bottom; returns per-date portfolio R."""
    common = scores.index.intersection(fwd.index)
    scores, fwd = scores.loc[common], fwd.loc[common]
    port = []
    dates = []
    for dt in common:
        s = scores.loc[dt].dropna()
        f = fwd.loc[dt].reindex(s.index).dropna()
        s = s.reindex(f.index).dropna()
        if len(s) < 6:
            continue
        n = max(1, int(len(s) * top_q))
        long = s.nlargest(n).index
        short = s.nsmallest(n).index
        r = float(f.loc[long].mean() - f.loc[short].mean())
        port.append(r)
        dates.append(dt)
    return pd.Series(port, index=pd.Index(dates), dtype=float)


def forward_returns(closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return closes.pct_change(horizon).shift(-horizon)


def cost_drag(turnover_one_way: float, round_trip_pct: float) -> float:
    # round_trip_pct is percent points (e.g. 0.32); convert to fraction
    return float(turnover_one_way) * (round_trip_pct / 100.0)


def harness_pack(returns, *, n_trials: int = 1, min_n: int = 30) -> dict:
    ev = evaluate(returns, n_trials=n_trials, min_n=min_n)
    return {
        "verdict": ev.verdict,
        "n": ev.n,
        "n_eff": float(effective_sample_size(returns)) if len(returns) else 0.0,
        "mean_r": ev.mean_r,
        "sharpe": ev.sharpe,
        "psr": ev.psr,
        "dsr": ev.dsr,
        "p_value": ev.p_value,
        "insight": ev.insight,
    }


def fdr_on_pvalues(named_p: dict[str, float], alpha: float = 0.05) -> dict:
    names = list(named_p)
    p = [named_p[n] for n in names]
    if not p:
        return {"rejected": [], "detail": {}}
    res = benjamini_hochberg(p, alpha=alpha)
    rejected_mask = np.asarray(res["rejected"], dtype=bool)
    qvalues = np.asarray(res["qvalues"], dtype=float)
    rejected = []
    detail = {}
    for i, n in enumerate(names):
        flag = bool(rejected_mask[i])
        detail[n] = {"p": named_p[n], "rejected": flag, "q": float(qvalues[i])}
        if flag:
            rejected.append(n)
    return {"rejected": rejected, "detail": detail, "threshold": res["threshold"]}



def gate_research_grade(manifest: dict) -> dict:
    """Hard gate: exploratory sources cannot earn scientific PASS/FAIL verdicts.

    Scientific evaluation is allowed when:
      • global ``trust_class=RESEARCH_GRADE`` and ``research_grade=True``, OR
      • Phase A.5 **scoped** certification is ``READY_FOR_SCIENTIFIC_RERUN``
        (global trust may remain ``OPERATIONAL_ONLY``).

    ``may_promote`` here means "may issue scientific PASS/FAIL" — it does **not**
    grant production authority. Callers must keep ``production_authority=False``.
    """
    rg = bool(manifest.get("research_grade"))
    trust = str(manifest.get("trust_class") or "")
    scoped_ok = (
        str(manifest.get("scoped_certification") or "") == "READY_FOR_SCIENTIFIC_RERUN"
        and bool(manifest.get("scoped_eligible_for_scientific_rerun"))
        and str(manifest.get("scope") or "") == "PHASE_A5_FROZEN_PROTOCOL"
        and bool(manifest.get("snapshot_id"))
    )
    global_ok = rg and trust == "RESEARCH_GRADE"
    scientific = bool(global_ok or scoped_ok)
    if global_ok:
        reason = "RESEARCH_GRADE inputs present"
    elif scoped_ok:
        reason = (
            "Phase A.5 scoped certification READY_FOR_SCIENTIFIC_RERUN "
            f"(snapshot={manifest.get('snapshot_id')}); "
            f"global trust remains {trust or 'UNKNOWN'}"
        )
    else:
        reason = (
            "inputs are not RESEARCH_GRADE — exploratory metrics only; "
            "PASS_ALPHA/PASS_RISK promotion blocked"
        )
    return {
        "research_grade": bool(rg or scoped_ok),
        "trust_class": trust,
        "scoped_certification": manifest.get("scoped_certification"),
        "scoped_ok": scoped_ok,
        "scientific_evaluation": scientific,
        "may_promote": scientific,
        "production_authority": False,
        "reason": reason,
    }


def scientific_verdict(raw: str) -> str:
    """Map internal runner tags to report vocabulary PASS|FAIL|INCONCLUSIVE."""
    v = str(raw or "").upper()
    if v in {"PASS", "PASS_ALPHA", "PASS_RISK", "PROMOTE"}:
        return "PASS"
    if v in {"FAIL", "REJECT", "REJECTED"}:
        return "FAIL"
    return "INCONCLUSIVE"
