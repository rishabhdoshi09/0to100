"""Decision Memory — every YES, NO and WAIT is an experiment.

The Case is the UX object. This module is the moat underneath:

  taken vs rejected (shadow book)
  similar setups (MAE / MFE / hold when the analog corpus can speak)
  forecast honesty (trust score)
  why-not for a searched name
  edge lifecycle for a setup family

Composition only. Places no orders. n < 30 never claims a guard “saved money”
or that probabilities are well calibrated. Empty is empty — never invent 18.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

PROVEN_N = 30


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def stance_for_row(row: Mapping[str, Any] | None) -> str:
    """YES / NO / WAIT from a scan or reco row. Not an order."""
    src = row or {}
    verdict = str(src.get("verdict") or src.get("action_badge") or "").upper()
    if src.get("chase_risk") or "WAIT FOR PULLBACK" in str(src.get("status") or "").upper():
        return "NO"
    if verdict in {"BUY", "STRONG BUY"} or str(src.get("action_badge") or "").lower() == "buy":
        return "YES"
    if verdict in {"AVOID"}:
        return "NO"
    return "WAIT"


def setup_quality(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Heuristic setup quality 0–100. Explicitly not a win probability."""
    src = row or {}
    score = _f(src.get("setup_quality") or src.get("conviction_score") or src.get("score")
               or src.get("combined_score"))
    score = round(max(0.0, min(100.0, score)), 0) if score else None
    return {
        "score": score,
        "label": "Setup Quality",
        "scale": 100,
        "not_probability": True,
        "blurb": (
            "Setup Quality is a weighted checklist (tape, volume, RSI, regime). "
            "It is not the chance the trade works."
        ),
    }


def similar_setup(
    row: Mapping[str, Any] | None = None,
    *,
    features: Mapping[str, Any] | None = None,
    frame: Any = None,
    symbol: str = "",
) -> dict[str, Any]:
    """Lookalike setups. Prefer MAE/MFE analogs when a corpus is already built;
    never build one on a page load. Feature-store similar history is the cheap
    fallback. Empty → not enough similar setups yet."""
    feats = dict(features or {})
    src = dict(row or {})
    if not feats:
        for key in ("rsi", "volume_ratio", "atr_pct", "momentum_5d"):
            if src.get(key) is not None:
                feats[key if key != "volume_ratio" else "vol_ratio"] = src.get(key)
                if key == "volume_ratio":
                    feats["volume_ratio"] = src.get(key)
        if src.get("regime"):
            feats["regime"] = src.get("regime")
        if src.get("from_high_pct") is not None:
            feats["dist_high_pct"] = src.get("from_high_pct")
    empty = {
        "found": False,
        "n_similar": 0,
        "line": "Not enough similar setups yet to say what usually happens next.",
        "places_orders": False,
    }
    if frame is not None and (symbol or src.get("symbol")):
        try:
            from research.market_memory import _corpus_cache, find_analogs
            if _corpus_cache.get("data") is not None:
                analog = find_analogs(str(symbol or src.get("symbol") or ""), frame)
                if analog and int(analog.get("n") or 0) >= 20:
                    return {
                        "found": True,
                        "n_similar": int(analog["n"]),
                        "win_rate": analog.get("win_rate"),
                        "avg_r": analog.get("avg_r"),
                        "avg_mae": analog.get("avg_mae"),
                        "avg_mfe": analog.get("avg_mfe"),
                        "median_hold": analog.get("median_hold"),
                        "environment": [],
                        "line": analog.get("insight") or "",
                        "places_orders": False,
                    }
        except Exception:
            pass
    try:
        from research.similar_history import similar
        got = similar(feats) if feats else {"found": False}
    except Exception:
        got = {"found": False}
    if not got.get("found"):
        note = str(got.get("note") or "")
        if note:
            empty["line"] = note[0].upper() + note[1:] if note else empty["line"]
        return empty
    n = int(got.get("n_similar") or 0)
    wr = got.get("win_rate")
    med = got.get("median_outcome_pct")
    env = list(got.get("environment") or [])
    bits = [f"{n} similar situations found"]
    if wr is not None:
        bits.append(f"{wr * 100:.0f}% profitable")
    if med is not None:
        bits.append(f"median move {med:+.2f}%")
    if env:
        bits.append("Environment: " + " / ".join(str(x) for x in env[:3]))
    return {
        "found": True,
        "n_similar": n,
        "win_rate": wr,
        "median_outcome_pct": med,
        "avg_r": None,
        "avg_mae": None,
        "avg_mfe": None,
        "median_hold": None,
        "environment": env,
        "line": ". ".join(bits) + ".",
        "places_orders": False,
    }


def trust_score() -> dict[str, Any]:
    """Public honesty about our own probabilities. Never ‘AI accuracy 94%’."""
    empty = {
        "n": 0,
        "predicted_pct": None,
        "actual_pct": None,
        "calibration_error_pct": None,
        "status": "unmeasured",
        "line": "No resolved forecasts yet — QuantTerm will not claim accuracy.",
        "places_orders": False,
    }
    try:
        from research.calibration import calibration_report
        rep = calibration_report() or {}
    except Exception:
        return empty
    n = int(rep.get("n") or 0)
    if n <= 0:
        return empty
    pred = rep.get("mean_pred")
    actual = rep.get("mean_obs")
    ece = rep.get("ece")
    pred_pct = round(float(pred) * 100.0, 1) if pred is not None else None
    actual_pct = round(float(actual) * 100.0, 1) if actual is not None else None
    err_pct = round(float(ece) * 100.0, 1) if ece is not None else None
    if n < PROVEN_N:
        status = "unproven"
        line = (
            f"Predictions made: {n}. Too few resolved forecasts to call this "
            "well calibrated. No accuracy headline."
        )
    else:
        if ece is not None and ece < 0.05:
            status = "well_calibrated"
        elif ece is not None and ece < 0.10:
            status = "roughly_calibrated"
        else:
            status = "poorly_calibrated"
        line = (
            f"Predictions made: {n}. Predicted success: {pred_pct:.0f}%. "
            f"Actual success: {actual_pct:.0f}%. Calibration error: {err_pct:.1f}%. "
            f"Current status: {status.replace('_', ' ')}."
        )
    return {
        "n": n,
        "predicted_pct": pred_pct,
        "actual_pct": actual_pct,
        "calibration_error_pct": err_pct,
        "status": status,
        "line": line,
        "places_orders": False,
    }


def shadow_book(*, min_n: int = PROVEN_N) -> dict[str, Any]:
    """Taken vs rejected. Proof that NOT buying saved money — only after n≥30."""
    empty = {
        "proven": False,
        "taken": {"n": 0, "avg_r": None, "avg_pct": None, "win_rate": None},
        "rejected": {"n": 0, "avg_r": None, "avg_pct": None, "win_rate": None},
        "wait": {"n": 0},
        "gates": [],
        "line": (
            "Shadow book is still gathering. Taken and rejected both need "
            f"{min_n} settled outcomes before QuantTerm will say a guard "
            "saved or cost money."
        ),
        "places_orders": False,
    }
    try:
        from core.decision_journal import decision_report
        from research.counterfactual import _load_decisions, _decision_r
        from research.counterfactual import gate_attribution
        report = decision_report(min_n=1)
        rejected_r, taken_r = _load_decisions()
    except Exception:
        return empty
    taken_pct = report.get("taken") or {}
    rej_pct = report.get("rejected") or {}
    wait_pct = report.get("wait") or {}
    taken_n = int(taken_pct.get("n") or 0)
    rej_n = int(rej_pct.get("n") or 0)
    taken_avg_r = round(sum(taken_r) / len(taken_r), 3) if taken_r else None
    all_rej = [r for rs in rejected_r.values() for r in rs]
    rej_avg_r = round(sum(all_rej) / len(all_rej), 3) if all_rej else None
    proven = taken_n >= min_n and rej_n >= min_n
    gates: list[dict[str, Any]] = []
    if proven:
        try:
            findings = gate_attribution(rejected_r, taken_r=taken_r, min_n=min_n)
        except Exception:
            findings = []
        for item in findings[:6]:
            label = str(item.get("gate") or "Guard")
            verdict = str(item.get("verdict") or "")
            mean_r = item.get("mean_reject_r")
            if verdict == "EARNING":
                line = f"{label} saved money (rejected names averaged {mean_r:+.2f}R)."
            elif verdict == "COSTING":
                line = f"{label} is rejecting too many winners (those names averaged {mean_r:+.2f}R)."
            else:
                line = str(item.get("insight") or label)
            gates.append({
                "gate": label,
                "n": item.get("n"),
                "mean_reject_r": mean_r,
                "verdict": verdict,
                "line": line,
            })
    if proven:
        line = (
            f"Taken trades: {taken_avg_r:+.2f}R average (n={taken_n}). "
            f"Rejected trades: {rej_avg_r:+.2f}R average (n={rej_n})."
        )
        if gates:
            line += " " + " ".join(g["line"] for g in gates[:3])
    else:
        line = empty["line"]
        extra = []
        if taken_n:
            extra.append(f"taken n={taken_n}")
        if rej_n:
            extra.append(f"rejected n={rej_n}")
        if extra:
            line += " So far: " + ", ".join(extra) + "."
    return {
        "proven": proven,
        "taken": {
            "n": taken_n,
            "avg_r": taken_avg_r if proven else None,
            "avg_pct": taken_pct.get("avg_outcome_pct") if taken_n else None,
            "win_rate": taken_pct.get("win_rate") if taken_n else None,
        },
        "rejected": {
            "n": rej_n,
            "avg_r": rej_avg_r if proven else None,
            "avg_pct": rej_pct.get("avg_outcome_pct") if rej_n else None,
            "win_rate": rej_pct.get("win_rate") if rej_n else None,
        },
        "wait": {"n": int(wait_pct.get("n") or 0)},
        "gates": gates,
        "line": line,
        "places_orders": False,
    }


def _current_gate(row: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """Gate on the live scan row, if any. Does not invent a track record."""
    src = row or {}
    reason = None
    if src.get("chase_risk"):
        reason = "EXTENSION"
    else:
        try:
            from core.decision_journal import _norm_reason
        except Exception:
            def _norm_reason(text, decision):  # type: ignore[no-redef]
                return str(text or "")
        known = {
            "EXTENSION", "BLOWOFF_RSI", "POOR_BREADTH", "CORRELATION",
            "LIQUIDITY", "LAGGARD", "DRIFT", "MACRO", "RISK_LIMIT",
        }
        for item in src.get("reasons") or []:
            mapped = _norm_reason(str(item), "REJECTED")
            if mapped in known:
                reason = mapped
                break
    if not reason:
        return None, None
    try:
        from research.explainability import REASON_LABELS
        label = REASON_LABELS.get(reason, reason.replace("_", " ").title())
    except Exception:
        label = reason.replace("_", " ").title()
    return reason, label


def why_not(symbol: str, row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Why didn't QuantTerm recommend this name? Fail-open, never invented."""
    try:
        from research.explainability import explain_rejection
        got = explain_rejection(symbol) or {}
    except Exception:
        got = {"found": False, "summary": f"No recorded rejection for {str(symbol or '').upper()}."}
    found = bool(got.get("found"))
    n = int(got.get("n_observations") or 0)
    proven = n >= PROVEN_N
    line = str(got.get("summary") or "")
    reason = got.get("reason")
    label = got.get("label")
    live_reason, live_label = _current_gate(row)
    if live_reason and not found:
        found = True
        reason = live_reason
        label = live_label
        n = 0
        proven = False
        line = (
            f"{str(symbol or '').upper()} is a NO right now — {live_label}. "
            "No settled rejections for this reason yet. QuantTerm will not "
            "say this rule saved money."
        )
    elif found and not proven:
        label = label or reason or "This filter"
        line = (
            f"{str(symbol or '').upper()} was passed — {label}. "
            f"QuantTerm has only {n} settled rejections for this reason. "
            "Promising as a rule, but not proven yet."
        )
    return {
        "found": found,
        "symbol": str(symbol or "").upper(),
        "reason": reason,
        "label": label,
        "n_observations": n if found else 0,
        "avg_fwd_pct": got.get("avg_fwd_pct") if proven else None,
        "verdict": got.get("verdict") if proven else ("unproven" if found else "unmeasured"),
        "line": line,
        "places_orders": False,
    }


def edge_state(setup: str) -> dict[str, Any]:
    """Durable character of a setup family. Empty if the timeline has nothing."""
    key = str(setup or "").strip()
    empty = {
        "setup": key,
        "profile": "UNKNOWN",
        "line": "No edge timeline for this setup yet.",
        "places_orders": False,
    }
    if not key:
        return empty
    try:
        from research.edge_timeline import signal_profile
        prof = signal_profile(key) or {}
    except Exception:
        return empty
    profile = str(prof.get("profile") or "UNKNOWN")
    rationale = str(prof.get("rationale") or "")
    line = f"{key.replace('_', ' ')} · edge state: {profile}."
    if rationale:
        line += " " + rationale
    return {
        "setup": key,
        "profile": profile,
        "n_decays": prof.get("n_decays"),
        "n_recoveries": prof.get("n_recoveries"),
        "line": line,
        "places_orders": False,
    }


def attach_to_case(case: dict[str, Any], *, row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fold Decision Memory onto an existing Case without a new page."""
    src = row or case
    case["stance"] = stance_for_row(src)
    case["setup_quality"] = setup_quality(src)
    case["similar"] = similar_setup(src, symbol=str(case.get("symbol") or ""))
    case["edge"] = edge_state(str(case.get("setup") or ""))
    case.setdefault("places_orders", False)
    return case


def morning_strip() -> dict[str, Any]:
    """Market Reports: shadow book + trust. Honest when empty."""
    shadow = shadow_book()
    trust = trust_score()
    return {
        "title": "Decision Memory",
        "blurb": (
            "Every YES, NO and WAIT is an experiment. Fewer than 30 settled "
            "outcomes stays unproven. Empty means nothing has been remembered yet."
        ),
        "shadow": shadow,
        "trust": trust,
        "places_orders": False,
    }


def for_symbol(
    symbol: str,
    *,
    row: Mapping[str, Any] | None = None,
    frame: Any = None,
) -> dict[str, Any]:
    """Stock Intelligence / search: why yes, why no, similar, trust."""
    src = dict(row or {})
    src.setdefault("symbol", str(symbol or "").upper())
    stance = stance_for_row(src)
    why = why_not(symbol, row=src) if stance != "YES" else {
        "found": False, "line": "", "places_orders": False,
    }
    return {
        "symbol": str(symbol or "").upper(),
        "stance": stance,
        "setup_quality": setup_quality(src),
        "similar": similar_setup(src, frame=frame, symbol=symbol),
        "why_not": why,
        "trust": trust_score(),
        "edge": edge_state(str(src.get("setup") or (src.get("signals") or ["UNTYPED"])[0])),
        "places_orders": False,
    }
