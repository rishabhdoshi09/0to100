"""Method and family scorecards. Sample size is always shown.

Do not rank methods from three trades. Disagreement rows are first-class.
Calibration is separated from expected return and from win rate.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.evidence_families import METHOD_AUDIT, CONFIRMATION_FAMILIES
from product.risk_audit import r_multiple
from product.decision_taxonomy import is_judgment_row

MIN_RANK_N = 20


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _r(row: Mapping[str, Any]) -> float | None:
    if row.get("r_multiple") is not None:
        return _f(row.get("r_multiple"))
    exit_px = None
    entry = _f(row.get("entry"))
    fwd = _f(row.get("forward_return_pct"))
    if entry is not None and fwd is not None:
        exit_px = entry * (1.0 + fwd / 100.0)
    return r_multiple(entry=entry, stop=_f(row.get("stop")), exit_price=exit_px)


def _empty_card(name: str, family: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "independent_family": family,
        "decisions_participated": 0,
        "buy_participation": 0,
        "wins": 0,
        "losses": 0,
        "r_expectancy": None,
        "regimes": {},
        "setup_families": {},
        "sectors": {},
        "sample_size": 0,
        "confidence": "INSUFFICIENT_SAMPLE",
        "ranked": False,
    }


def _update(card: dict[str, Any], row: Mapping[str, Any], *, buy: bool) -> None:
    card["decisions_participated"] += 1
    if buy:
        card["buy_participation"] += 1
    r = _r(row)
    if r is not None:
        prev_n = card["sample_size"]
        prev = card["r_expectancy"] or 0.0
        card["r_expectancy"] = round((prev * prev_n + r) / (prev_n + 1), 3) if prev_n else r
        card["sample_size"] += 1
        if r > 0:
            card["wins"] += 1
        elif r < 0:
            card["losses"] += 1
    regime = str(row.get("regime") or "UNKNOWN")
    card["regimes"][regime] = card["regimes"].get(regime, 0) + 1
    setup = str(row.get("setup") or row.get("tier") or "UNKNOWN")
    card["setup_families"][setup] = card["setup_families"].get(setup, 0) + 1
    sector = str(row.get("sector") or "UNKNOWN")
    card["sectors"][sector] = card["sectors"].get(sector, 0) + 1
    n = card["sample_size"]
    card["confidence"] = "INSUFFICIENT_SAMPLE" if n < MIN_RANK_N else ("MEDIUM" if n < 50 else "HIGH")
    card["ranked"] = n >= MIN_RANK_N


def build_scorecards(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = {mid: _empty_card(audit["label"], audit["primary_family"]) for mid, audit in METHOD_AUDIT.items()}
    families = {fam: _empty_card(fam, fam) for fam in CONFIRMATION_FAMILIES}
    disagreements: list[dict[str, Any]] = []
    judgment_rows = [row for row in rows if is_judgment_row(row)]

    for row in judgment_rows:
        labels = [str(x).lower() for x in (row.get("methods_buy") or [])]
        votes = row.get("method_votes") or []
        if isinstance(votes, list):
            for item in votes:
                if isinstance(item, Mapping) and str(item.get("status") or "") == "SUPPORTIVE":
                    labels.append(str(item.get("id") or item.get("label") or "").lower())
        for mid, card in methods.items():
            audit_label = METHOD_AUDIT[mid]["label"].lower()
            if mid in labels or audit_label in labels:
                _update(card, row, buy=str(row.get("decision") or "") == "BUY")
        fam_votes = row.get("evidence_family_votes") or row.get("families") or {}
        if isinstance(fam_votes, Mapping):
            for fam, status in fam_votes.items():
                if str(status).upper() in {"SUPPORTIVE", "PASS", "BUY"} and fam in families:
                    _update(families[fam], row, buy=str(row.get("decision") or "") == "BUY")
        buy_m = list(row.get("methods_buy") or [])
        avoid_m = list(row.get("methods_avoid") or [])
        wait_m = list(row.get("methods_wait") or [])
        if buy_m and (avoid_m or wait_m):
            disagreements.append({
                "symbol": row.get("symbol"),
                "decision": row.get("decision"),
                "buy": buy_m,
                "wait": wait_m,
                "avoid": avoid_m,
                "final": row.get("decision"),
                "classification": row.get("classification"),
                "r_multiple": _r(row),
                "sample_note": "Veto value is learned only after many such rows.",
            })

    return {
        "methods": methods,
        "families": families,
        "disagreements": disagreements[:200],
        "n_rows": len(judgment_rows),
        "n_excluded_non_judgment": len(list(rows)) - len(judgment_rows),
        "min_rank_n": MIN_RANK_N,
        "note": "No ranking below sample floor. Confidence is sample-size language, not a win probability.",
        "affects_production": False,
    }


def reason_scorecards(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate outcomes by stable reason code. Sample size always visible."""
    cards: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not is_judgment_row(row):
            continue
        code = str(row.get("reason_code") or "UNSPECIFIED")
        card = cards.setdefault(code, {
            "reason_code": code,
            "n": 0,
            "wins": 0,
            "losses": 0,
            "r_sum": 0.0,
            "r_n": 0,
            "mfe_sum": 0.0,
            "mfe_n": 0,
            "mae_sum": 0.0,
            "mae_n": 0,
            "later_valid_entry": 0,
        })
        card["n"] += 1
        r = _r(row)
        if r is not None:
            card["r_sum"] += r
            card["r_n"] += 1
            if r > 0:
                card["wins"] += 1
            elif r < 0:
                card["losses"] += 1
        mfe = _f(row.get("mfe_pct"))
        mae = _f(row.get("mae_pct"))
        if mfe is not None:
            card["mfe_sum"] += mfe
            card["mfe_n"] += 1
        if mae is not None:
            card["mae_sum"] += mae
            card["mae_n"] += 1
        if row.get("later_valid_entry") or row.get("later_entered"):
            card["later_valid_entry"] += 1
    out = []
    for card in cards.values():
        n = card["n"]
        out.append({
            "reason_code": card["reason_code"],
            "n": n,
            "win_rate": round(card["wins"] / n, 3) if n else None,
            "loss_rate": round(card["losses"] / n, 3) if n else None,
            "r_expectancy": round(card["r_sum"] / card["r_n"], 3) if card["r_n"] else None,
            "mfe_pct": round(card["mfe_sum"] / card["mfe_n"], 3) if card["mfe_n"] else None,
            "mae_pct": round(card["mae_sum"] / card["mae_n"], 3) if card["mae_n"] else None,
            "later_valid_entry_rate": round(card["later_valid_entry"] / n, 3) if n else None,
            "sample_size": n,
            "ranked": n >= MIN_RANK_N,
            "confidence": "INSUFFICIENT_SAMPLE" if n < MIN_RANK_N else "MEDIUM",
        })
    out.sort(key=lambda r: r["n"], reverse=True)
    judgment_n = sum(card["n"] for card in cards.values())
    return {
        "reasons": out,
        "n_rows": judgment_n,
        "min_rank_n": MIN_RANK_N,
        "note": "Reason scorecards are descriptive. Tiny n is never a promotion signal.",
        "affects_production": False,
    }


def quality_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rs = [r for r in (_r(row) for row in rows) if r is not None]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    n = len(rs)
    return {
        "sample_size": n,
        "win_rate": round(len(wins) / n, 3) if n else None,
        "loss_rate": round(len(losses) / n, 3) if n else None,
        "average_win_r": round(sum(wins) / len(wins), 3) if wins else None,
        "average_loss_r": round(sum(losses) / len(losses), 3) if losses else None,
        "expectancy_r": round(sum(rs) / n, 3) if n else None,
        "note": "R uses frozen entry/stop. Stops are never invented retrospectively.",
    }
