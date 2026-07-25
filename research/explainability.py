"""
💬 Explainability — turn the research memory into a customer-facing "why not?".

Most retail tools answer "why didn't you recommend this?" with "low conviction".
With a Feature Platform, a counterfactual control group, and an Evidence Graph,
this system can answer it with an audit trail the user can trust:

    Rejected because  ·  Extension Guard
      Of the names this reason passed:   417 observations
      Correctly avoided (fell):          379   ·   False-rejection rate: 9.1%
      Observed avg forward return:       −2.3%     (this "no" SAVED money)
      Modeled (ATR-2x stop):             −0.8R     (counterfactual)
      Verdict:                           EARNING its keep
      Backed by belief:                  "breakouts fail when extended"
      Last revalidated:                  18 days ago

Every number is sourced: observed stats from the settled non-event outcomes
(`research.non_event`), the backing belief + freshness from the Scientific Memory,
and the provenance chain from the Evidence Graph. Observed and MODELED figures are
kept visibly separate (a rejected trade had no real stop — R is modeled, never
observed). Composition only; owns no statistics. Fail-open: a missing piece is
omitted, never guessed.
"""
from __future__ import annotations

from datetime import datetime

# friendly names for the structured rejection causes
REASON_LABELS = {
    "EXTENSION": "Extension Guard",
    "LOW_CONVICTION": "Low Conviction",
    "WEAK_CLOSE": "Weak Close (bull-trap guard)",
    "BLOWOFF_RSI": "Overheated RSI",
    "LAGGARD": "Laggard (far below highs)",
    "POOR_BREADTH": "Poor Market Breadth",
    "RISK_LIMIT": "Portfolio Risk Limit",
    "CORRELATION": "Position Correlation",
    "LIQUIDITY": "Insufficient Liquidity",
    "MACRO": "Macro Risk-Off",
    "ALREADY_OWNED": "Already Held",
    "DRIFT": "Signal in Decay",
    "OTHER": "Other",
}

_VERDICT_PHRASE = {
    "EARNING": "earning its keep (correctly avoids losers)",
    "TOO_CONSERVATIVE": "too conservative (passing winners)",
    "NEUTRAL": "roughly neutral so far",
    "INSUFFICIENT": "not yet enough evidence to judge",
}


def _days_since(iso: str | None):
    if not iso:
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(str(iso)[:19])).days
    except Exception:
        return None


def explain_reason(reason: str) -> dict:
    """The evidence behind ONE rejection reason: its counterfactual track record
    (observed) + an optional modeled R + any belief backing it. Fail-open → a
    minimal stub. Numbers are sourced, never invented."""
    from research.non_event import rejection_analysis, _norm_reason
    rc = _norm_reason(reason)
    stats = next((a for a in rejection_analysis() if a["reason"] == rc), None)
    label = REASON_LABELS.get(rc, rc.title())
    out: dict = {"reason": rc, "label": label}
    if stats:
        n = stats["n"] or 0
        missed = stats["missed_winners"]
        out.update({
            "n_observations": n,
            "correctly_avoided": stats["correctly_avoided"],
            "missed_winners": missed,
            "false_rejection_rate": round(missed / n, 3) if n else None,
            "avg_fwd_pct": stats["avg_fwd_pct"],           # observed, canonical
            "modeled_avg_r": stats.get("modeled_avg_r"),   # modeled, labelled
            "modeled_assumption": stats.get("modeled_assumption"),
            "verdict": stats["verdict"],
        })
    # optional: the belief that justifies the gate + its freshness/confidence
    try:
        from research.scientific_memory import list_beliefs
        b = next((x for x in list_beliefs() if (x.get("signal") or "").upper()
                  == rc or rc.lower() in (x.get("statement") or "").lower()), None)
        if b:
            out["belief"] = b.get("statement")
            out["confidence"] = b.get("confidence")
            out["belief_status"] = b.get("status")
            out["last_revalidated_days"] = _days_since(b.get("last_validated_at"))
    except Exception:
        pass
    out["summary"] = _summarise(out)
    return out


def _summarise(o: dict) -> str:
    label = o.get("label", o.get("reason", "This filter"))
    if "n_observations" not in o:
        return f"{label}: no settled evidence yet — still gathering the control group."
    n = o["n_observations"]
    verdict = _VERDICT_PHRASE.get(o.get("verdict", ""), "")
    obs = o.get("avg_fwd_pct")
    saved = ("SAVED money" if obs is not None and obs < 0
             else "COST opportunity" if obs is not None and obs > 0 else "was flat")
    bits = [f"{label}: across {n} names it passed, the reason {saved} "
            f"(observed avg {obs:+.1f}% forward)"]
    if o.get("false_rejection_rate") is not None:
        bits.append(f"false-rejection rate {o['false_rejection_rate']*100:.1f}%")
    if o.get("modeled_avg_r") is not None:
        bits.append(f"modeled {o['modeled_avg_r']:+.2f}R ({o.get('modeled_assumption','counterfactual')})")
    if verdict:
        bits.append(f"verdict: {verdict}")
    if o.get("belief"):
        fresh = (f", revalidated {o['last_revalidated_days']}d ago"
                 if o.get("last_revalidated_days") is not None else "")
        bits.append(f"backed by belief “{o['belief']}” "
                    f"({o.get('confidence', '—')} confidence{fresh})")
    return "; ".join(bits) + "."


def explain_rejection(symbol: str) -> dict:
    """"Why wasn't <symbol> recommended?" — find its most recent REJECTION
    observation, and explain the reason with the evidence behind it. Fail-open →
    {'found': False}."""
    sym = (symbol or "").upper()
    try:
        from research import feature_store as _FS
        c = _FS._conn()
        try:
            row = c.execute(
                "SELECT reason, ts FROM observations WHERE symbol=? AND "
                "kind='REJECTION' ORDER BY ts DESC LIMIT 1", (sym,)).fetchone()
        finally:
            c.close()
    except Exception:
        row = None
    if not row or not row["reason"]:
        return {"symbol": sym, "found": False,
                "summary": f"No recorded rejection for {sym}."}
    ex = explain_reason(row["reason"])
    ex.update({"symbol": sym, "found": True, "rejected_at": row["ts"]})
    ex["summary"] = f"{sym} was passed — {ex['summary']}"
    return ex
