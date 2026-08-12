"""
EV engine — rank by EXPECTED VALUE, not arbitrary points.

The mindset shift (north star): the daily question is not "which stocks are
breaking out?" but "where is the highest expected return per unit of risk?".
A score of 91 says nothing about money; an EV says everything:

    EV% = [ P(win) × avgWin_R − P(loss) × avgLoss_R ] × risk% of the setup

All three components come from OUR OWN forward-tested outcomes in
signal_log (same rows and R-math as scan/live_edge — one language for
evidence). For a setup carrying several signals, the components are blended
sample-weighted across its constituent signals.

Evidence-gated like everything else: fewer than MIN_N combined outcomes →
NO EV claim (fields stay None) and the caller falls back to conviction
ranking. We never invent probabilities.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)

MIN_N = 30                     # below this, no probability claim


def signal_stats() -> dict[str, dict]:
    """Per-signal {n, p_win, avg_win_r, avg_loss_r} from real tracked
    outcomes. Empty dict when nothing is closed yet."""
    from scan.live_edge import _closed_rows, _row_r
    acc: dict[str, dict] = {}
    for row in _closed_rows():
        r = _row_r(row)
        if r is None:
            continue
        for sig in str(row.get("archetype") or "").split("|"):
            sig = sig.strip()
            if not sig:
                continue
            d = acc.setdefault(sig, {"n": 0, "wins": 0, "win_r": 0.0,
                                     "loss_r": 0.0, "n_win": 0, "n_loss": 0})
            d["n"] += 1
            if r > 0:
                d["n_win"] += 1
                d["win_r"] += r
            elif r < 0:
                d["n_loss"] += 1
                d["loss_r"] += -r
            if int(row.get("worked") or 0) == 1:
                d["wins"] += 1
    out: dict[str, dict] = {}
    for sig, d in acc.items():
        out[sig] = {
            "n": d["n"],
            "p_win": round(d["n_win"] / d["n"], 3) if d["n"] else 0.0,
            "avg_win_r": round(d["win_r"] / d["n_win"], 3) if d["n_win"] else 0.0,
            "avg_loss_r": round(d["loss_r"] / d["n_loss"], 3) if d["n_loss"] else 0.0,
        }
    return out


def wilson_lb(p: float, n: int, z: float = 1.28) -> float:
    """Wilson score lower bound for a win probability — the Bayesian-flavoured
    shrinkage that makes the system trust 520 trades @ 67% MORE than 52 trades
    @ 68%. Small n pulls the bound down hard; large n barely moves it."""
    if n <= 0:
        return 0.0
    z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    spread = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - spread) / denom)


def confidence_tier(p: float, n: int, z: float = 1.28) -> str:
    """HIGH / MEDIUM / LOW from the width of the p-estimate's uncertainty —
    principled (interval width), not an arbitrary n cutoff."""
    if n <= 0:
        return "LOW"
    half_width = z * ((p * (1 - p) / n + (z * z) / (4 * n * n)) ** 0.5)
    if half_width < 0.06:
        return "HIGH"
    if half_width < 0.12:
        return "MEDIUM"
    return "LOW"


def estimate_ev(signals: list[str], entry: float, stop: float,
                stats: dict[str, dict] | None = None,
                min_n: int = MIN_N) -> dict | None:
    """EV for one setup from its constituent signals' live stats.

    Returns {ev_pct, ev_lb_pct, p_win, avg_win_pct, avg_loss_pct, n,
    confidence} or None when the combined evidence is thinner than min_n
    (no claim). Blend is sample-weighted so a 400-trade signal outweighs a
    40-trade one. ev_lb_pct is the CONSERVATIVE EV (Wilson lower-bound win
    probability) — ranking uses it, so thin samples can't outrank deep ones
    on a lucky streak."""
    if stats is None:
        stats = signal_stats()
    if entry <= 0 or stop <= 0 or stop >= entry:
        return None
    rows = [stats[s] for s in signals if s in stats and stats[s]["n"] > 0]
    if not rows:
        return None
    total_n = sum(d["n"] for d in rows)
    if total_n < min_n:
        return None
    p_win = sum(d["p_win"] * d["n"] for d in rows) / total_n
    avg_win_r = sum(d["avg_win_r"] * d["n"] for d in rows) / total_n
    avg_loss_r = sum(d["avg_loss_r"] * d["n"] for d in rows) / total_n
    risk_pct = (entry - stop) / entry * 100          # this setup's own risk
    ev_r = p_win * avg_win_r - (1 - p_win) * avg_loss_r
    p_lb = wilson_lb(p_win, total_n)
    ev_lb_r = p_lb * avg_win_r - (1 - p_lb) * avg_loss_r
    return {
        "ev_pct": round(ev_r * risk_pct / 100 * 100, 2),   # EV in price %
        "ev_lb_pct": round(ev_lb_r * risk_pct / 100 * 100, 2),
        "p_win": round(p_win * 100, 1),
        "avg_win_pct": round(avg_win_r * risk_pct, 2),
        "avg_loss_pct": round(avg_loss_r * risk_pct, 2),
        "n": total_n,
        "confidence": confidence_tier(p_win, total_n),
    }


def tag_ev(serialized: list[dict]) -> None:
    """In-place: add ev_pct / p_win / ev_n to each scan result (None-safe).
    ADDITIVE — conviction_rank stays untouched as the fallback ranking for
    thin evidence. Uses signal LABELS mapped back to keys where needed."""
    try:
        stats = signal_stats()
    except Exception as exc:
        log.debug("ev_stats_failed", error=str(exc))
        return
    if not stats:
        return
    # results carry display labels; stats are keyed by signal KEYS. Build a
    # label→key map once from the scanner metadata.
    try:
        from scan.unified_scanner import SIGNAL_META
        label_to_key = {meta[0]: key for key, meta in SIGNAL_META.items()}
    except Exception:
        label_to_key = {}
    for r in serialized:
        labels = r.get("signals") or []
        keys = [label_to_key.get(lbl, lbl) for lbl in labels]
        ev = estimate_ev(keys, float(r.get("entry") or 0),
                         float(r.get("stop") or 0), stats=stats)
        if ev:
            r["ev_pct"] = ev["ev_pct"]
            r["ev_lb_pct"] = ev["ev_lb_pct"]
            r["p_win"] = ev["p_win"]
            r["ev_n"] = ev["n"]
            r["ev_conf"] = ev["confidence"]


def ev_rank_key(r: dict) -> tuple:
    """Sort key for the EV-first cross-sectional ranking: setups with a real
    EV claim rank above no-claim ones; within each group the CONSERVATIVE EV
    (Wilson lower bound — big samples trusted, small shrunk) decides, then
    conviction breaks ties. Use with sort(reverse=True)."""
    has_ev = r.get("ev_pct") is not None
    lb = r.get("ev_lb_pct")
    return (1 if has_ev else 0,
            float(lb if lb is not None else (r.get("ev_pct") or 0)),
            float(r.get("conviction_rank") or r.get("score") or 0))
