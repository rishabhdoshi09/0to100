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
# The same stock logged again inside its own outcome window is the SAME trade
# (overlapping 5-session holding periods), not new evidence.
DEDUPE_WINDOW_DAYS = 7
# Evidence older than this says what worked THEN, not now -> no payoff claim.
STALE_AFTER_DAYS = 45
# Confidence also needs breadth across market sessions: thousands of signals
# from one three-week tape are one regime, not a high-confidence edge.
HIGH_MIN_SESSIONS = 60
MEDIUM_MIN_SESSIONS = 20


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


def outcome_table() -> list[dict]:
    """One row per DISTINCT closed trade: {signals, r, day, symbol}.

    Rows are deduplicated per symbol within DEDUPE_WINDOW_DAYS (a re-logged
    signal on the same stock is the same holding window counted twice).
    """
    from datetime import date
    from scan.live_edge import _closed_rows, _row_r

    def _day(row: dict) -> str:
        return str(row.get("logged_at") or "")[:10]

    rows = sorted(_closed_rows(), key=_day)
    last_kept: dict[str, date] = {}
    table: list[dict] = []
    for row in rows:
        r = _row_r(row)
        if r is None:
            continue
        day_s = _day(row)
        symbol = str(row.get("symbol") or "").upper()
        try:
            day = date.fromisoformat(day_s)
        except ValueError:
            day = None
        if symbol and day is not None:
            prev = last_kept.get(symbol)
            if prev is not None and (day - prev).days < DEDUPE_WINDOW_DAYS:
                continue
            last_kept[symbol] = day
        signals = frozenset(s.strip() for s in str(row.get("archetype") or "").split("|") if s.strip())
        if signals:
            table.append({"signals": signals, "r": r, "day": day_s, "symbol": symbol})
    return table


def effective_n(days: list[str], wins: list[int]) -> tuple[float, int]:
    """(effective sample size, sessions) under same-session clustering.

    Outcomes logged on one session share that day's market move. The ANOVA
    intra-class correlation of the win indicator gives the design effect
    1 + (m-1)*rho; one session is one draw however many signals it holds.
    """
    groups: dict[str, list[int]] = {}
    for d, w in zip(days, wins):
        groups.setdefault(d, []).append(int(w))
    n, k = len(wins), len(groups)
    if n == 0:
        return 0.0, 0
    if k < 2:
        return 1.0, k
    mean = sum(wins) / n
    msb = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups.values()) / (k - 1)
    msw = (sum(sum((y - sum(g) / len(g)) ** 2 for y in g) for g in groups.values()) / (n - k)
           if n > k else 0.0)
    n0 = (n - sum(len(g) ** 2 for g in groups.values()) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    rho = 0.0 if denom <= 0 else max(0.0, min(1.0, (msb - msw) / denom))
    design = 1.0 + (n / k - 1.0) * rho
    return n / design, k


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
                min_n: int = MIN_N, table: list[dict] | None = None,
                today: "date | None" = None) -> dict | None:
    """EV for one setup from the DISTINCT closed trades carrying any of its signals.

    Returns {ev_pct, ev_lb_pct, p_win, avg_win_pct, avg_loss_pct, n, n_eff,
    sessions, last_outcome, stale, confidence} or None when fewer than min_n
    distinct trades exist (no claim). A trade tagged A|B counts ONCE for a
    setup carrying A and B -- summing per-signal counts used to claim
    n=4564 from 2884 trades. ev_lb_pct is the CONSERVATIVE EV (Wilson lower
    bound on the effective sample) and ranking uses it. ``stats`` is accepted
    for backward compatibility and unused."""
    from datetime import date as _date

    if entry <= 0 or stop <= 0 or stop >= entry:
        return None
    if table is None:
        table = outcome_table()
    wanted = {s for s in signals if s}
    rows = [t for t in table if t["signals"] & wanted]
    total_n = len(rows)
    if total_n == 0 or total_n < min_n:
        return None
    wins = [t["r"] for t in rows if t["r"] > 0]
    losses = [-t["r"] for t in rows if t["r"] < 0]
    p_win = len(wins) / total_n
    avg_win_r = sum(wins) / len(wins) if wins else 0.0
    avg_loss_r = sum(losses) / len(losses) if losses else 0.0
    n_eff, sessions = effective_n([t["day"] for t in rows], [1 if t["r"] > 0 else 0 for t in rows])
    risk_pct = (entry - stop) / entry * 100          # this setup's own risk
    ev_r = p_win * avg_win_r - (1 - p_win) * avg_loss_r
    n_bound = max(1, int(round(n_eff)))
    p_lb = wilson_lb(p_win, n_bound)
    ev_lb_r = p_lb * avg_win_r - (1 - p_lb) * avg_loss_r
    last = max((t["day"] for t in rows), default="")
    try:
        age = ((today or _date.today()) - _date.fromisoformat(last)).days
    except ValueError:
        age = None
    stale = age is None or age > STALE_AFTER_DAYS
    confidence = confidence_tier(p_win, n_bound)
    if sessions < MEDIUM_MIN_SESSIONS or stale:
        confidence = "LOW"
    elif sessions < HIGH_MIN_SESSIONS and confidence == "HIGH":
        confidence = "MEDIUM"
    return {
        "ev_pct": round(ev_r * risk_pct / 100 * 100, 2),   # EV in price %
        "ev_lb_pct": round(ev_lb_r * risk_pct / 100 * 100, 2),
        "p_win": round(p_win * 100, 1),
        "avg_win_pct": round(avg_win_r * risk_pct, 2),
        "avg_loss_pct": round(avg_loss_r * risk_pct, 2),
        "n": total_n,
        "n_eff": round(n_eff, 1),
        "sessions": sessions,
        "last_outcome": last,
        "stale": stale,
        "confidence": confidence,
    }


def tag_ev(serialized: list[dict]) -> None:
    """In-place: add ev_pct / p_win / ev_n to each scan result (None-safe).
    ADDITIVE — conviction_rank stays untouched as the fallback ranking for
    thin evidence. Uses signal LABELS mapped back to keys where needed."""
    try:
        table = outcome_table()
    except Exception as exc:
        log.debug("ev_stats_failed", error=str(exc))
        return
    if not table:
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
                         float(r.get("stop") or 0), table=table)
        if ev:
            r["ev_pct"] = ev["ev_pct"]
            r["ev_lb_pct"] = ev["ev_lb_pct"]
            r["p_win"] = ev["p_win"]
            r["ev_n"] = ev["n"]
            r["ev_n_eff"] = ev["n_eff"]
            r["ev_sessions"] = ev["sessions"]
            r["ev_last_outcome"] = ev["last_outcome"]
            r["ev_stale"] = ev["stale"]
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
