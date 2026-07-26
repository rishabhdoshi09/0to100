"""
Live edge — learn from our OWN forward-tested outcomes, not just the backtest.

`signal_outcome_tracker` has logged hundreds of real BUY signals and, five
days later, whether each worked and by how much. Each row's `archetype`
field carries the pipe-joined signal names that fired ("BREAKOUT_52W|
POCKET_PIVOT"), so we can attribute the realised R to every signal that
took part and measure, from live results, which signals actually earn and
which leak.

This is the SAME evidence-over-vibes philosophy as the walk-forward
backtest calibration — just fed by real forward outcomes. It is used
CONSERVATIVELY: a signal is only ever demoted (never inflated) by live
data, and only once it has a real sample (≥ min_n). No data → no change,
so nothing here can hurt a fresh install.

R per signal = outcome_pct / risk_pct, where risk_pct is the signal's own
entry→stop distance (identical to the verdict dashboard).
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)

_R_CLIP = (-1.5, 4.0)          # cap gap outliers, same as the verdict dashboard
_MIN_N = 30                    # <30 outcomes = no claim (evidence over vibes)


def _closed_rows() -> list[dict]:
    try:
        from core.signal_outcome_tracker import _get_conn
        conn = _get_conn()
        rows = conn.execute(
            """SELECT symbol, archetype, regime, entry_price, stop_price, quality_score,
                      outcome_pct, worked
               FROM signal_log
               WHERE worked IS NOT NULL AND outcome_pct IS NOT NULL
                 AND entry_price > 0 AND stop_price > 0"""
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("live_edge_read_failed", error=str(exc))
        return []


def _row_r(row: dict) -> float | None:
    entry, stop = float(row["entry_price"]), float(row["stop_price"])
    if entry <= stop:
        return None
    risk_frac = (entry - stop) / entry
    r = (float(row["outcome_pct"]) / 100) / risk_frac
    # NET of round-trip trading costs — the edge that survives brokerage/STT/
    # slippage, which is what live_edge / EV / drift / beliefs should learn from.
    try:
        from core.costs import cost_in_r
        r -= cost_in_r(risk_frac, "CNC")
    except Exception:
        pass
    return max(_R_CLIP[0], min(_R_CLIP[1], r))


def _bucket(expectancy_r: float) -> float:
    """Expectancy → score multiplier. Identical banding to the backtest
    calibration so the two evidence sources speak the same language."""
    if expectancy_r >= 0.30:
        return 1.25
    if expectancy_r >= 0.10:
        return 1.0
    if expectancy_r >= -0.10:
        return 0.75
    return 0.45                 # proven loser — heavily discounted


def profile_edge() -> dict:
    """Per-signal (and per-quality-bucket) live expectancy from real tracked
    outcomes. Shape:
      {"signals": {sig: {n, wins, win_rate, expectancy_r}},
       "quality": {"score<50": {...}, "50-70": {...}, "70+": {...}},
       "overall": {n, win_rate, expectancy_r}}
    Empty dicts when there is no closed-outcome data yet.
    """
    rows = _closed_rows()
    per_sig: dict[str, list[tuple[float, int]]] = {}
    per_combo: dict[str, list[tuple[float, int]]] = {}
    per_regime: dict[str, list[tuple[float, int]]] = {}
    quality: dict[str, list[tuple[float, int]]] = {
        "score<50": [], "50-70": [], "70+": []}
    all_r: list[tuple[float, int]] = []
    for row in rows:
        r = _row_r(row)
        if r is None:
            continue
        worked = int(row["worked"])
        all_r.append((r, worked))
        # credit each constituent signal that fired in this trade
        combo = str(row.get("archetype") or "").strip()
        for sig in combo.split("|"):
            sig = sig.strip()
            if sig:
                per_sig.setdefault(sig, []).append((r, worked))
        if combo:                              # the full combination, as fired
            per_combo.setdefault(combo, []).append((r, worked))
        reg = str(row.get("regime") or "").strip()
        if reg:                                # only rows that recorded a tape
            per_regime.setdefault(reg, []).append((r, worked))
        q = float(row.get("quality_score") or 0)
        key = "score<50" if q < 50 else ("50-70" if q < 70 else "70+")
        quality[key].append((r, worked))

    def _agg(pairs: list[tuple[float, int]]) -> dict:
        n = len(pairs)
        if not n:
            return {"n": 0, "wins": 0, "win_rate": 0.0, "expectancy_r": 0.0}
        wins = sum(w for _, w in pairs)
        return {"n": n, "wins": wins, "win_rate": round(wins / n * 100, 1),
                "expectancy_r": round(sum(r for r, _ in pairs) / n, 3)}

    return {
        "signals": {s: _agg(v) for s, v in per_sig.items()},
        "combos": {c: _agg(v) for c, v in per_combo.items()},
        "regimes": {reg: _agg(v) for reg, v in per_regime.items()},
        "quality": {k: _agg(v) for k, v in quality.items()},
        "overall": _agg(all_r),
    }


def symbol_edge(min_n: int = 5) -> dict[str, dict]:
    """🧠 SYMBOL MEMORY — har stock ka apna charitra hota hai. Kuch naam
    breakout respect karte hain; kuch SERIAL FALSE-BREAKERS hain jo har
    baar retail ko phansaate hain. Firms per-instrument models rakhti
    hain; retail kabhi nahi — humare paas data pehle se tha (signal_log).

    {symbol: {n, win_rate, expectancy_r}} — sirf min_n+ outcomes waale
    (per-symbol data dheere banta hai, isliye bar chhota par claims sirf
    demote ke liye)."""
    acc: dict[str, list] = {}
    for row in _closed_rows():
        r = _row_r(row)
        if r is None:
            continue
        sym = str(row.get("symbol") or "").upper()
        if sym:
            acc.setdefault(sym, []).append((r, int(row.get("worked") or 0)))
    out: dict[str, dict] = {}
    for sym, pairs in acc.items():
        n = len(pairs)
        if n < min_n:
            continue
        wins = sum(w for _, w in pairs)
        out[sym] = {"n": n, "win_rate": round(wins / n * 100, 1),
                    "expectancy_r": round(sum(r for r, _ in pairs) / n, 3)}
    return out


def serial_losers(min_n: int = 5, max_expectancy: float = -0.30) -> set[str]:
    """Stocks jinhone humein baar-baar kata — is naam pe measured expectancy
    strongly negative hai. DEMOTE-ONLY consumer ke liye (gate/filter);
    kisi symbol ko kabhi PROMOTE nahi karta."""
    return {sym for sym, d in symbol_edge(min_n).items()
            if d["expectancy_r"] <= max_expectancy}


def regime_calibration(regime: str, min_n: int = _MIN_N) -> dict[str, float]:
    """{signal: multiplier} learned ONLY from outcomes logged in `regime`.

    A signal can be gold in a trending tape and poison in distribution — the
    lifetime average blends the two. This conditions the calibration on the
    CURRENT tape so the scorer can demote a signal that leaks *in this regime*
    even when it looks fine overall. Gated at min_n same-regime outcomes; empty
    (no claim) for an unknown regime or thin data."""
    if not regime or str(regime).strip().upper() in ("", "UNKNOWN"):
        return {}
    reg = str(regime).strip()
    per_sig: dict[str, list[float]] = {}
    for row in _closed_rows():
        if str(row.get("regime") or "").strip() != reg:
            continue
        r = _row_r(row)
        if r is None:
            continue
        for sig in str(row.get("archetype") or "").split("|"):
            sig = sig.strip()
            if sig:
                per_sig.setdefault(sig, []).append(r)
    out: dict[str, float] = {}
    for sig, rs in per_sig.items():
        if len(rs) >= min_n:
            out[sig] = _bucket(sum(rs) / len(rs))
    if out:
        log.info("regime_calibration_ready", regime=reg, signals=len(out))
    return out


def rolling_expectancy(windows: tuple = (50, 100, 250)) -> dict[int, float]:
    """Edge-decay lens: expectancy over the LAST 50/100/250 outcomes
    (chronological). A 250-window healthy but 50-window sinking = edge dying
    NOW — funds kill strategies, retail marries them. Windows without enough
    data are omitted (no claim)."""
    rows = _closed_rows()
    rs: list[float] = []
    for row in rows:                      # insertion order = logged order
        r = _row_r(row)
        if r is not None:
            rs.append(r)
    out: dict[int, float] = {}
    for w in windows:
        if len(rs) >= w:
            tail = rs[-w:]
            out[w] = round(sum(tail) / w, 3)
    return out


def live_calibration(min_n: int = _MIN_N) -> dict[str, float]:
    """{signal: score_multiplier} from live outcomes, gated at min_n.

    Only signals with a real sample are returned; everything else is left
    for the caller to default to 1.0. The caller blends this CONSERVATIVELY
    (see UnifiedScanner) so live data can demote a leaky signal but never
    inflate one the backtest already distrusts."""
    prof = profile_edge()
    out: dict[str, float] = {}
    for sig, a in prof["signals"].items():
        if a["n"] >= min_n:
            out[sig] = _bucket(a["expectancy_r"])
    if out:
        log.info("live_calibration_ready", signals=len(out))
    return out
