"""Probability of winning for every trade card, from evidence QuantTerm holds.

Definition: P(the trade closes with R > 0) under each evidence source's own
exit rules. It is a Beta-Binomial posterior over three sources, each
discounted for how far it is from "this trade, now":

  FORWARD_PAPER      settled forward paper trades of the same context   weight 1.0
  HISTORICAL_REPLAY  point-in-time replay analogs of the same setup      weight 0.5
  LIVE_SIGNALS       tracked outcomes of the same scanner signals        weight 0.5 x age decay

A weak Beta(2, 2) prior keeps an empty evidence set at 50% with a wide range;
the label says how much evidence actually stands behind the number. Nothing
here is fitted to price: it only counts what already happened.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping

PRIOR_ALPHA = 2.0
PRIOR_BETA = 2.0
HISTORICAL_WEIGHT = 0.5
LIVE_WEIGHT = 0.5
LIVE_HALF_LIFE_DAYS = 60.0
MAX_PSEUDO_N = 80.0          # no single source may dominate with a huge n
MEASURED_FORWARD_N = 30


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(value: Any) -> float | None:
    rate = _f(value)
    if rate is None:
        return None
    if rate > 1.0:               # percent -> fraction
        rate /= 100.0
    return rate if 0.0 <= rate <= 1.0 else None


def _interval(alpha: float, beta: float, mass: float = 0.80) -> tuple[float, float]:
    lo_q, hi_q = (1 - mass) / 2, 1 - (1 - mass) / 2
    try:
        from scipy.stats import beta as beta_dist

        return float(beta_dist.ppf(lo_q, alpha, beta)), float(beta_dist.ppf(hi_q, alpha, beta))
    except Exception:
        mean = alpha / (alpha + beta)
        sd = (alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))) ** 0.5
        return max(0.0, mean - 1.2816 * sd), min(1.0, mean + 1.2816 * sd)


def estimate_win_probability(
    *,
    historical: Mapping[str, Any] | None = None,
    live: Mapping[str, Any] | None = None,
    forward: Mapping[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Posterior P(win) with an 80% credible range and a per-source breakdown.

    historical: evidence_intelligence (positive_rate, effective_n)
    live:       scan-row EV fields (p_win %, ev_n_eff | ev_n, ev_last_outcome)
    forward:    {"wins": int, "n": int} settled forward paper trades
    """
    sources: list[dict[str, Any]] = []

    hist = dict(historical or {})
    h_rate, h_n = _rate(hist.get("positive_rate")), _f(hist.get("effective_n") or hist.get("raw_n"))
    if h_rate is not None and h_n and h_n > 0:
        sources.append({"source": "HISTORICAL_REPLAY", "rate": h_rate, "n": h_n,
                        "weight": HISTORICAL_WEIGHT})

    lv = dict(live or {})
    l_rate, l_n = _rate(lv.get("p_win")), _f(lv.get("ev_n_eff") or lv.get("ev_n"))
    if l_rate is not None and l_n and l_n > 0:
        decay = 1.0
        last = str(lv.get("ev_last_outcome") or "")[:10]
        try:
            age = ((today or date.today()) - date.fromisoformat(last)).days
            decay = 0.5 ** (max(0, age) / LIVE_HALF_LIFE_DAYS)
        except ValueError:
            decay = 0.25           # undated evidence counts for little
        sources.append({"source": "LIVE_SIGNALS", "rate": l_rate, "n": l_n,
                        "weight": round(LIVE_WEIGHT * decay, 4), "last_outcome": last or None})

    fw = dict(forward or {})
    f_n, f_wins = _f(fw.get("n")), _f(fw.get("wins"))
    if f_n and f_n > 0 and f_wins is not None and 0 <= f_wins <= f_n:
        sources.append({"source": "FORWARD_PAPER", "rate": f_wins / f_n, "n": f_n, "weight": 1.0})

    alpha, beta = PRIOR_ALPHA, PRIOR_BETA
    effective = 0.0
    for src in sources:
        pseudo = min(MAX_PSEUDO_N, src["n"] * src["weight"])
        src["pseudo_n"] = round(pseudo, 1)
        alpha += pseudo * src["rate"]
        beta += pseudo * (1.0 - src["rate"])
        effective += pseudo
    p = alpha / (alpha + beta)
    low, high = _interval(alpha, beta)

    forward_n = int(f_n or 0)
    if forward_n >= MEASURED_FORWARD_N:
        label = "MEASURED"
    elif effective >= 20:
        label = "ESTIMATED"
    elif effective > 0:
        label = "THIN"
    else:
        label = "UNPROVEN"
    names = {"HISTORICAL_REPLAY": "historical replay", "LIVE_SIGNALS": "tracked signals",
             "FORWARD_PAPER": "forward paper"}
    basis = ", ".join(f"{names[s['source']]} {s['rate']:.0%} (n={s['n']:.0f})" for s in sources)
    return {
        "p_win": round(p, 4),
        "low_80": round(low, 4),
        "high_80": round(high, 4),
        "effective_n": round(effective, 1),
        "label": label,
        "sources": sources,
        "definition": "P(trade closes with R > 0) under each source's own exit rules",
        "text": (
            f"Win probability {p:.0%} (80% range {low:.0%}–{high:.0%}) · {label.lower()} · "
            + (basis if basis else "no evidence yet — prior only")
        ),
    }


_BOARD_TTL_S = 60.0
_board_cache: dict[str, Any] = {"at": 0.0, "rows": {}}


def _board_by_symbol() -> dict[str, Mapping[str, Any]]:
    """Discovery rows by symbol, re-read at most once a minute (the board is
    rebuilt per scan; the request path must not re-validate it every time)."""
    import time

    now = time.monotonic()
    if _board_cache["rows"] and now - float(_board_cache["at"]) < _BOARD_TTL_S:
        return _board_cache["rows"]
    try:
        from product.decision_discovery_store import load_current

        board = load_current() or {}
    except Exception:
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for row in board.get("decisions") or []:
        if isinstance(row, Mapping):
            rows.setdefault(str(row.get("symbol") or "").upper(), row)
    _board_cache.update({"at": now, "rows": rows})
    return rows


def overlay_cards(payload: dict[str, Any]) -> dict[str, Any]:
    """Join historical-replay evidence from the discovery board onto cards.

    Cards are built before the discovery board, so the card-time estimate only
    sees live signal outcomes. At serve time each card whose symbol is on the
    board is re-estimated with its setup's replay evidence. Never raises.
    """
    by_symbol = _board_by_symbol()
    if not by_symbol:
        return payload
    for category in payload.get("categories") or []:
        for card in (category or {}).get("cards") or []:
            row = by_symbol.get(str(card.get("symbol") or "").upper())
            if not row:
                continue
            try:
                card["win_probability"] = estimate_win_probability(
                    historical=row.get("evidence_intelligence"),
                    live=card.get("win_inputs"),
                )
            except Exception:
                continue
    return payload
