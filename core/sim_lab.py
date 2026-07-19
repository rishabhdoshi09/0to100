"""
🧪 Simulation Lab — thousands of plausible futures before one live change.

Backtesting replays THE past once. The Lab bootstraps the system's OWN
closed-trade R distribution into Monte Carlo paths and asks: "agar hum yeh
parameter badlein, toh hazaaron plausible futures mein kya hota hai?" —
expected growth, drawdown distribution, probability of a painful hole —
BEFORE the risk slider moves on live capital.

Also carries the Capital Scaling advisor: capital is EARNED by measured
performance (PF, drawdown, sample), never granted because a week felt good.

Everything here is pure math over real trade outcomes — no market data, no
network, fully deterministic under a seed, ADVICE-ONLY (nothing here moves
allocation or risk by itself).
"""
from __future__ import annotations

import random

from logger import get_logger

log = get_logger(__name__)

MIN_TRADES = 30                 # below this, simulation is fiction — refuse


def simulate(rs: list[float], risk_frac: float = 0.01, horizon: int = 250,
             n_paths: int = 500, seed: int = 42) -> dict | None:
    """Bootstrap Monte Carlo. rs = realised R-multiples of closed trades.
    Each path: `horizon` trades sampled with replacement, equity compounding
    at `risk_frac` per trade. None when evidence is too thin (no fiction)."""
    if len(rs) < MIN_TRADES or risk_frac <= 0:
        return None
    rng = random.Random(seed)
    finals: list[float] = []
    max_dds: list[float] = []
    dd20 = 0
    for _ in range(n_paths):
        eq = 1.0
        peak = 1.0
        worst_dd = 0.0
        for _ in range(horizon):
            eq *= (1 + risk_frac * rng.choice(rs))
            peak = max(peak, eq)
            worst_dd = max(worst_dd, (peak - eq) / peak)
        finals.append(eq)
        max_dds.append(worst_dd)
        if worst_dd >= 0.20:
            dd20 += 1
    finals.sort()
    max_dds.sort()

    def _pct(vals: list[float], q: float) -> float:
        return vals[min(len(vals) - 1, int(q * len(vals)))]

    return {
        "n_paths": n_paths, "horizon": horizon,
        "risk_frac": risk_frac, "n_trades_evidence": len(rs),
        "median_growth_pct": round((_pct(finals, 0.50) - 1) * 100, 1),
        "p05_growth_pct": round((_pct(finals, 0.05) - 1) * 100, 1),
        "p95_growth_pct": round((_pct(finals, 0.95) - 1) * 100, 1),
        "median_max_dd_pct": round(_pct(max_dds, 0.50) * 100, 1),
        "p95_max_dd_pct": round(_pct(max_dds, 0.95) * 100, 1),
        "prob_loss_pct": round(sum(1 for f in finals if f < 1) / n_paths * 100, 1),
        "prob_dd20_pct": round(dd20 / n_paths * 100, 1),
    }


def compare(rs: list[float], risk_a: float, risk_b: float,
            horizon: int = 250, n_paths: int = 500,
            seed: int = 42) -> dict | None:
    """A/B two risk settings over the same futures (same seed → same trade
    sequences, only sizing differs). The pre-flight check for the slider."""
    a = simulate(rs, risk_a, horizon, n_paths, seed)
    b = simulate(rs, risk_b, horizon, n_paths, seed)
    if not a or not b:
        return None
    growth_gain = b["median_growth_pct"] - a["median_growth_pct"]
    dd_cost = b["p95_max_dd_pct"] - a["p95_max_dd_pct"]
    if growth_gain > 0 and dd_cost <= 2.0:
        verdict = (f"{risk_b * 100:.2f}% risk median growth "
                   f"{growth_gain:+.1f}pp deta hai bina meaningful extra "
                   f"drawdown ke — defensible.")
    elif growth_gain > 0:
        verdict = (f"{risk_b * 100:.2f}% risk growth {growth_gain:+.1f}pp "
                   f"deta hai LEKIN worst-case DD {dd_cost:+.1f}pp badhta "
                   f"hai — survival pehle, soch ke.")
    else:
        verdict = (f"{risk_b * 100:.2f}% risk se growth improve NAHI hota "
                   f"({growth_gain:+.1f}pp) — change ka koi case nahi.")
    return {"a": a, "b": b, "verdict": verdict}


def scaling_advice(profit_factor: float, max_dd_pct: float,
                   n_trades: int) -> dict:
    """Capital is earned: PF ≥1.8 + DD <5% + 200 trades → step up 20%;
    PF <1.2 on a real sample → step down 25%; otherwise hold and let
    evidence accumulate. ADVICE-ONLY — the user moves the allocation."""
    if n_trades < 50:
        return {"action": "HOLD", "change_pct": 0,
                "reason": (f"Sirf {n_trades} closed trades — scaling ka faisla "
                           f"50+ pe hota hai. Evidence jama karo.")}
    if profit_factor >= 1.8 and max_dd_pct < 5.0 and n_trades >= 200:
        return {"action": "INCREASE", "change_pct": 20,
                "reason": (f"PF {profit_factor:.2f}, DD {max_dd_pct:.1f}%, "
                           f"{n_trades} trades — system ne capital EARN kiya. "
                           f"+20% step (ek baar mein zyada nahi).")}
    if profit_factor < 1.2:
        return {"action": "REDUCE", "change_pct": -25,
                "reason": (f"PF {profit_factor:.2f} < 1.2 ({n_trades} trades) — "
                           f"deployment 25% ghatao jab tak edge wapas prove "
                           f"na ho.")}
    return {"action": "HOLD", "change_pct": 0,
            "reason": (f"PF {profit_factor:.2f}, DD {max_dd_pct:.1f}%, "
                       f"{n_trades} trades — theek chal raha, size wahi rakho.")}
