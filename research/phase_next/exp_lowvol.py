"""EXP-NEXT-02 — Low-volatility effect within certified panel (ALPHA)."""
from __future__ import annotations

import pandas as pd

from research.phase_a5 import metrics as M
from research.phase_a5 import prereg
from research.phase_next import eval_utils as E
from research.phase_next import protocol as P
from research.phase_next.data import PanelBundle, period_masks


def run_exp_next_02(panel: PanelBundle) -> dict:
    closes = panel.closes
    masks = period_masks(closes)
    disc_dates = masks["discovery"]
    conf_dates = masks["confirm"]

    vol = E.realized_vol(closes, P.LOWVOL_LOOKBACK)
    # Score = inverse vol rank (high score = low vol)
    inv_vol_rank = (-vol).rank(axis=1, pct=True)
    fwd = M.forward_returns(closes, P.LOWVOL_HOLD)

    hid = prereg.preregister(
        experiment_id="EXP-NEXT-02",
        hypothesis=(
            "Lowest trailing 20d realized-volatility quintile outperforms the "
            "highest-vol quintile on a 21d hold after CNC costs (risk-adjusted "
            "and mean-net gates)."
        ),
        null_hypothesis=(
            "No positive cost-aware OOS low-minus-high vol edge; any gap is "
            "mechanical vol scaling without economic content."
        ),
        success_criteria={
            "discovery_pass": {"eq": 1},
            "confirm_pass": {"eq": 1},
            "mean_net": {"gt": 0.0},
        },
        data_window={
            "snapshot_id": panel.snapshot_id,
            "discovery": f"{P.DISCOVERY_START}→{P.DISCOVERY_END}",
            "confirm": f"{P.CONFIRM_START}→end",
            "vol_lookback": P.LOWVOL_LOOKBACK,
            "hold": P.LOWVOL_HOLD,
            "rebalance": P.LOWVOL_REBALANCE,
        },
        protocol={
            "type": "ALPHA",
            "portfolio": "long low-vol quintile / short high-vol quintile",
            "rebalance_every": P.LOWVOL_REBALANCE,
            "costs": "CNC round_trip",
            "multiple_testing": "single primary specification; DSR n_trials=1",
            "not_momentum": True,
            "production_authority": False,
        },
        seed=42,
        code_hash="phase_next_exp02_v1",
    )

    def _eval_on(dates: pd.DatetimeIndex) -> dict:
        # subsample rebalance dates
        ordered = list(dates)
        reb = ordered[:: P.LOWVOL_REBALANCE]
        reb_idx = pd.DatetimeIndex(reb)
        gross = E.long_short_period(
            inv_vol_rank, fwd, reb_idx, invert=False, top_q=P.LOWVOL_Q,
        )
        net = E.net_of_costs(gross)
        pack = E.pack_stream(net, n_trials=1)
        pack["mean_gross"] = round(float(gross.mean()) if len(gross) else 0.0, 4)
        # Vol-scaled EW control: is edge just "less vol"?
        # Compare Sharpe of low-vol long-only vs EW on same reb dates
        long_only = []
        ew = []
        for dt in reb_idx:
            if dt not in inv_vol_rank.index or dt not in fwd.index:
                continue
            s = inv_vol_rank.loc[dt].dropna()
            f = fwd.loc[dt].reindex(s.index).dropna()
            s = s.reindex(f.index).dropna()
            if len(s) < 6:
                continue
            n = max(1, int(len(s) * P.LOWVOL_Q))
            long_only.append(float(f.loc[s.nlargest(n).index].mean()))
            ew.append(float(f.mean()))
        lo = pd.Series(long_only, dtype=float)
        ew_s = pd.Series(ew, dtype=float)

        def _sh(x):
            x = pd.Series(x, dtype=float)
            if len(x) < 2 or x.std(ddof=1) == 0:
                return 0.0
            return float(x.mean() / x.std(ddof=1) * (252 / P.LOWVOL_HOLD) ** 0.5)

        verdict = E.map_discovery_verdict(
            pack["verdict"], mean_net=pack["mean_net"], fdr_ok=True,
        )
        if pack["mean_net"] <= 0:
            verdict = "FAIL"
        elif pack["verdict"] == "PROMOTE" and pack["mean_net"] > 0:
            verdict = "PASS"
        elif pack["mean_net"] > 0:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FAIL"

        return {
            "pack": pack,
            "long_only_sharpe_proxy": round(_sh(lo), 4),
            "ew_sharpe_proxy": round(_sh(ew_s), 4),
            "n_rebalances": int(len(gross)),
            "verdict": verdict,
        }

    discovery = _eval_on(disc_dates)
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = _eval_on(conf_dates)

    final = E.final_after_confirm(
        discovery["verdict"],
        None if confirmation is None else confirmation["verdict"],
    )
    metrics = {
        "discovery_pass": 1 if discovery["verdict"] == "PASS" else 0,
        "confirm_pass": 1 if confirmation and confirmation["verdict"] == "PASS" else 0,
        "mean_net": discovery["pack"]["mean_net"],
        "live_behaviour_changed": 0,
    }
    reg = prereg.record(hid, metrics)

    if final in {"FAIL", "FAILED_CONFIRMATION"}:
        prereg.remember_negative(
            f"EXP-NEXT-02 {final}: low-vol effect not confirmed "
            f"(net={discovery['pack']['mean_net']})",
            signal="low_volatility_effect",
            evidence_n=int(discovery["pack"]["n"]),
            notes="Do not add fundamentals to rescue.",
        )
    elif final == "INCONCLUSIVE":
        prereg.remember_watch(
            f"EXP-NEXT-02 INCONCLUSIVE net={discovery['pack']['mean_net']}",
            signal="low_volatility_effect",
            evidence_n=int(discovery["pack"]["n"]),
            ev_r=float(discovery["pack"]["mean_net"]),
            hypothesis_id=hid,
            notes="No tuning.",
        )

    return {
        "experiment_id": "EXP-NEXT-02",
        "type": "ALPHA",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "snapshot_id": panel.snapshot_id,
        "partitions": {
            "discovery": f"{P.DISCOVERY_START}→{P.DISCOVERY_END}",
            "confirm": f"{P.CONFIRM_START}→panel_end",
        },
        "cost_pct": E.cost_pct(),
        "discovery": discovery,
        "confirmation": confirmation,
        "final_verdict": final,
        "production_authority": False,
        "result_hash": E.result_hash({"d": discovery["verdict"], "f": final}),
    }
