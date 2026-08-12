"""EXP-NEXT-01 — Short-horizon cross-sectional reversal (ALPHA)."""
from __future__ import annotations

import pandas as pd

from research.phase_a5 import metrics as M
from research.phase_a5 import prereg
from research.phase_next import eval_utils as E
from research.phase_next import protocol as P
from research.phase_next.data import PanelBundle, period_masks


def run_exp_next_01(panel: PanelBundle) -> dict:
    closes = panel.closes
    masks = period_masks(closes)
    disc_dates = masks["discovery"]
    conf_dates = masks["confirm"]

    cells = [(f, h) for f in P.REVERSAL_FORMATIONS for h in P.REVERSAL_HOLDS]
    n_trials = len(cells)

    hid = prereg.preregister(
        experiment_id="EXP-NEXT-01",
        hypothesis=(
            "Cross-sectional short-horizon losers (lowest 1/3/5d return quintile) "
            "outperform winners over preregistered 5d/10d holds after CNC costs."
        ),
        null_hypothesis=(
            "No positive cost-aware OOS edge in the preregistered short-horizon "
            "reversal family after BH-FDR / DSR control."
        ),
        success_criteria={
            "discovery_pass": {"eq": 1},
            "confirm_pass": {"eq": 1},
            "best_mean_net": {"gt": 0.0},
        },
        data_window={
            "snapshot_id": panel.snapshot_id,
            "discovery": f"{P.DISCOVERY_START}→{P.DISCOVERY_END}",
            "confirm": f"{P.CONFIRM_START}→end",
            "formations": list(P.REVERSAL_FORMATIONS),
            "holds": list(P.REVERSAL_HOLDS),
        },
        protocol={
            "type": "ALPHA",
            "distinct_from": "EXP-A2-01_60d_momentum",
            "portfolio": "long bottom 20% / short top 20% by formation return",
            "costs": "CNC round_trip",
            "multiple_testing": f"BH-FDR across {n_trials} formation×hold cells; DSR n_trials={n_trials}",
            "confirmation": "later chronological OOS untouched until discovery known",
            "production_authority": False,
        },
        seed=42,
        code_hash="phase_next_exp01_v1",
    )

    def _eval_on(dates: pd.DatetimeIndex) -> dict:
        named_p = {}
        rows = {}
        for form, hold in cells:
            key = f"f{form}_h{hold}"
            # formation score = raw lookback return (low = loser)
            scores = closes.pct_change(form)
            fwd = M.forward_returns(closes, hold)
            gross = E.long_short_period(
                scores, fwd, dates, invert=True, top_q=P.REVERSAL_Q,
            )
            net = E.net_of_costs(gross)
            pack = E.pack_stream(net, n_trials=n_trials)
            pack["mean_gross"] = round(float(gross.mean()) if len(gross) else 0.0, 4)
            pack["formation"] = form
            pack["hold"] = hold
            pack["n_oos"] = int(len(net))
            # p for FDR: only claim positive edge
            named_p[key] = pack["p_value"] if pack["mean_r"] > 0 else 1.0
            rows[key] = pack
        fdr = M.fdr_on_pvalues(named_p, alpha=0.05)
        # best by mean_net among FDR survivors; else best among all (for reporting only)
        survivors = fdr["rejected"] or []
        if survivors:
            best_key = max(survivors, key=lambda k: rows[k]["mean_net"])
            fdr_ok = True
        else:
            best_key = max(rows, key=lambda k: rows[k]["mean_net"])
            fdr_ok = False
        best = rows[best_key]
        # Contrast vs rejected 60d momentum on same dates/hold of best cell
        mom_scores = M.cross_sectional_momentum_scores(closes, lookback=60)
        mom_fwd = M.forward_returns(closes, best["hold"])
        mom_gross = E.long_short_period(
            mom_scores, mom_fwd, dates, invert=False, top_q=P.REVERSAL_Q,
        )
        mom_net = E.net_of_costs(mom_gross)
        mom_pack = E.pack_stream(mom_net, n_trials=1)
        discovery_like = E.map_discovery_verdict(
            best["verdict"], mean_net=best["mean_net"], fdr_ok=fdr_ok and best["mean_net"] > 0,
        )
        # Stricter: need FDR survivor AND harness PROMOTE-ish (or positive with FDR)
        if not (fdr_ok and best["mean_net"] > 0):
            discovery_like = "FAIL" if best["mean_net"] <= 0 or not fdr_ok else discovery_like
            if not fdr_ok:
                discovery_like = "FAIL" if best["mean_net"] <= 0 else "INCONCLUSIVE"
        if fdr_ok and best["mean_net"] > 0 and best["verdict"] == "PROMOTE":
            discovery_like = "PASS"
        elif fdr_ok and best["mean_net"] > 0:
            discovery_like = "INCONCLUSIVE"  # point+FDR but harness not PROMOTE
        elif best["mean_net"] <= 0:
            discovery_like = "FAIL"
        else:
            discovery_like = "INCONCLUSIVE" if not fdr_ok else discovery_like

        return {
            "cells": rows,
            "fdr": fdr,
            "best_key": best_key,
            "best": best,
            "momentum_contrast_net": mom_pack,
            "verdict": discovery_like,
        }

    discovery = _eval_on(disc_dates)
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = _eval_on(conf_dates)
        # confirmation pass requires same cell positive net + harness not REJECT with mean>0
        cbest = confirmation["best"]
        # lock to discovery best_key for confirmation (no cell reselection)
        locked = confirmation["cells"].get(discovery["best_key"]) or cbest
        confirmation["locked_key"] = discovery["best_key"]
        confirmation["locked"] = locked
        if locked["mean_net"] > 0 and locked["verdict"] == "PROMOTE":
            confirmation["verdict"] = "PASS"
        elif locked["mean_net"] <= 0:
            confirmation["verdict"] = "FAIL"
        else:
            confirmation["verdict"] = "INCONCLUSIVE"

    final = E.final_after_confirm(
        discovery["verdict"],
        None if confirmation is None else confirmation["verdict"],
    )
    metrics = {
        "discovery_pass": 1 if discovery["verdict"] == "PASS" else 0,
        "confirm_pass": 1 if confirmation and confirmation["verdict"] == "PASS" else 0,
        "best_mean_net": discovery["best"]["mean_net"],
        "live_behaviour_changed": 0,
    }
    reg = prereg.record(hid, metrics)

    if final in {"FAIL", "FAILED_CONFIRMATION"}:
        prereg.remember_negative(
            f"EXP-NEXT-01 {final}: short-horizon reversal not confirmed "
            f"(best={discovery['best_key']} net={discovery['best']['mean_net']})",
            signal="short_horizon_reversal",
            evidence_n=int(discovery["best"]["n"]),
            notes="Do not invert momentum variants to rescue.",
        )
    elif final == "INCONCLUSIVE":
        prereg.remember_watch(
            f"EXP-NEXT-01 INCONCLUSIVE best={discovery['best_key']}",
            signal="short_horizon_reversal",
            evidence_n=int(discovery["best"]["n"]),
            ev_r=float(discovery["best"]["mean_net"]),
            hypothesis_id=hid,
            notes="No tuning.",
        )

    return {
        "experiment_id": "EXP-NEXT-01",
        "type": "ALPHA",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "snapshot_id": panel.snapshot_id,
        "partitions": {
            "discovery": f"{P.DISCOVERY_START}→{P.DISCOVERY_END}",
            "confirm": f"{P.CONFIRM_START}→panel_end",
            "n_discovery_dates": int(len(disc_dates)),
            "n_confirm_dates": int(len(conf_dates)),
        },
        "cost_pct": E.cost_pct(),
        "discovery": discovery,
        "confirmation": confirmation,
        "final_verdict": final,
        "production_authority": False,
        "result_hash": E.result_hash({"d": discovery["verdict"], "f": final, "m": metrics}),
    }
