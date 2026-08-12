"""EXP-NEXT-03 — Volatility compression as RISK context (not BUY)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.phase_a5 import metrics as M
from research.phase_a5 import prereg
from research.phase_next import eval_utils as E
from research.phase_next import protocol as P
from research.phase_next.data import PanelBundle, period_masks


def _fit_tau(closes: pd.DataFrame, warmup_dates: pd.DatetimeIndex) -> float:
    """Freeze compression threshold from warmup only (never from discovery/confirm)."""
    vol_s = E.realized_vol(closes, P.VOLCOMP_SHORT)
    vol_l = E.realized_vol(closes, P.VOLCOMP_LONG)
    ratio = (vol_s / vol_l.replace(0, np.nan)).loc[warmup_dates]
    flat = ratio.to_numpy(dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size < 50:
        return 0.5  # fail-safe neutral; still frozen before OOS
    return float(np.quantile(flat, P.VOLCOMP_TAU_QUANTILE))


def run_exp_next_03(panel: PanelBundle) -> dict:
    closes = panel.closes
    masks = period_masks(closes)
    tau = _fit_tau(closes, masks["warmup"])

    vol_s = E.realized_vol(closes, P.VOLCOMP_SHORT)
    vol_l = E.realized_vol(closes, P.VOLCOMP_LONG)
    ratio = vol_s / vol_l.replace(0, np.nan)
    compressed = ratio < tau
    fwd = M.forward_returns(closes, P.VOLCOMP_FWD)
    abs_vol = vol_s  # control: absolute short vol

    hid = prereg.preregister(
        experiment_id="EXP-NEXT-03",
        hypothesis=(
            "After realized-vol compression (vol10/vol60 below warmup-frozen τ), "
            "forward 10d downside risk (loss rate / left tail) is materially worse "
            "than in non-compressed states, incremental to absolute short-vol."
        ),
        null_hypothesis=(
            "Compression does not worsen preregistered downside metrics beyond "
            "what absolute volatility already implies."
        ),
        success_criteria={
            "discovery_pass": {"eq": 1},
            "confirm_pass": {"eq": 1},
            "loss_gap_gte": {"gte": 0.05},
        },
        data_window={
            "snapshot_id": panel.snapshot_id,
            "discovery": f"{P.DISCOVERY_START}→{P.DISCOVERY_END}",
            "confirm": f"{P.CONFIRM_START}→end",
            "tau_quantile": P.VOLCOMP_TAU_QUANTILE,
            "tau_frozen": tau,
            "fwd": P.VOLCOMP_FWD,
        },
        protocol={
            "type": "RISK",
            "not_buy_signal": True,
            "compression": "vol_10 / vol_60 < tau_warmup",
            "incrementality_control": "absolute_vol_10 quintile",
            "materiality": "loss_rate_gap >= 0.05 OR left_tail_gap <= -0.01",
            "multiple_testing": "single primary definition; no τ search",
            "production_authority": False,
        },
        seed=42,
        code_hash="phase_next_exp03_v1",
    )

    def _eval_on(dates: pd.DatetimeIndex) -> dict:
        rows = []
        for dt in dates:
            if dt not in fwd.index or dt not in compressed.index:
                continue
            frow = fwd.loc[dt]
            crow = compressed.loc[dt]
            vrow = abs_vol.loc[dt]
            rrow = ratio.loc[dt]
            for sym in closes.columns:
                fv = frow.get(sym)
                cv = crow.get(sym)
                vv = vrow.get(sym)
                rv = rrow.get(sym)
                if pd.isna(fv) or pd.isna(cv) or pd.isna(vv) or pd.isna(rv):
                    continue
                rows.append({
                    "fwd": float(fv),
                    "compressed": bool(cv),
                    "abs_vol": float(vv),
                    "ratio": float(rv),
                })
        df = pd.DataFrame(rows)
        if len(df) < 60:
            return {
                "verdict": "INCONCLUSIVE",
                "reason": "underpowered",
                "n": int(len(df)),
                "tau": tau,
            }

        hi = df[df["compressed"]]
        lo = df[~df["compressed"]]
        if len(hi) < 30 or len(lo) < 30:
            return {
                "verdict": "INCONCLUSIVE",
                "reason": "split_underpowered",
                "n": int(len(df)),
                "n_comp": int(len(hi)),
                "n_not": int(len(lo)),
                "tau": tau,
            }

        def loss_rate(sub):
            return float((sub["fwd"] < 0).mean())

        def p05(sub):
            return float(np.percentile(sub["fwd"].to_numpy(), 5))

        loss_gap = loss_rate(hi) - loss_rate(lo)
        tail_gap = p05(hi) - p05(lo)
        mean_fwd_gap = float(hi["fwd"].mean() - lo["fwd"].mean())

        # Incrementality: within high absolute-vol half, does compression still matter?
        med_v = float(df["abs_vol"].median())
        high_vol = df[df["abs_vol"] >= med_v]
        hv_c = high_vol[high_vol["compressed"]]
        hv_n = high_vol[~high_vol["compressed"]]
        if len(hv_c) >= 20 and len(hv_n) >= 20:
            incr_loss_gap = loss_rate(hv_c) - loss_rate(hv_n)
            incr_tail_gap = p05(hv_c) - p05(hv_n)
            incremental = (incr_loss_gap >= 0.05) or (incr_tail_gap <= -0.01)
        else:
            incr_loss_gap = incr_tail_gap = None
            incremental = False

        material = (loss_gap >= 0.05) or (tail_gap <= -0.01)
        # RISK hypothesis: compression worsens downside
        if material and incremental:
            verdict = "PASS"
            reason = "material downside worsening incremental to abs vol"
        elif material and not incremental:
            verdict = "FAIL"
            reason = "compression not incremental to absolute volatility"
        elif not material:
            verdict = "FAIL"
            reason = "no material downside gap under frozen criteria"
        else:
            verdict = "INCONCLUSIVE"
            reason = "mixed"

        # Direction prediction is NOT success — report only
        return {
            "verdict": verdict,
            "reason": reason,
            "tau": tau,
            "n": int(len(df)),
            "n_compressed": int(len(hi)),
            "n_not": int(len(lo)),
            "loss_rate_compressed": round(loss_rate(hi), 4),
            "loss_rate_not": round(loss_rate(lo), 4),
            "loss_rate_gap": round(loss_gap, 4),
            "left_tail_p05_compressed": round(p05(hi), 4),
            "left_tail_p05_not": round(p05(lo), 4),
            "left_tail_gap": round(tail_gap, 4),
            "mean_fwd_gap_comp_minus_not": round(mean_fwd_gap, 4),
            "incremental_to_abs_vol": incremental,
            "incr_loss_gap": None if incr_loss_gap is None else round(incr_loss_gap, 4),
            "incr_tail_gap": None if incr_tail_gap is None else round(incr_tail_gap, 4),
            "note": "mean_fwd_gap is descriptive only — not a BUY criterion",
        }

    discovery = _eval_on(masks["discovery"])
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = _eval_on(masks["confirm"])

    final = E.final_after_confirm(
        discovery["verdict"],
        None if confirmation is None else confirmation["verdict"],
    )
    metrics = {
        "discovery_pass": 1 if discovery["verdict"] == "PASS" else 0,
        "confirm_pass": 1 if confirmation and confirmation["verdict"] == "PASS" else 0,
        "loss_gap_gte": float(discovery.get("loss_rate_gap") or 0.0),
        "live_behaviour_changed": 0,
    }
    reg = prereg.record(hid, metrics)

    if final in {"FAIL", "FAILED_CONFIRMATION"}:
        prereg.remember_negative(
            f"EXP-NEXT-03 {final}: vol-compression risk context not confirmed "
            f"({discovery.get('reason')})",
            signal="vol_compression_risk",
            evidence_n=int(discovery.get("n") or 0),
            notes="Do not mine alternate compression formulas; simpler abs-vol wins if FAIL.",
        )
    elif final == "INCONCLUSIVE":
        prereg.remember_watch(
            f"EXP-NEXT-03 INCONCLUSIVE: {discovery.get('reason')}",
            signal="vol_compression_risk",
            evidence_n=int(discovery.get("n") or 0),
            ev_r=float(discovery.get("loss_rate_gap") or 0),
            hypothesis_id=hid,
            notes="No τ retuning.",
        )

    return {
        "experiment_id": "EXP-NEXT-03",
        "type": "RISK",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "snapshot_id": panel.snapshot_id,
        "tau_frozen_from_warmup": tau,
        "partitions": {
            "warmup_for_tau": f"≤{P.WARMUP_END}",
            "discovery": f"{P.DISCOVERY_START}→{P.DISCOVERY_END}",
            "confirm": f"{P.CONFIRM_START}→panel_end",
        },
        "discovery": discovery,
        "confirmation": confirmation,
        "final_verdict": final,
        "production_authority": False,
        "result_hash": E.result_hash({"d": discovery["verdict"], "f": final, "tau": tau}),
    }
