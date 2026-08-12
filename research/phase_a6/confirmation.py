"""EXP-A6-CONF-01 — Independent confirmation of signal × network concentration.

Uses the pre-OOS holdout of scoped snapshot a7a9828ec37e09e4 only.
Does not overwrite EXP-A5A6-01. Production authority always False.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.costs import round_trip_cost_pct
from product.plain_language import PlainCard, render_layers
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg
from research.phase_a5.dataset import (
    CERTIFIED_SNAPSHOT_ID,
    load_certified_snapshot,
    load_sectors,
)
from research.phase_a5.scoped_certification import FROZEN_PANEL
from research.phase_a6.frozen_hypothesis import (
    CONFIRMATION_PROTOCOL,
    DISCOVERY,
    REJECTED_BRANCHES,
    write_frozen_protocol,
)
from research.portfolio_network import analyze_network
from research.portfolio_network.engine import corr_from_returns
from risk.correlation import clusters_from_corr

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "logs" / "phase_a6" / "confirmation_results.json"
OUT_MD = REPO_ROOT / "PHASE_A6_NETWORK_INTERACTION_CONFIRMATION.md"
OOS_START = DISCOVERY["oos_start"]
LOOKBACK = 60
STEP = 21
FWD_BARS = 10
RHO = 0.70
ALPHA = 0.05
MIN_SPLIT = 30


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def _sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_rejected_branches() -> list[dict]:
    """Record rejected A.5 branches as negative scientific memory (idempotent notes)."""
    out = []
    for b in REJECTED_BRANCHES:
        stmt = (
            f"PHASE_A6_FREEZE REJECT {b['experiment_id']} ({b['branch']}): "
            f"certified FAIL — do not retune or escalate."
        )
        prereg.remember_negative(
            stmt,
            signal=b["branch"],
            evidence_n=1,
            notes=b["freeze"],
        )
        out.append({"hypothesis_id": b["hypothesis_id"], "status": "REJECT_FROZEN"})
    return out


def _interaction_test(df: pd.DataFrame, context_col: str) -> dict:
    if df.empty or df[context_col].nunique() < 2:
        return {
            "delta_corr": 0.0, "p": 1.0, "n_low": 0, "n_high": 0,
            "corr_low": 0.0, "corr_high": 0.0, "median_split": None,
        }
    med = float(df[context_col].median())
    low = df[df[context_col] <= med]
    high = df[df[context_col] > med]

    def _r(sub):
        if len(sub) < MIN_SPLIT or sub["signal"].std() == 0 or sub["fwd"].std() == 0:
            return 0.0
        return float(np.corrcoef(sub["signal"], sub["fwd"])[0, 1])

    r_low, r_high = _r(low), _r(high)

    def _z(r, n):
        r = max(min(r, 0.999), -0.999)
        return 0.5 * np.log((1 + r) / (1 - r)), max(n - 3, 1)

    z1, n1 = _z(r_low, len(low))
    z2, n2 = _z(r_high, len(high))
    se = np.sqrt(1 / n1 + 1 / n2)
    p = float(2 * norm.sf(abs((z2 - z1) / se))) if se > 0 else 1.0
    return {
        "corr_low": round(r_low, 4),
        "corr_high": round(r_high, 4),
        "delta_corr": round(r_high - r_low, 4),
        "p": round(p, 4),
        "n_low": int(len(low)),
        "n_high": int(len(high)),
        "median_split": med,
    }


def _build_rows(
    closes: pd.DataFrame,
    sectors: dict,
    *,
    loc_start: int,
    loc_end: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Build per-(date,symbol) rows for confirmation/discovery windows."""
    rets = M.returns_panel(closes)
    scores = M.cross_sectional_momentum_scores(closes, lookback=LOOKBACK)
    fwd = M.forward_returns(closes, FWD_BARS)
    dates = list(closes.index)
    rows = []
    eval_dates = []
    for loc in range(loc_start, loc_end, STEP):
        dt = dates[loc]
        dstr = str(pd.Timestamp(dt).date())
        eval_dates.append(dstr)
        window_rets = rets.iloc[max(0, loc - LOOKBACK): loc + 1]
        corr = corr_from_returns(
            {s: window_rets[s].to_numpy() for s in window_rets.columns},
            min_overlap=30,
        )
        srow = scores.iloc[loc].dropna()
        if len(srow) < 8:
            continue
        port = list(srow.nlargest(5).index)
        net = analyze_network(
            corr, list(window_rets.columns),
            as_of=dstr, threshold=RHO, portfolio=port,
        )
        # Controls: pairwise cluster concentration + sector HHI of top-5
        clusters = clusters_from_corr(port, corr, threshold=RHO)
        pairwise_conc = max(len(g) for g in clusters) / max(len(port), 1)
        sec_counts = Counter(sectors.get(s, "UNK") for s in port)
        sector_hhi = sum((c / len(port)) ** 2 for c in sec_counts.values())

        frow = fwd.iloc[loc]
        for sym in srow.index:
            if sym not in frow.index or np.isnan(frow[sym]):
                continue
            rows.append({
                "date": dstr,
                "symbol": sym,
                "signal": float(srow[sym]),
                "fwd": float(frow[sym]),
                "net_conc": float(net.portfolio_network_concentration),
                "pairwise_conc": float(pairwise_conc),
                "sector_hhi": float(sector_hhi),
                "in_port": 1.0 if sym in port else 0.0,
            })
    return pd.DataFrame(rows), eval_dates


def certify_confirmation_sample(
    closes: pd.DataFrame,
    disc_dates: list[str],
    conf_dates: list[str],
) -> dict:
    """Protocol-scoped fitness for the confirmation holdout (inherits snapshot cert)."""
    overlap = sorted(set(disc_dates) & set(conf_dates))
    ok = (
        len(overlap) == 0
        and len(conf_dates) >= 5
        and closes.shape[1] == 29
    )
    return {
        "ok": ok,
        "status": "READY" if ok else "CONFIRMATION_DATA_NOT_AVAILABLE",
        "snapshot_id": CERTIFIED_SNAPSHOT_ID,
        "global_trust_class": "OPERATIONAL_ONLY",
        "scoped_parent_certification": "READY_FOR_SCIENTIFIC_RERUN",
        "sample_mode": CONFIRMATION_PROTOCOL["sample"]["mode"],
        "n_confirmation_eval_dates": len(conf_dates),
        "n_discovery_eval_dates": len(disc_dates),
        "overlap_eval_dates": overlap,
        "confirmation_date_first": conf_dates[0] if conf_dates else None,
        "confirmation_date_last": conf_dates[-1] if conf_dates else None,
        "discovery_date_first": disc_dates[0] if disc_dates else None,
        "discovery_date_last": disc_dates[-1] if disc_dates else None,
        "independence_proof": (
            "Confirmation evaluation dates are disjoint from discovery evaluation "
            "dates; each confirmation forward window ends before discovery oos_start "
            f"{OOS_START}."
        ),
        "forward_post_discovery_available": False,
        "forward_note": (
            "No certified sessions after discovery end (2026-08-11). "
            "Preferred later-period confirmation is NOT available."
        ),
        "required_if_unavailable": {
            "later_period": (
                "Additional certified NSE sessions after 2026-08-11 for the same "
                "29-name panel (enough for lookback=60 + multiple step=21 points "
                "+ 10d forward)."
            ),
            "or_broader_panel": (
                "Separately protocol-scoped certified broader panel with the same "
                "CA/identity/price gates, never used in discovery."
            ),
        },
    }


def _risk_economics(df: pd.DataFrame) -> dict:
    """Frozen economic/risk layer on above-median signal cohort."""
    if df.empty:
        return {"ok": False, "reason": "empty"}
    med_c = float(df["net_conc"].median())
    cohort = df[df["signal"] >= 0.5].copy()
    if len(cohort) < MIN_SPLIT:
        return {"ok": False, "reason": "underpowered_signal_cohort", "n": int(len(cohort))}
    high = cohort[cohort["net_conc"] > med_c]
    low = cohort[cohort["net_conc"] <= med_c]
    if len(high) < 10 or len(low) < 10:
        return {
            "ok": False, "reason": "underpowered_splits",
            "n_high": int(len(high)), "n_low": int(len(low)),
        }

    def loss_rate(sub):
        return float((sub["fwd"] < 0).mean()) if len(sub) else 0.0

    def p05(sub):
        return float(np.percentile(sub["fwd"], 5)) if len(sub) else 0.0

    def mean_neg(sub):
        neg = sub.loc[sub["fwd"] < 0, "fwd"]
        return float(neg.mean()) if len(neg) else 0.0

    lr_h, lr_l = loss_rate(high), loss_rate(low)
    t_h, t_l = p05(high), p05(low)
    loss_gap = lr_h - lr_l
    tail_gap = t_h - t_l
    # Opportunity cost of demoting high-concentration signal names
    demoted = cohort[cohort["net_conc"] > med_c]
    kept = cohort[cohort["net_conc"] <= med_c]
    opp = {
        "n_signal_cohort": int(len(cohort)),
        "n_would_demote": int(len(demoted)),
        "demote_rate": round(len(demoted) / max(len(cohort), 1), 4),
        "mean_fwd_demoted": round(float(demoted["fwd"].mean()) if len(demoted) else 0.0, 4),
        "mean_fwd_kept": round(float(kept["fwd"].mean()) if len(kept) else 0.0, 4),
        "mean_fwd_gap_kept_minus_demoted": round(
            float(kept["fwd"].mean() - demoted["fwd"].mean())
            if len(kept) and len(demoted) else 0.0,
            4,
        ),
        "cost_pct_reporting_only": round_trip_cost_pct("CNC"),
    }
    # Economic risk meaning per frozen OR rules
    risk_ok = (loss_gap >= 0.05) or (tail_gap <= -0.01)
    return {
        "ok": True,
        "median_net_conc": med_c,
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "loss_rate_high": round(lr_h, 4),
        "loss_rate_low": round(lr_l, 4),
        "loss_rate_gap": round(loss_gap, 4),
        "left_tail_p05_high": round(t_h, 4),
        "left_tail_p05_low": round(t_l, 4),
        "left_tail_gap": round(tail_gap, 4),
        "mean_negative_fwd_high": round(mean_neg(high), 4),
        "mean_negative_fwd_low": round(mean_neg(low), 4),
        "mean_fwd_high": round(float(high["fwd"].mean()), 4),
        "mean_fwd_low": round(float(low["fwd"].mean()), 4),
        "max_adverse_proxy_p05_high": round(t_h, 4),
        "economic_risk_meaning": risk_ok,
        "opportunity_cost": opp,
    }


def _incrementality(df: pd.DataFrame) -> dict:
    """Partial association after residualizing on pairwise + sector controls."""
    if len(df) < 60:
        return {"ok": False, "reason": "underpowered", "incremental": False}
    Y = df["fwd"].to_numpy(float)
    S = df["signal"].to_numpy(float)
    C = np.column_stack([
        np.ones(len(df)),
        df["pairwise_conc"].to_numpy(float),
        df["sector_hhi"].to_numpy(float),
    ])
    # Residualize signal and fwd on controls
    try:
        b_y = np.linalg.lstsq(C, Y, rcond=None)[0]
        b_s = np.linalg.lstsq(C, S, rcond=None)[0]
        ry = Y - C @ b_y
        rs = S - C @ b_s
    except Exception:
        return {"ok": False, "reason": "lstsq_failed", "incremental": False}
    resid = df.copy()
    resid["fwd"] = ry
    resid["signal"] = rs
    stats = _interaction_test(resid, "net_conc")
    incremental = bool(stats["delta_corr"] > 0 and stats["p"] < ALPHA
                       and stats["n_low"] >= MIN_SPLIT and stats["n_high"] >= MIN_SPLIT)
    return {
        "ok": True,
        "controls": ["pairwise_conc", "sector_hhi"],
        "residual_interaction": stats,
        "incremental": incremental,
    }


def _verdict(primary: dict, risk: dict, incr: dict) -> tuple[str, str]:
    underpowered = (
        primary.get("n_low", 0) < MIN_SPLIT or primary.get("n_high", 0) < MIN_SPLIT
        or not risk.get("ok", False)
    )
    stat_ok = (
        primary.get("delta_corr", 0) > 0
        and primary.get("p", 1) < ALPHA
        and primary.get("n_low", 0) >= MIN_SPLIT
        and primary.get("n_high", 0) >= MIN_SPLIT
    )
    if underpowered and not stat_ok:
        return "INCONCLUSIVE", "underpowered confirmation sample"
    if not stat_ok:
        return "FAILED_CONFIRMATION", (
            f"primary interaction not replicated "
            f"(delta_corr={primary.get('delta_corr')}, p={primary.get('p')})"
        )
    if not incr.get("incremental"):
        return "FAILED_CONFIRMATION", (
            "effect not incremental to pairwise correlation / sector controls"
        )
    if not risk.get("economic_risk_meaning"):
        return "INCONCLUSIVE", (
            "statistical replication without economically meaningful risk effect "
            "under frozen criteria"
        )
    return "CONFIRMED", "statistical + incremental + economic risk criteria met"


def _next_action(verdict: str) -> str:
    if verdict == "CONFIRMED":
        return "DESIGN_SEPARATE_POLICY_EXPERIMENT"
    if verdict == "FAILED_CONFIRMATION":
        return "REJECT"
    return "RETEST_ONLY_WITH_NEW_INDEPENDENT_DATA"


def run_confirmation() -> dict[str, Any]:
    # C1 — freeze protocol to disk BEFORE evaluation
    proto_path = write_frozen_protocol()
    rejected = freeze_rejected_branches()

    sid, pit, manifest, closes = load_certified_snapshot(CERTIFIED_SNAPSHOT_ID)
    ordered = [s for s in FROZEN_PANEL if s in closes.columns]
    if len(ordered) != 29:
        raise ValueError(f"panel size {len(ordered)} != 29")
    closes = closes[ordered]
    sectors = load_sectors()

    dates = list(closes.index)
    oos_i = next(
        i for i, d in enumerate(dates)
        if str(pd.Timestamp(d).date()) >= OOS_START
    )
    disc_start = max(LOOKBACK + 5, oos_i)
    disc_end = len(dates) - FWD_BARS - 2
    conf_start = LOOKBACK + 5
    conf_end = oos_i - FWD_BARS - 2  # forward window ends before oos

    # Independence: build date lists
    disc_dates = [
        str(pd.Timestamp(dates[loc]).date())
        for loc in range(disc_start, disc_end, STEP)
    ]
    conf_dates = [
        str(pd.Timestamp(dates[loc]).date())
        for loc in range(conf_start, conf_end, STEP)
    ]
    sample_cert = certify_confirmation_sample(closes, disc_dates, conf_dates)

    if not sample_cert["ok"]:
        payload = _unavailable_payload(sample_cert, proto_path, rejected, manifest)
        _write_outputs(payload)
        return payload

    conf_df, conf_eval_dates = _build_rows(
        closes, sectors, loc_start=conf_start, loc_end=conf_end,
    )
    # Re-verify independence on actual row dates
    disc_df, disc_eval_dates = _build_rows(
        closes, sectors, loc_start=disc_start, loc_end=disc_end,
    )
    row_overlap = sorted(set(conf_df["date"].unique()) & set(disc_df["date"].unique()))
    if row_overlap:
        sample_cert["ok"] = False
        sample_cert["status"] = "CONFIRMATION_DATA_NOT_AVAILABLE"
        sample_cert["row_date_overlap"] = row_overlap
        payload = _unavailable_payload(sample_cert, proto_path, rejected, manifest)
        _write_outputs(payload)
        return payload

    primary = _interaction_test(conf_df, "net_conc")
    risk = _risk_economics(conf_df)
    incr = _incrementality(conf_df)
    verdict, reason = _verdict(primary, risk, incr)
    next_action = _next_action(verdict)

    # Register NEW confirmation experiment (do not overwrite discovery id)
    hid = prereg.preregister(
        experiment_id="EXP-A6-CONF-01",
        hypothesis=(
            "Higher portfolio-network concentration materially worsens the risk "
            "profile of otherwise comparable 60d-momentum-rank signals; the "
            "signal×network_concentration interaction replicates out of the "
            "discovery OOS sample in the discovery direction (delta_corr>0) and "
            "remains incremental to pairwise/sector controls."
        ),
        null_hypothesis=(
            "The discovery signal×network_concentration effect does not replicate "
            "on the independent holdout, is not incremental to existing controls, "
            "or lacks economically meaningful risk content."
        ),
        success_criteria={
            "delta_corr_positive": {"eq": 1},
            "p_below_alpha": {"eq": 1},
            "incremental": {"eq": 1},
            "economic_risk_meaning": {"eq": 1},
        },
        data_window={
            "snapshot_id": sid,
            "confirmation_start": sample_cert["confirmation_date_first"],
            "confirmation_end": sample_cert["confirmation_date_last"],
            "discovery_oos_start": OOS_START,
            "parent_discovery_id": DISCOVERY["hypothesis_id"],
            "sample_mode": sample_cert["sample_mode"],
        },
        protocol=CONFIRMATION_PROTOCOL,
        seed=42,
        code_hash="phase_a6_conf_v1",
    )
    metrics = {
        "delta_corr_positive": 1 if primary["delta_corr"] > 0 else 0,
        "p_below_alpha": 1 if primary["p"] < ALPHA else 0,
        "incremental": 1 if incr.get("incremental") else 0,
        "economic_risk_meaning": 1 if risk.get("economic_risk_meaning") else 0,
        "delta_corr": primary["delta_corr"],
        "p": primary["p"],
        "live_behaviour_changed": 0,
    }
    reg = prereg.record(hid, metrics)

    if verdict == "FAILED_CONFIRMATION":
        prereg.remember_negative(
            f"EXP-A6-CONF-01 FAILED_CONFIRMATION: {reason}",
            signal="signal_x_network_concentration",
            evidence_n=int(len(conf_df)),
            notes="Close branch; do not mine alternative interactions.",
        )
    elif verdict == "INCONCLUSIVE":
        prereg.remember_watch(
            f"EXP-A6-CONF-01 INCONCLUSIVE: {reason}",
            signal="signal_x_network_concentration",
            evidence_n=int(len(conf_df)),
            ev_r=float(primary.get("delta_corr") or 0),
            hypothesis_id=hid,
            notes="Retest only with new independent data.",
        )
    else:
        prereg.remember_watch(
            f"EXP-A6-CONF-01 CONFIRMED: eligible for separate policy experiment only",
            signal="signal_x_network_concentration",
            evidence_n=int(len(conf_df)),
            ev_r=float(primary.get("delta_corr") or 0),
            hypothesis_id=hid,
            notes="NOT production authority.",
        )

    plain = _plain(verdict)
    payload = {
        "global_trust_class": "OPERATIONAL_ONLY",
        "phase_b_started": False,
        "production_behaviour_changed": False,
        "production_authority": False,
        "discovery": DISCOVERY,
        "rejected_branches_frozen": rejected,
        "frozen_protocol_path": str(proto_path),
        "frozen_protocol_sha256": _file_sha(proto_path),
        "experiment_id": "EXP-A6-CONF-01",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "snapshot_id": sid,
        "sample_certification": sample_cert,
        "n_confirmation_rows": int(len(conf_df)),
        "n_discovery_rows_reference": int(len(disc_df)),
        "confirmation_eval_dates": conf_eval_dates,
        "discovery_eval_dates": disc_eval_dates,
        "primary_metric": primary,
        "incrementality": incr,
        "risk_economics": risk,
        "metrics": metrics,
        "confirmation_verdict": verdict,
        "reason": reason,
        "next_action": next_action,
        "cost_pct": round_trip_cost_pct("CNC"),
        "git_sha": _git_sha(),
        "result_hash": _sha({"primary": primary, "risk": risk, "incr": incr, "verdict": verdict}),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "plain": plain,
        "manifest_excerpt": {
            k: manifest.get(k)
            for k in (
                "snapshot_id", "trust_class", "scoped_certification",
                "manifest_checksum", "equity_sha256", "adjustment_policy_version",
            )
        },
    }
    _write_outputs(payload)
    return payload


def _unavailable_payload(sample_cert, proto_path, rejected, manifest) -> dict:
    plain = render_layers(PlainCard(
        label="Confirmation data",
        state="NOT_ENOUGH_DATA",
        explanation=(
            "QuantTerm cannot yet re-test the portfolio-overlap finding on a new, "
            "independent stretch of verified history. Waiting for more data is safer "
            "than reusing the same dates that produced the first result."
        ),
        implication="Do not use the finding. Do not begin a policy experiment.",
        technical="CONFIRMATION_DATA_NOT_AVAILABLE",
    ))
    return {
        "global_trust_class": "OPERATIONAL_ONLY",
        "phase_b_started": False,
        "production_behaviour_changed": False,
        "production_authority": False,
        "discovery": DISCOVERY,
        "rejected_branches_frozen": rejected,
        "frozen_protocol_path": str(proto_path),
        "frozen_protocol_sha256": _file_sha(proto_path),
        "experiment_id": "EXP-A6-CONF-01",
        "hypothesis_id": None,
        "confirmation_verdict": "CONFIRMATION_DATA_NOT_AVAILABLE",
        "reason": "No independent research-safe confirmation sample available",
        "next_action": "RETEST_ONLY_WITH_NEW_INDEPENDENT_DATA",
        "sample_certification": sample_cert,
        "plain": plain,
        "git_sha": _git_sha(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_excerpt": {
            k: (manifest or {}).get(k)
            for k in ("snapshot_id", "trust_class", "scoped_certification")
        },
    }


def _plain(verdict: str) -> dict:
    if verdict == "CONFIRMED":
        card = PlainCard(
            label="Portfolio overlap risk check",
            state="PROMISING",
            explanation=(
                "We tested that finding again on new historical data and it held up. "
                "Portfolio overlap may contain useful risk information. It is still "
                "not being used for real trades."
            ),
            implication="Eligible only for a separate policy/value-add experiment — not live use.",
            technical=f"confirmation_verdict={verdict}",
        )
    elif verdict == "FAILED_CONFIRMATION":
        card = PlainCard(
            label="Portfolio overlap risk check",
            state="FAILED",
            explanation=(
                "The earlier result did not repeat on new data, so QuantTerm will not use it."
            ),
            implication="Close this research branch. Do not mine similar interactions.",
            technical=f"confirmation_verdict={verdict}",
        )
    else:
        card = PlainCard(
            label="Portfolio overlap risk check",
            state="NOT_ENOUGH_DATA",
            explanation=(
                "QuantTerm noticed that some otherwise-good trade signals looked "
                "different when the portfolio was already crowded into stocks behaving "
                "similarly, but the new-data check is still inconclusive."
            ),
            implication="Retest only when a new independent sample is available.",
            technical=f"confirmation_verdict={verdict}",
        )
    return render_layers(card)


def render_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Phase A.6 — Network Interaction Confirmation")
    lines.append("")
    lines.append(
        "> Independent confirmation of the sole surviving Phase A.5 finding "
        "(`signal_x_network_concentration`). "
        "**Not Phase B. Production unchanged. Not a BUY/SELL signal.**"
    )
    lines.append("")
    plain = (payload.get("plain") or {}).get("layer1") or {}
    lines.append("## Plain English")
    lines.append("")
    lines.append(
        "QuantTerm noticed that some otherwise-good trade signals performed differently "
        "when the portfolio was already crowded into stocks behaving similarly."
    )
    lines.append("")
    lines.append(plain.get("explanation") or "")
    if plain.get("implication"):
        lines.append("")
        lines.append(plain["implication"])
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Discovery result reference")
    lines.append("")
    d = payload.get("discovery") or DISCOVERY
    lines.append(f"- Parent experiment: `{d.get('experiment_id')}` / `{d.get('hypothesis_id')}`")
    lines.append(f"- Snapshot: `{d.get('snapshot_id')}`")
    lines.append(f"- Surviving interaction: `{d.get('surviving_interaction')}`")
    lines.append(
        f"- Discovery delta_corr={d.get('discovery_delta_corr')} "
        f"p={d.get('discovery_p')} (FDR-cleared)"
    )
    lines.append(f"- Interpretation: {d.get('interpretation')}")
    lines.append("")
    lines.append("### Rejected branches (frozen)")
    lines.append("")
    for b in REJECTED_BRANCHES:
        lines.append(
            f"- `{b['experiment_id']}` `{b['branch']}` → **REJECT** "
            f"(do not retune / escalate)"
        )
    lines.append("")
    lines.append("## 2. Frozen confirmation hypothesis")
    lines.append("")
    lines.append(
        f"- Protocol file: `{payload.get('frozen_protocol_path')}`"
    )
    lines.append(f"- Protocol sha256: `{payload.get('frozen_protocol_sha256')}`")
    lines.append(
        "- Primary H1: higher `portfolio_network_concentration` context changes "
        "signal→10d-forward association in the **discovery direction** "
        "(`delta_corr > 0`), as risk/context — not standalone alpha."
    )
    lines.append(
        "- No post-hoc `network_concentration > X` threshold invented from "
        "confirmation data (within-sample median split only)."
    )
    lines.append("")
    lines.append("## 3. New experiment ID")
    lines.append("")
    lines.append(f"- Experiment: `{payload.get('experiment_id')}`")
    lines.append(f"- Hypothesis ID: `{payload.get('hypothesis_id')}`")
    lines.append(f"- Registry status: `{payload.get('registry_status')}`")
    lines.append("- Does **not** overwrite `EXP-A5A6-01` / `3734b8a0a9124a60`.")
    lines.append("")
    lines.append("## 4. Confirmation dataset certification")
    lines.append("")
    sc = payload.get("sample_certification") or {}
    lines.append(f"- Status: `{sc.get('status')}`")
    lines.append(f"- Global trust: `{sc.get('global_trust_class')}`")
    lines.append(f"- Parent scoped cert: `{sc.get('scoped_parent_certification')}`")
    lines.append(f"- Snapshot: `{sc.get('snapshot_id')}`")
    lines.append(f"- Sample mode: `{sc.get('sample_mode')}`")
    lines.append(f"- Forward post-discovery available: `{sc.get('forward_post_discovery_available')}`")
    lines.append(f"- Note: {sc.get('forward_note')}")
    lines.append("")
    lines.append("## 5. Proof of sample independence")
    lines.append("")
    lines.append(sc.get("independence_proof") or "")
    lines.append("")
    lines.append(
        f"- Discovery eval dates: {sc.get('discovery_date_first')} → "
        f"{sc.get('discovery_date_last')} (n={sc.get('n_discovery_eval_dates')})"
    )
    lines.append(
        f"- Confirmation eval dates: {sc.get('confirmation_date_first')} → "
        f"{sc.get('confirmation_date_last')} (n={sc.get('n_confirmation_eval_dates')})"
    )
    lines.append(f"- Overlap: `{sc.get('overlap_eval_dates')}`")
    if payload.get("confirmation_verdict") == "CONFIRMATION_DATA_NOT_AVAILABLE":
        lines.append("")
        lines.append("### Required additional data")
        lines.append("")
        req = sc.get("required_if_unavailable") or {}
        for k, v in req.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines.append("## 6. Exact metric definitions")
    lines.append("")
    lines.append(
        "- Signal: 60d cross-sectional momentum rank on frozen 29-name panel."
    )
    lines.append(
        "- Network concentration: Herfindahl of correlation-community weights "
        "(ρ≥0.70) for top-5 momentum names, 60d lookback."
    )
    lines.append(
        "- Primary: median-split Fisher-z δcorr(signal, 10d fwd) high−low."
    )
    lines.append(
        "- Controls: top-5 pairwise cluster concentration + sector HHI."
    )
    lines.append(
        "- Economic risk: among signal≥0.5, require loss-rate gap ≥5pp OR "
        "left-tail (p05) gap ≤ −1pp for high vs low concentration."
    )
    lines.append("")
    if payload.get("confirmation_verdict") == "CONFIRMATION_DATA_NOT_AVAILABLE":
        lines.append("## 7–11. Skipped — confirmation data not available")
        lines.append("")
    else:
        lines.append("## 7. Baseline / control comparison")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload.get("incrementality"), indent=2, default=str))
        lines.append("```")
        lines.append("")
        lines.append("## 8. Statistical result")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload.get("primary_metric"), indent=2, default=str))
        lines.append("```")
        lines.append("")
        lines.append(
            f"- Discovery reference delta_corr={d.get('discovery_delta_corr')} "
            f"vs confirmation delta_corr="
            f"{(payload.get('primary_metric') or {}).get('delta_corr')}"
        )
        lines.append("")
        lines.append("## 9. Economic / risk result")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload.get("risk_economics"), indent=2, default=str))
        lines.append("```")
        lines.append("")
        lines.append("## 10. Opportunity-cost analysis")
        lines.append("")
        opp = ((payload.get("risk_economics") or {}).get("opportunity_cost") or {})
        lines.append("```json")
        lines.append(json.dumps(opp, indent=2, default=str))
        lines.append("```")
        lines.append("")
        lines.append("## 11. Regime / subperiod diagnostics")
        lines.append("")
        lines.append(
            "Not preregistered beyond the single holdout window; no post-hoc "
            "regime mining performed."
        )
        lines.append("")
    lines.append("## 12. Confirmation verdict")
    lines.append("")
    lines.append(f"- **{payload.get('confirmation_verdict')}**")
    lines.append(f"- Reason: {payload.get('reason')}")
    lines.append(f"- Next action: `{payload.get('next_action')}`")
    lines.append("")
    lines.append("## 13. Scientific-memory update")
    lines.append("")
    lines.append(
        "- Four rejected A.5 branches frozen as negative evidence."
    )
    lines.append(
        f"- Confirmation outcome recorded under `{payload.get('hypothesis_id')}` "
        "(discovery id untouched)."
    )
    lines.append("")
    lines.append("## 14. Production behaviour confirmation")
    lines.append("")
    lines.append(f"- production_behaviour_changed: `{payload.get('production_behaviour_changed')}`")
    lines.append(f"- production_authority: `{payload.get('production_authority')}`")
    lines.append(f"- phase_b_started: `{payload.get('phase_b_started')}`")
    lines.append(
        "- Brain / ranking / sizing / risk vetoes / execution / broker: **unchanged**."
    )
    lines.append("")
    lines.append("## 15. Plain-English explanation")
    lines.append("")
    lines.append(plain.get("explanation") or "")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Final matrix")
    lines.append("")
    primary = payload.get("primary_metric") or {}
    incr = payload.get("incrementality") or {}
    risk = payload.get("risk_economics") or {}
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| DISCOVERY RESULT | PASS_RISK / `{d.get('surviving_interaction')}` "
        f"δcorr={d.get('discovery_delta_corr')} |"
    )
    lines.append(
        f"| CONFIRMATION RESULT | `{payload.get('confirmation_verdict')}` |"
    )
    lines.append(
        f"| INCREMENTAL TO EXISTING CONTROLS? | "
        f"`{incr.get('incremental') if incr else 'N/A'}` |"
    )
    lines.append(
        f"| ECONOMICALLY MEANINGFUL? | "
        f"`{risk.get('economic_risk_meaning') if risk else 'N/A'}` |"
    )
    lines.append(f"| FINAL VERDICT | **{payload.get('confirmation_verdict')}** |")
    lines.append(f"| NEXT ACTION | `{payload.get('next_action')}` |")
    lines.append("")
    lines.append(
        "STOP. Do not begin the policy experiment. Do not begin Phase B."
    )
    lines.append("")
    lines.append(f"_Evaluated at: {payload.get('evaluated_at')}_")
    lines.append(f"_git_sha: `{payload.get('git_sha')}`_")
    lines.append("")
    return "\n".join(lines)


def _write_outputs(payload: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    payload["report_path"] = str(OUT_MD)
    payload["results_path"] = str(OUT_JSON)


if __name__ == "__main__":
    out = run_confirmation()
    print(json.dumps({
        "experiment_id": out.get("experiment_id"),
        "hypothesis_id": out.get("hypothesis_id"),
        "confirmation_verdict": out.get("confirmation_verdict"),
        "next_action": out.get("next_action"),
        "sample_status": (out.get("sample_certification") or {}).get("status"),
        "production_behaviour_changed": out.get("production_behaviour_changed"),
        "phase_b_started": out.get("phase_b_started"),
        "report_path": out.get("report_path"),
        "primary": out.get("primary_metric"),
        "incremental": (out.get("incrementality") or {}).get("incremental"),
        "economic_risk_meaning": (out.get("risk_economics") or {}).get(
            "economic_risk_meaning"
        ),
    }, indent=2, default=str))
