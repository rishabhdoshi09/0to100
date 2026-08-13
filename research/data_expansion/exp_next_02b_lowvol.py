"""EXP-NEXT-02B — Low-volatility retest on expanded certified panel (Path B primary).

Frozen BEFORE outcome inspection. Does not overwrite EXP-NEXT-02.
Production behaviour unchanged. No ML.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.intelligence.data.pit_contract import PitContract
from research.intelligence.data.snapshot_store import SnapshotStore
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg
from research.phase_a5.scoped_certification import FROZEN_PANEL
from research.phase_next import eval_utils as E
from research.phase_next import protocol as P0

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPANDED_SNAP_ROOT = REPO_ROOT / "logs" / "research_expansion" / "snapshots"
PRIOR_SNAP_ROOT = REPO_ROOT / "logs" / "phase_a5_scoped" / "snapshots"
FROZEN_PATH = REPO_ROOT / "docs" / "overhaul" / "EXP_NEXT_02B_FROZEN_PROTOCOL.json"
OUT_MD = REPO_ROOT / "EXP_NEXT_02_LOW_VOL_EXPANDED_RETEST.md"
OUT_JSON = REPO_ROOT / "logs" / "research_expansion" / "exp_next_02b_result.json"

EXPERIMENT_ID = "EXP-NEXT-02B"
SNAPSHOT_ID = "2f683be0c73eaa33"
PRIOR_SNAPSHOT_ID = "a7a9828ec37e09e4"

# Frozen partitions (also in FROZEN_PATH) — do not change after outcome inspection
DISCOVERY_START = "2021-01-01"
DISCOVERY_END = "2023-12-31"
CONFIRM_START = "2024-01-01"

# Identical economic knobs to EXP-NEXT-02
LOOKBACK = P0.LOWVOL_LOOKBACK
REBALANCE = P0.LOWVOL_REBALANCE
HOLD = P0.LOWVOL_HOLD
Q = P0.LOWVOL_Q


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def load_frozen() -> dict:
    return json.loads(FROZEN_PATH.read_text())


def load_expanded_panel(snapshot_id: str = SNAPSHOT_ID) -> tuple[pd.DataFrame, dict, PitContract]:
    """Load scoped expanded snapshot closes (PIT-safe; no global store mix-in)."""
    store = SnapshotStore(EXPANDED_SNAP_ROOT)
    ok, fails = store.verify_snapshot(snapshot_id)
    if not ok:
        raise ValueError(f"snapshot {snapshot_id} verify failed: {fails}")
    snap = store.open_snapshot(snapshot_id)
    manifest = dict(snap.manifest)
    if manifest.get("snapshot_id") != snapshot_id:
        raise ValueError("snapshot_id mismatch")
    if manifest.get("scoped_certification") != "SCOPED_RESEARCH_READY":
        raise ValueError(
            f"snapshot not SCOPED_RESEARCH_READY: {manifest.get('scoped_certification')}"
        )
    if manifest.get("global_trust_class") != "OPERATIONAL_ONLY":
        raise ValueError("refusing unexpected global trust upgrade")

    panel = [str(s).upper() for s in (manifest.get("panel") or [])]
    if len(panel) < 100:
        raise ValueError(f"expanded panel unexpectedly small: {len(panel)}")

    by_sym: dict[str, dict[str, float]] = {}
    for r in snap._equity:
        sym = str(r["symbol"]).upper()
        if sym not in panel and panel:
            continue
        by_sym.setdefault(sym, {})[r["date"]] = float(r["close"])
    closes = pd.DataFrame({sym: pd.Series(vals) for sym, vals in by_sym.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index()
    # Keep date-wise availability (do NOT dropna-any — would collapse breadth)
    closes = closes.dropna(how="all")
    ordered = [s for s in panel if s in closes.columns]
    closes = closes[ordered]

    pit = PitContract.from_store(store, snapshot_id)
    mid = closes.index[len(closes.index) // 2].strftime("%Y-%m-%d")
    sample = ordered[0]
    read = pit.as_of("bars", when=mid, symbol=sample)
    if not read.usable:
        raise ValueError(f"PitContract unusable: {read}")
    if any(getattr(b, "date", "") > mid for b in (read.data or [])):
        raise ValueError("PIT violation: future bar leaked")

    return closes, manifest, pit


def load_prior_29_panel() -> tuple[pd.DataFrame, dict]:
    store = SnapshotStore(PRIOR_SNAP_ROOT)
    ok, fails = store.verify_snapshot(PRIOR_SNAPSHOT_ID)
    if not ok:
        raise ValueError(f"prior snapshot verify failed: {fails}")
    snap = store.open_snapshot(PRIOR_SNAPSHOT_ID)
    manifest = dict(snap.manifest)
    if manifest.get("scoped_certification") != "READY_FOR_SCIENTIFIC_RERUN":
        raise ValueError("prior snapshot not scoped-certified")
    by_sym: dict[str, dict[str, float]] = {}
    for r in snap._equity:
        sym = str(r["symbol"]).upper()
        by_sym.setdefault(sym, {})[r["date"]] = float(r["close"])
    closes = pd.DataFrame({sym: pd.Series(vals) for sym, vals in by_sym.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="any")
    ordered = [s for s in FROZEN_PANEL if s in closes.columns]
    if len(ordered) != 29:
        raise ValueError(f"prior panel size {len(ordered)} != 29")
    return closes[ordered], manifest


def period_index(
    closes: pd.DataFrame, start: str | None, end: str | None
) -> pd.DatetimeIndex:
    idx = closes.index
    mask = pd.Series(True, index=idx)
    if start:
        mask &= idx >= pd.Timestamp(start)
    if end:
        mask &= idx <= pd.Timestamp(end)
    return idx[mask]


def _ann_factor() -> float:
    return (252 / HOLD) ** 0.5


def _sharpe(x: pd.Series) -> float:
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2 or float(x.std(ddof=1)) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * _ann_factor())


def _sortino(x: pd.Series) -> float:
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2:
        return 0.0
    downside = x[x < 0]
    if len(downside) < 1 or float(downside.std(ddof=1)) == 0:
        return 0.0 if float(x.mean()) <= 0 else float("inf")
    return float(x.mean() / downside.std(ddof=1) * _ann_factor())


def evaluate_partition(
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    label: str,
) -> dict[str, Any]:
    vol = E.realized_vol(closes, LOOKBACK)
    inv_vol_rank = (-vol).rank(axis=1, pct=True)
    fwd = M.forward_returns(closes, HOLD)

    ordered = list(dates)
    # Skip early dates inside partition until vol lookback is fully available
    reb = []
    for dt in ordered[::REBALANCE]:
        if dt not in inv_vol_rank.index or dt not in fwd.index:
            continue
        row = inv_vol_rank.loc[dt].dropna()
        if len(row) < max(10, int(1 / Q) * 3):
            continue
        reb.append(dt)
    reb_idx = pd.DatetimeIndex(reb)

    gross = E.long_short_period(
        inv_vol_rank, fwd, reb_idx, invert=False, top_q=Q,
    )
    net = E.net_of_costs(gross)
    pack = E.pack_stream(net, n_trials=1)
    pack["mean_gross"] = round(float(gross.mean()) if len(gross) else 0.0, 6)
    pack["mean_net"] = round(float(net.mean()) if len(net) else 0.0, 6)
    pack["cost_drag"] = round(E.cost_pct() / 100.0 * P0.TURNOVER_ONE_WAY, 6)
    # pack_stream already sets cost_drag via M.cost_drag — keep consistent
    pack["cost_drag"] = round(M.cost_drag(P0.TURNOVER_ONE_WAY, E.cost_pct()), 6)

    long_only = []
    ew = []
    cohort_sizes = []
    for dt in reb_idx:
        if dt not in inv_vol_rank.index or dt not in fwd.index:
            continue
        s = inv_vol_rank.loc[dt].dropna()
        f = fwd.loc[dt].reindex(s.index).dropna()
        s = s.reindex(f.index).dropna()
        if len(s) < 6:
            continue
        n = max(1, int(len(s) * Q))
        cohort_sizes.append(n)
        long_only.append(float(f.loc[s.nlargest(n).index].mean()))
        ew.append(float(f.mean()))
    lo = pd.Series(long_only, dtype=float)
    ew_s = pd.Series(ew, dtype=float)
    lo_net = lo - pack["cost_drag"]  # approx one-way long-only cost same drag assumption

    # Benchmark-relative: low-vol long-only minus EW (gross)
    bm_rel = lo - ew_s if len(lo) and len(ew_s) else pd.Series(dtype=float)

    verdict = E.map_discovery_verdict(
        pack["verdict"], mean_net=pack["mean_net"], fdr_ok=True,
    )
    # Explicit economic gate (frozen): mean_net <= 0 → FAIL
    if pack["mean_net"] <= 0:
        verdict = "FAIL"
    elif pack["verdict"] == "PROMOTE" and pack["mean_net"] > 0:
        verdict = "PASS"
    elif pack["mean_net"] > 0 and pack["verdict"] in {"UNDERPOWERED", "INCONCLUSIVE"}:
        verdict = "INCONCLUSIVE"
    elif pack["verdict"] == "REJECT":
        verdict = "FAIL"

    downside = float(net[net < 0].std(ddof=1)) if (net < 0).sum() >= 2 else 0.0

    return {
        "label": label,
        "n_rebalances": int(len(gross)),
        "median_cohort_size": int(np.median(cohort_sizes)) if cohort_sizes else 0,
        "mean_names_scored": (
            int(np.mean([
                int(inv_vol_rank.loc[dt].dropna().shape[0])
                for dt in reb_idx if dt in inv_vol_rank.index
            ])) if len(reb_idx) else 0
        ),
        "pack": pack,
        "mean_gross": pack["mean_gross"],
        "cost_drag": pack["cost_drag"],
        "mean_net": pack["mean_net"],
        "sharpe_net": round(_sharpe(net), 4),
        "sortino_net": round(_sortino(net), 4) if _sortino(net) != float("inf") else None,
        "downside_std": round(downside, 6),
        "max_drawdown": pack["max_drawdown"],
        "hit_rate": pack["hit_rate"],
        "long_only_sharpe_proxy_gross": round(_sharpe(lo), 4),
        "ew_sharpe_proxy_gross": round(_sharpe(ew_s), 4),
        "long_only_minus_ew_mean_gross": round(float(bm_rel.mean()) if len(bm_rel) else 0.0, 6),
        "turnover_one_way_assumed": P0.TURNOVER_ONE_WAY,
        "verdict": verdict,
        "rebalance_dates_first_last": (
            [str(reb_idx[0].date()), str(reb_idx[-1].date())] if len(reb_idx) else None
        ),
    }


def subperiod_stability(closes: pd.DataFrame, dates: pd.DatetimeIndex) -> list[dict]:
    """Calendar-year stability inside a partition (descriptive; not a new primary)."""
    rows = []
    years = sorted({int(ts.year) for ts in dates})
    for y in years:
        sub = dates[(dates >= pd.Timestamp(f"{y}-01-01")) & (dates <= pd.Timestamp(f"{y}-12-31"))]
        if len(sub) < REBALANCE * 2:
            continue
        ev = evaluate_partition(closes, sub, label=f"year_{y}")
        rows.append({
            "year": y,
            "n_rebalances": ev["n_rebalances"],
            "mean_gross": ev["mean_gross"],
            "mean_net": ev["mean_net"],
            "sharpe_net": ev["sharpe_net"],
            "verdict": ev["verdict"],
        })
    return rows


def run_path_b() -> dict[str, Any]:
    frozen = load_frozen()
    closes, manifest, pit = load_expanded_panel(SNAPSHOT_ID)
    disc_dates = period_index(closes, DISCOVERY_START, DISCOVERY_END)
    conf_dates = period_index(closes, CONFIRM_START, None)

    # Sample-size facts (pre-outcome structural)
    security_sessions = int(closes.notna().sum().sum())
    sample = {
        "n_securities": int(closes.shape[1]),
        "n_sessions_full": int(len(closes)),
        "security_sessions": security_sessions,
        "security_years": round(security_sessions / 252.0, 1),
        "discovery_sessions": int(len(disc_dates)),
        "confirm_sessions": int(len(conf_dates)),
        "date_range": [str(closes.index.min().date()), str(closes.index.max().date())],
    }

    discovery = evaluate_partition(closes, disc_dates, label="discovery")
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = evaluate_partition(closes, conf_dates, label="confirmation")

    final = E.final_after_confirm(
        discovery["verdict"],
        None if confirmation is None else confirmation["verdict"],
    )
    # If discovery PASS but confirmation underpowered/inconclusive → keep mapping
    # If partitions were too thin for confirm by design we'd cap — here confirm is powered

    stability = subperiod_stability(closes, disc_dates)

    hid = prereg.preregister(
        experiment_id=EXPERIMENT_ID,
        hypothesis=frozen["hypothesis"],
        null_hypothesis=frozen["null_hypothesis"],
        success_criteria={
            "discovery_pass": {"eq": 1},
            "confirm_pass": {"eq": 1},
            "mean_net": {"gt": 0.0},
        },
        data_window={
            "snapshot_id": SNAPSHOT_ID,
            "discovery": f"{DISCOVERY_START}→{DISCOVERY_END}",
            "confirm": f"{CONFIRM_START}→end",
            "vol_lookback": LOOKBACK,
            "hold": HOLD,
            "rebalance": REBALANCE,
            "path": "B_PRIMARY",
        },
        protocol={
            "type": "ALPHA",
            "portfolio": "long low-vol quintile / short high-vol quintile",
            "rebalance_every": REBALANCE,
            "costs": "CNC round_trip",
            "multiple_testing": "single primary specification; DSR n_trials=1",
            "protocol_version": frozen["protocol_version"],
            "production_authority": False,
            "parent_experiment": "EXP-NEXT-02",
        },
        seed=42,
        code_hash=frozen["code_hash"],
    )

    metrics = {
        "discovery_pass": 1 if discovery["verdict"] == "PASS" else 0,
        "confirm_pass": 1 if confirmation and confirmation["verdict"] == "PASS" else 0,
        "mean_net": discovery["mean_net"],
        "live_behaviour_changed": 0,
    }
    reg = prereg.record(hid, metrics)

    next_action = "NONE"
    if final == "CONFIRMED":
        next_action = "ELIGIBLE_FOR_FOLLOWUP_VALIDATION"
        prereg.remember_watch(
            f"{EXPERIMENT_ID} CONFIRMED net_disc={discovery['mean_net']} "
            f"net_conf={confirmation['mean_net'] if confirmation else None}",
            signal="low_volatility_effect_expanded",
            evidence_n=int(discovery["pack"]["n"]),
            ev_r=float(discovery["mean_net"]),
            hypothesis_id=hid,
            notes="Not production. Follow-up validation required before any live consideration.",
        )
    elif final in {"FAIL", "FAILED_CONFIRMATION"}:
        next_action = "REJECT_CLOSE_BRANCH"
        prereg.remember_negative(
            f"{EXPERIMENT_ID} {final}: expanded-panel low-vol not confirmed "
            f"(disc_net={discovery['mean_net']})",
            signal="low_volatility_effect",
            evidence_n=int(discovery["pack"]["n"]),
            notes="Do not rescue with ML or factor combinations. Branch closed.",
        )
    elif final == "INCONCLUSIVE":
        next_action = "HOLD_NO_TUNING"
        prereg.remember_watch(
            f"{EXPERIMENT_ID} INCONCLUSIVE net={discovery['mean_net']}",
            signal="low_volatility_effect_expanded",
            evidence_n=int(discovery["pack"]["n"]),
            ev_r=float(discovery["mean_net"]),
            hypothesis_id=hid,
            notes="No tuning. No ML.",
        )
    elif final == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION":
        next_action = "WAIT_FOR_INDEPENDENT_CONFIRMATION_SAMPLE"
        prereg.remember_watch(
            f"{EXPERIMENT_ID} discovery PASS needs future confirmation",
            signal="low_volatility_effect_expanded",
            evidence_n=int(discovery["pack"]["n"]),
            ev_r=float(discovery["mean_net"]),
            hypothesis_id=hid,
            notes="Do not promote. Do not tune.",
        )

    return {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "snapshot_id": SNAPSHOT_ID,
        "manifest_scoped_certification": manifest.get("scoped_certification"),
        "global_trust_class": "OPERATIONAL_ONLY",
        "path": "B_PRIMARY",
        "partitions": {
            "discovery": f"{DISCOVERY_START}→{DISCOVERY_END}",
            "confirm": f"{CONFIRM_START}→panel_end",
        },
        "sample": sample,
        "cost_pct_points": E.cost_pct(),
        "discovery": discovery,
        "confirmation": confirmation,
        "subperiod_stability_discovery": stability,
        "final_verdict": final,
        "next_action": next_action,
        "production_authority": False,
        "git_sha": _git_sha(),
        "result_hash": E.result_hash({
            "d": discovery["verdict"],
            "c": None if confirmation is None else confirmation["verdict"],
            "f": final,
            "net": discovery["mean_net"],
        }),
        "pit_ok": True,
    }


def run_path_a_secondary() -> dict[str, Any]:
    """Original 29-name snapshot + original partitions — robustness only."""
    closes, manifest = load_prior_29_panel()
    disc = period_index(closes, P0.DISCOVERY_START, P0.DISCOVERY_END)
    conf = period_index(closes, P0.CONFIRM_START, None)
    discovery = evaluate_partition(closes, disc, label="path_a_discovery")
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = evaluate_partition(closes, conf, label="path_a_confirmation")
    final = E.final_after_confirm(
        discovery["verdict"],
        None if confirmation is None else confirmation["verdict"],
    )
    return {
        "role": "SECONDARY_ROBUSTNESS_EVIDENCE",
        "experiment_ref": "EXP-NEXT-02 reproducibility surface",
        "snapshot_id": PRIOR_SNAPSHOT_ID,
        "n_securities": 29,
        "partitions": {
            "discovery": f"{P0.DISCOVERY_START}→{P0.DISCOVERY_END}",
            "confirm": f"{P0.CONFIRM_START}→end",
        },
        "discovery": discovery,
        "confirmation": confirmation,
        "final_verdict": final,
        "cannot_override_primary": True,
        "scoped_certification": manifest.get("scoped_certification"),
    }


def _plain_sections(final: str, discovery: dict, confirmation: dict | None) -> dict[str, str]:
    what_tested = (
        "Do calmer stocks give investors better returns for the amount of risk taken?"
    )
    if final == "CONFIRMED":
        happened = (
            "On the larger certified stock group, quieter stocks beat louder stocks "
            "after costs in the discovery period, and the same pattern held in a later "
            "untouched confirmation period."
        )
        means = (
            "The low-volatility idea looks real on this certified history — but it is "
            "not approved for live trading yet."
        )
        will_do = (
            "The finding is worth further testing, but it will not be used for real "
            "trades yet."
        )
    elif final == "FAILED_CONFIRMATION":
        happened = (
            "Quieter stocks looked helpful in the first test window, but that result "
            "did not hold up in the later untouched window."
        )
        means = "The idea does not clear QuantTerm's confirmation bar."
        will_do = "Nothing. QuantTerm will not use this idea."
    elif final == "FAIL":
        happened = (
            "After realistic costs, quieter stocks did not deliver a useful edge over "
            "louder stocks under the frozen rules."
        )
        means = "The evidence rejects the low-volatility hypothesis on this dataset."
        will_do = "Nothing. QuantTerm will not use this idea."
    elif final == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION":
        happened = (
            "The first out-of-sample window looked promising after costs, but there "
            "is not yet a strong enough independent confirmation sample."
        )
        means = "Promising, not proven."
        will_do = (
            "Wait for more independent history before treating this as confirmed. "
            "No live use. No tuning."
        )
    else:
        happened = (
            "Even with more stocks and more years, the evidence was still mixed or "
            "too weak to decide cleanly under the frozen rules."
        )
        means = "Still uncertain — not a green light and not a clean rejection."
        will_do = "No tuning. No live use. Do not keep mining variants."
    return {
        "what_we_tested": what_tested,
        "what_happened": happened,
        "what_it_means": means,
        "what_quantterm_will_do": will_do,
    }


def write_report(primary: dict, path_a: dict | None, frozen: dict) -> Path:
    plain = _plain_sections(
        primary["final_verdict"], primary["discovery"], primary.get("confirmation")
    )
    disc = primary["discovery"]
    conf = primary.get("confirmation")
    lines = [
        "# EXP-NEXT-02B — Low-Volatility Expanded Retest",
        "",
        "> Scientific report. Production unchanged. Not a live trading authorization.",
        "> Global trust remains `OPERATIONAL_ONLY`. Phase B not started. No ML.",
        "",
        "## WHAT WE TESTED",
        "",
        plain["what_we_tested"],
        "",
        "## WHAT HAPPENED",
        "",
        plain["what_happened"],
        "",
        "## WHAT IT MEANS",
        "",
        plain["what_it_means"],
        "",
        "## WHAT QUANTTERM WILL DO",
        "",
        plain["what_quantterm_will_do"],
        "",
        "---",
        "",
        "## 1. Previous 29-name result",
        "",
        "| Field | Value |",
        "|-------|--------|",
        "| Experiment | `EXP-NEXT-02` |",
        "| Hypothesis ID | `5eb01b27fc75b885` |",
        "| Snapshot | `a7a9828ec37e09e4` |",
        "| Final | **INCONCLUSIVE** |",
        "| Next | `HOLD_NO_TUNING` |",
        "| Discovery n_rebalances | 13 (UNDERPOWERED) |",
        "| Discovery mean_net | +0.0076 |",
        "",
        "## 2. New experiment ID",
        "",
        f"- **`{primary['experiment_id']}`** (does not overwrite EXP-NEXT-02)",
        f"- Hypothesis ID: `{primary['hypothesis_id']}`",
        f"- Protocol version: `{frozen['protocol_version']}`",
        f"- Result hash: `{primary['result_hash']}`",
        "",
        "## 3. Expanded snapshot certification",
        "",
        f"- Snapshot: `{primary['snapshot_id']}`",
        f"- Scoped certification: `{primary['manifest_scoped_certification']}`",
        f"- Global trust: `{primary['global_trust_class']}`",
        f"- Securities: **{primary['sample']['n_securities']}**",
        f"- Sessions (full): **{primary['sample']['n_sessions_full']}**",
        f"- Security-years: **{primary['sample']['security_years']}**",
        f"- Date range: {primary['sample']['date_range'][0]} → {primary['sample']['date_range'][1]}",
        f"- PIT smoke: `{primary['pit_ok']}`",
        "",
        "## 4. Frozen hypothesis",
        "",
        f"- Plain: {frozen['hypothesis_plain']}",
        f"- Technical: {frozen['hypothesis']}",
        f"- Null: {frozen['null_hypothesis']}",
        f"- Frozen file: `docs/overhaul/EXP_NEXT_02B_FROZEN_PROTOCOL.json`",
        f"- Frozen before outcomes: `{frozen['frozen_before_outcome_inspection']}`",
        "",
        "## 5. Protocol",
        "",
        f"- Vol lookback: **{LOOKBACK}d** realized vol",
        f"- Cohorts: lowest vs highest **{int(Q*100)}%** by inverse-vol rank",
        f"- Rebalance: every **{REBALANCE}** sessions",
        f"- Hold: **{HOLD}** sessions",
        f"- Costs: CNC round-trip **{primary['cost_pct_points']}** pct points; "
        f"turnover one-way={P0.TURNOVER_ONE_WAY}",
        "- Primary portfolio: long low-vol / short high-vol",
        "- Multiple testing: single primary specification (n_trials=1)",
        "- Methodological deltas vs EXP-NEXT-02: see frozen JSON "
        "`methodological_versioning_vs_EXP_NEXT_02` (universe + partitions only; "
        "not tuned on the prior inconclusive result)",
        "",
        "## 6. Discovery / confirmation partitions",
        "",
        f"- Discovery: `{primary['partitions']['discovery']}` "
        f"({primary['sample']['discovery_sessions']} sessions)",
        f"- Confirmation: `{primary['partitions']['confirm']}` "
        f"({primary['sample']['confirm_sessions']} sessions)",
        "- Method: chronological; confirmation untouched during discovery",
        "- Pre-registered approx rebalances: discovery ~42, confirm ~31 "
        "(both ≥ harness min_n=30) — stated before outcome inspection",
        "",
        "## 7. Sample size",
        "",
        f"- Discovery rebalances: **{disc['n_rebalances']}**",
        f"- Discovery n_eff: **{disc['pack']['n_eff']}**",
        f"- Discovery median cohort size (per leg): **{disc['median_cohort_size']}**",
        f"- Discovery mean names scored: **{disc['mean_names_scored']}**",
        f"- Confirmation rebalances: "
        f"**{conf['n_rebalances'] if conf else 'n/a (not opened / not PASS)'}**",
        "",
        "## 8–11. Gross / costs / net / risk-adjusted",
        "",
        "### Discovery",
        "",
        "```json",
        json.dumps({
            "mean_gross": disc["mean_gross"],
            "cost_drag": disc["cost_drag"],
            "mean_net": disc["mean_net"],
            "sharpe_net": disc["sharpe_net"],
            "sortino_net": disc["sortino_net"],
            "downside_std": disc["downside_std"],
            "max_drawdown": disc["max_drawdown"],
            "hit_rate": disc["hit_rate"],
            "long_only_sharpe_proxy_gross": disc["long_only_sharpe_proxy_gross"],
            "ew_sharpe_proxy_gross": disc["ew_sharpe_proxy_gross"],
            "long_only_minus_ew_mean_gross": disc["long_only_minus_ew_mean_gross"],
            "pack": disc["pack"],
            "verdict": disc["verdict"],
        }, indent=2),
        "```",
        "",
        "### Confirmation",
        "",
        "```json",
        json.dumps(conf, indent=2, default=str) if conf else "null",
        "```",
        "",
        "## 12. Statistical evidence",
        "",
        f"- Discovery harness verdict: `{disc['pack']['verdict']}`",
        f"- Discovery p_value: `{disc['pack']['p_value']}`",
        f"- Discovery DSR/PSR: `{disc['pack']['dsr']}` / `{disc['pack']['psr']}`",
        f"- Discovery CI95 (mean net): `{disc['pack']['ci_95']}`",
        f"- Confirmation harness: "
        f"`{conf['pack']['verdict'] if conf else 'n/a'}`",
        "- Multiple-testing: single frozen specification",
        "",
        "## 13. Economic evidence",
        "",
        f"- Discovery gross mean: **{disc['mean_gross']}**",
        f"- Cost drag / rebalance: **{disc['cost_drag']}**",
        f"- Discovery net mean: **{disc['mean_net']}**",
        f"- Economic gate mean_net>0: "
        f"**{'PASS' if disc['mean_net'] > 0 else 'FAIL'}**",
        "",
        "## 14. Confirmation evidence",
        "",
        f"- Opened: **{conf is not None}**",
        f"- Confirmation verdict: **{conf['verdict'] if conf else 'n/a'}**",
        f"- Final after confirm mapping: **{primary['final_verdict']}**",
        "",
        "## 15. Subperiod stability (discovery calendar years)",
        "",
        "```json",
        json.dumps(primary.get("subperiod_stability_discovery"), indent=2),
        "```",
        "",
        "## 16. Path A secondary robustness",
        "",
    ]
    if path_a:
        lines += [
            "> SECONDARY ROBUSTNESS EVIDENCE — cannot override primary verdict.",
            "",
            f"- Snapshot: `{path_a['snapshot_id']}` (29 names)",
            f"- Partitions: `{path_a['partitions']}`",
            f"- Discovery verdict: `{path_a['discovery']['verdict']}` "
            f"(n={path_a['discovery']['n_rebalances']}, "
            f"net={path_a['discovery']['mean_net']})",
            f"- Confirmation: "
            f"`{path_a['confirmation']['verdict'] if path_a.get('confirmation') else None}`",
            f"- Path A final: **{path_a['final_verdict']}**",
            "",
            "```json",
            json.dumps(path_a, indent=2, default=str),
            "```",
            "",
        ]
    else:
        lines += ["Path A not run.", ""]

    lines += [
        "## 17. Scientific-memory update",
        "",
        f"- Registry status: `{primary['registry_status']}`",
        f"- Next action: `{primary['next_action']}`",
        "- Closed branches (momentum/reversal/structure/network/logistic/"
        "vol-compression) remain closed and were not reopened.",
        "",
        "## 18. Production behaviour confirmation",
        "",
        "| Surface | Status |",
        "|---------|--------|",
        "| Brain / ranking / risk / sizing | Unchanged |",
        "| Execution / broker / autopilot / alerts | Unchanged |",
        "| Production authority | `False` |",
        "",
        "## 19. Plain-English conclusion",
        "",
        plain["what_it_means"],
        "",
        plain["what_quantterm_will_do"],
        "",
        "## 20. Final verdict",
        "",
        f"**{primary['final_verdict']}**",
        "",
        "---",
        "",
        "## Status card",
        "",
        f"| Field | Value |",
        f"|-------|--------|",
        f"| PREVIOUS RESULT | INCONCLUSIVE (EXP-NEXT-02 / 29-name) |",
        f"| EXPANDED DISCOVERY | `{disc['verdict']}` "
        f"(net={disc['mean_net']}, n={disc['n_rebalances']}) |",
        f"| INDEPENDENT CONFIRMATION | "
        f"`{conf['verdict'] if conf else 'not opened'}` |",
        f"| STATISTICAL VERDICT | "
        f"disc `{disc['pack']['verdict']}` / "
        f"conf `{(conf or {}).get('pack', {}).get('verdict')}` |",
        f"| ECONOMIC VERDICT | "
        f"{'PASS' if disc['mean_net'] > 0 else 'FAIL'} "
        f"(gross={disc['mean_gross']}, drag={disc['cost_drag']}, "
        f"net={disc['mean_net']}) |",
        f"| FINAL VERDICT | **{primary['final_verdict']}** |",
        f"| NEXT ACTION | **{primary['next_action']}** |",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        f"_git_sha `{primary['git_sha']}`_",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return OUT_MD


def run() -> dict[str, Any]:
    frozen = load_frozen()
    assert frozen["frozen_before_outcome_inspection"] is True
    assert frozen["snapshot_id"] == SNAPSHOT_ID
    primary = run_path_b()
    path_a = run_path_a_secondary()
    write_report(primary, path_a, frozen)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {"primary": primary, "path_a": path_a, "frozen_protocol_path": str(FROZEN_PATH)}
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


if __name__ == "__main__":
    out = run()
    print(json.dumps({
        "experiment_id": out["primary"]["experiment_id"],
        "hypothesis_id": out["primary"]["hypothesis_id"],
        "final_verdict": out["primary"]["final_verdict"],
        "next_action": out["primary"]["next_action"],
        "discovery": out["primary"]["discovery"]["verdict"],
        "confirmation": (
            None if out["primary"]["confirmation"] is None
            else out["primary"]["confirmation"]["verdict"]
        ),
        "path_a_final": out["path_a"]["final_verdict"],
        "report": str(OUT_MD),
    }, indent=2))
