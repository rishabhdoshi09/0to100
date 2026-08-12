"""EXP-FUND-01..04 runners + cycle orchestrator."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research.fund_cycle import data as D
from research.phase_a5 import metrics as M
from research.phase_a5 import prereg
from research.phase_a5.metrics import fdr_on_pvalues
from research.phase_next import eval_utils as E

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "logs" / "research_expansion" / "fund_cycle"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def _write_report(path: Path, *, experiment_id: str, title: str, question: str,
                  result: dict, extra_blocks: list[str] | None = None) -> None:
    plain = D.plain_sections(question, result["final_verdict"])
    disc = result["discovery"]
    conf = result.get("confirmation")
    lines = [
        f"# {experiment_id} — {title}",
        "",
        "> Scientific report. Production unchanged. Not a live trading authorization.",
        "> Global trust `OPERATIONAL_ONLY`. No ML. Closed OHLCV branches not reopened.",
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
        "## Technical evidence",
        "",
        f"- Experiment ID: `{experiment_id}`",
        f"- Hypothesis ID: `{result['hypothesis_id']}`",
        f"- Type: `{result['type']}`",
        f"- Foundation package: `{D.FOUNDATION_ID}`",
        f"- Parent OHLCV snapshot: `{D.OHLCV_ID}`",
        f"- Partitions: discovery `{D.DISCOVERY_START}→{D.DISCOVERY_END}`; "
        f"confirm `{D.CONFIRM_START}→{D.CONFIRM_END}`",
        f"- Cost: CNC {D.cost_pct()} pct pts; drag/rebalance={D.cost_drag()}",
        f"- Discovery verdict: `{disc['verdict']}`",
        f"- Confirmation: `{None if conf is None else conf['verdict']}`",
        f"- Final verdict: **{result['final_verdict']}**",
        f"- Next action: `{result['next_action']}`",
        f"- Production authority: `False`",
        f"- Registry: `{result.get('registry_status')}`",
        f"- Result hash: `{result['result_hash']}`",
        f"- Multiple-testing: harness n_trials={D.N_TRIALS}",
        "",
        "### Discovery detail",
        "",
        "```json",
        json.dumps({k: v for k, v in disc.items() if k not in {"gross", "net"}}, indent=2, default=str),
        "```",
        "",
        "### Confirmation detail",
        "",
        "```json",
        json.dumps(
            None if conf is None else {k: v for k, v in conf.items() if k not in {"gross", "net"}},
            indent=2, default=str,
        ),
        "```",
        "",
    ]
    if extra_blocks:
        lines.extend(extra_blocks)
    lines += [
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        f"_git_sha `{result.get('git_sha')}`_",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _eval_cs_partition(
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    invert: bool,
) -> dict:
    fwd = M.forward_returns(closes, D.HOLD)
    reb = D.rebalance_dates(dates, D.REB_CS)
    gross, ew = D.long_short_from_scores(scores, fwd, reb, invert=invert)
    ev = D.pack_eval(gross)
    ev["n_rebalances"] = int(len(gross))
    ev["median_names"] = int(
        scores.loc[reb].notna().sum(axis=1).median()
    ) if len(reb) else 0
    ev["ew_mean_gross"] = round(float(ew.mean()) if len(ew) else 0.0, 6)
    ev["ls_minus_ew_mean_gross"] = round(
        float((gross - ew).mean()) if len(gross) and len(ew) else 0.0, 6
    )
    return ev


def run_exp_fund_01(panel: D.FundPanel) -> dict:
    spec = panel.frozen["experiments"]["EXP-FUND-01"]
    yoy = D._yoy_eps_map(panel.fundamentals)
    days = D.trading_days(panel.closes)
    fwd = M.forward_returns(panel.closes, D.HOLD)

    # Restrict to EARNINGS_RESULT with matching yoy row on same available_at (+/-0d exact match preferred)
    events = panel.events[panel.events["event_type"] == "EARNINGS_RESULT"].copy()
    events["available_at"] = pd.to_datetime(events["available_at"])
    events["symbol"] = events["symbol"].astype(str).str.upper()

    # Merge events to yoy on symbol + available_at date
    yoy2 = yoy.copy()
    yoy2["avail_day"] = yoy2["available_at"].dt.normalize()
    events["avail_day"] = events["available_at"].dt.normalize()
    merged = events.merge(
        yoy2[["symbol", "avail_day", "yoy_eps_growth", "basic_eps"]],
        on=["symbol", "avail_day"],
        how="inner",
    )

    def _event_returns(sub: pd.DataFrame) -> dict:
        rows = []
        for _, r in sub.iterrows():
            entry = D.next_session(days, str(r["avail_day"].date()))
            if entry is None or entry not in fwd.index:
                continue
            if r["symbol"] not in fwd.columns:
                continue
            ret = fwd.at[entry, r["symbol"]]
            if pd.isna(ret):
                continue
            rows.append({
                "entry": entry,
                "symbol": r["symbol"],
                "surprise": float(r["yoy_eps_growth"]),
                "ret": float(ret),
            })
        if not rows:
            return D.pack_eval(pd.Series(dtype=float)) | {
                "n_events": 0, "n_rebalances": 0, "median_names": 0,
                "ew_mean_gross": 0.0, "ls_minus_ew_mean_gross": 0.0,
            }
        df = pd.DataFrame(rows)
        # Cross-sectional: within each calendar month of entry, long top/short bottom surprise
        df["month"] = df["entry"].dt.to_period("M")
        port = []
        ew = []
        idx = []
        for month, g in df.groupby("month"):
            if len(g) < 10:
                continue
            n = max(1, int(len(g) * D.Q))
            g = g.sort_values("surprise")
            short = g.head(n)["ret"].mean()
            long = g.tail(n)["ret"].mean()
            port.append(float(long - short))
            ew.append(float(g["ret"].mean()))
            idx.append(pd.Timestamp(str(month)))
        gross = pd.Series(port, index=pd.Index(idx), dtype=float)
        ev = D.pack_eval(gross)
        ev["n_events"] = int(len(df))
        ev["n_rebalances"] = int(len(gross))
        ev["median_names"] = int(df.groupby("month").size().median()) if len(df) else 0
        ev["ew_mean_gross"] = round(float(pd.Series(ew).mean()) if ew else 0.0, 6)
        return ev

    disc_events = merged[
        (merged["avail_day"] >= pd.Timestamp(D.DISCOVERY_START))
        & (merged["avail_day"] <= pd.Timestamp(D.DISCOVERY_END))
    ]
    conf_events = merged[
        (merged["avail_day"] >= pd.Timestamp(D.CONFIRM_START))
        & (merged["avail_day"] <= pd.Timestamp(D.CONFIRM_END))
    ]

    hid = prereg.preregister(
        experiment_id="EXP-FUND-01",
        hypothesis=spec["hypothesis"],
        null_hypothesis=spec["null_hypothesis"],
        success_criteria=spec["success_criteria"],
        data_window={
            "foundation_id": D.FOUNDATION_ID,
            "ohlcv_snapshot_id": D.OHLCV_ID,
            "discovery": f"{D.DISCOVERY_START}→{D.DISCOVERY_END}",
            "confirm": f"{D.CONFIRM_START}→{D.CONFIRM_END}",
            "hold": D.HOLD,
        },
        protocol={
            "type": "ALPHA_EVENT",
            "entry": "next_session_after_available_at",
            "signal": "yoy_basic_eps_growth",
            "costs": "CNC",
            "n_trials": D.N_TRIALS,
            "production_authority": False,
        },
        seed=42,
        code_hash="exp_fund_01_v1",
    )

    discovery = _event_returns(disc_events)
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = _event_returns(conf_events)
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
    D.remember(final, "EXP-FUND-01", hid, discovery, "post_earnings_drift")
    result = {
        "experiment_id": "EXP-FUND-01",
        "type": "ALPHA_EVENT",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "discovery": discovery,
        "confirmation": confirmation,
        "final_verdict": final,
        "next_action": D.next_action_for(final),
        "git_sha": _git_sha(),
        "result_hash": E.result_hash({"d": discovery["verdict"], "f": final, "n": discovery["pack"]["mean_net"]}),
        "matched_events_discovery": int(len(disc_events)),
        "matched_events_confirm": int(len(conf_events)),
    }
    _write_report(
        REPO_ROOT / "EXP_FUND_01_POST_EARNINGS_DRIFT.md",
        experiment_id="EXP-FUND-01",
        title="Post-Earnings Drift",
        question=spec["hypothesis_plain"],
        result=result,
        extra_blocks=[
            "## Protocol notes",
            "",
            "- Signal: YoY basic EPS growth at earnings AVAILABLE_AT",
            "- Entry: next session after AVAILABLE_AT (conservative)",
            "- Hold: 21 sessions",
            "- Only EARNINGS_RESULT (no generic announcement mining)",
            f"- Matched events discovery/confirm: "
            f"{result['matched_events_discovery']} / {result['matched_events_confirm']}",
            "",
        ],
    )
    return result


def run_exp_fund_02(panel: D.FundPanel) -> dict:
    spec = panel.frozen["experiments"]["EXP-FUND-02"]
    fund = panel.fundamentals.copy()
    fund = fund.dropna(subset=["profit_after_tax", "revenue_from_operations"])
    fund = fund[fund["revenue_from_operations"].astype(float) > 0].copy()
    fund["net_margin"] = fund["profit_after_tax"].astype(float) / fund["revenue_from_operations"].astype(float)
    fund["available_at"] = pd.to_datetime(fund["available_at"])
    fund["symbol"] = fund["symbol"].astype(str).str.upper()
    # Prefer consolidated on same available_at
    fund["_c"] = fund["consolidated"].astype(str).str.lower().str.startswith("consolid").astype(int)
    fund = fund.sort_values(["symbol", "available_at", "_c"]).drop_duplicates(
        ["symbol", "available_at"], keep="last"
    )

    days = D.trading_days(panel.closes)
    disc_days = D.period_mask(days, D.DISCOVERY_START, D.DISCOVERY_END)
    conf_days = D.period_mask(days, D.CONFIRM_START, D.CONFIRM_END)
    scores = D.latest_asof_frame(
        fund, value_col="net_margin", asof_col="available_at",
        dates=days, symbols=list(panel.closes.columns),
    )

    hid = prereg.preregister(
        experiment_id="EXP-FUND-02",
        hypothesis=spec["hypothesis"],
        null_hypothesis=spec["null_hypothesis"],
        success_criteria=spec["success_criteria"],
        data_window={
            "foundation_id": D.FOUNDATION_ID,
            "ohlcv_snapshot_id": D.OHLCV_ID,
            "discovery": f"{D.DISCOVERY_START}→{D.DISCOVERY_END}",
            "confirm": f"{D.CONFIRM_START}→{D.CONFIRM_END}",
            "metric": "net_margin_PAT_over_revenue",
            "reb": D.REB_CS,
            "hold": D.HOLD,
        },
        protocol={
            "type": "ALPHA_FUNDAMENTAL",
            "portfolio": "long high margin / short low margin",
            "costs": "CNC",
            "n_trials": D.N_TRIALS,
            "no_sector_neutral": True,
            "production_authority": False,
        },
        seed=42,
        code_hash="exp_fund_02_v1",
    )
    discovery = _eval_cs_partition(scores, panel.closes, disc_days, invert=False)
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = _eval_cs_partition(scores, panel.closes, conf_days, invert=False)
    final = E.final_after_confirm(
        discovery["verdict"], None if confirmation is None else confirmation["verdict"]
    )
    reg = prereg.record(hid, {
        "discovery_pass": 1 if discovery["verdict"] == "PASS" else 0,
        "confirm_pass": 1 if confirmation and confirmation["verdict"] == "PASS" else 0,
        "mean_net": discovery["pack"]["mean_net"],
        "live_behaviour_changed": 0,
    })
    D.remember(final, "EXP-FUND-02", hid, discovery, "quality_profitability")
    result = {
        "experiment_id": "EXP-FUND-02",
        "type": "ALPHA_FUNDAMENTAL",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "discovery": discovery,
        "confirmation": confirmation,
        "final_verdict": final,
        "next_action": D.next_action_for(final),
        "git_sha": _git_sha(),
        "result_hash": E.result_hash({"d": discovery["verdict"], "f": final}),
    }
    _write_report(
        REPO_ROOT / "EXP_FUND_02_QUALITY_PROFITABILITY.md",
        experiment_id="EXP-FUND-02",
        title="Quality / Profitability",
        question=spec["hypothesis_plain"],
        result=result,
        extra_blocks=[
            "## Protocol notes",
            "",
            "- Quality metric: net margin = PAT / Revenue (PIT AVAILABLE_AT)",
            f"- Rebalance every {D.REB_CS} sessions; hold {D.HOLD}",
            "- No sector neutralization (PIT sectors unavailable)",
            "",
        ],
    )
    return result


def run_exp_fund_03(panel: D.FundPanel) -> dict:
    spec = panel.frozen["experiments"]["EXP-FUND-03"]
    yoy = D._yoy_eps_map(panel.fundamentals)
    yoy["symbol"] = yoy["symbol"].astype(str).str.upper()
    days = D.trading_days(panel.closes)
    disc_days = D.period_mask(days, D.DISCOVERY_START, D.DISCOVERY_END)
    conf_days = D.period_mask(days, D.CONFIRM_START, D.CONFIRM_END)
    scores = D.latest_asof_frame(
        yoy, value_col="yoy_eps_growth", asof_col="available_at",
        dates=days, symbols=list(panel.closes.columns),
    )
    hid = prereg.preregister(
        experiment_id="EXP-FUND-03",
        hypothesis=spec["hypothesis"],
        null_hypothesis=spec["null_hypothesis"],
        success_criteria=spec["success_criteria"],
        data_window={
            "foundation_id": D.FOUNDATION_ID,
            "ohlcv_snapshot_id": D.OHLCV_ID,
            "discovery": f"{D.DISCOVERY_START}→{D.DISCOVERY_END}",
            "confirm": f"{D.CONFIRM_START}→{D.CONFIRM_END}",
            "metric": "yoy_basic_eps_growth",
            "reb": D.REB_CS,
            "hold": D.HOLD,
        },
        protocol={
            "type": "ALPHA_FUNDAMENTAL",
            "portfolio": "long high YoY EPS growth / short low",
            "costs": "CNC",
            "n_trials": D.N_TRIALS,
            "production_authority": False,
        },
        seed=42,
        code_hash="exp_fund_03_v1",
    )
    discovery = _eval_cs_partition(scores, panel.closes, disc_days, invert=False)
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = _eval_cs_partition(scores, panel.closes, conf_days, invert=False)
    final = E.final_after_confirm(
        discovery["verdict"], None if confirmation is None else confirmation["verdict"]
    )
    reg = prereg.record(hid, {
        "discovery_pass": 1 if discovery["verdict"] == "PASS" else 0,
        "confirm_pass": 1 if confirmation and confirmation["verdict"] == "PASS" else 0,
        "mean_net": discovery["pack"]["mean_net"],
        "live_behaviour_changed": 0,
    })
    D.remember(final, "EXP-FUND-03", hid, discovery, "earnings_growth")
    result = {
        "experiment_id": "EXP-FUND-03",
        "type": "ALPHA_FUNDAMENTAL",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "discovery": discovery,
        "confirmation": confirmation,
        "final_verdict": final,
        "next_action": D.next_action_for(final),
        "git_sha": _git_sha(),
        "result_hash": E.result_hash({"d": discovery["verdict"], "f": final}),
    }
    _write_report(
        REPO_ROOT / "EXP_FUND_03_EARNINGS_GROWTH.md",
        experiment_id="EXP-FUND-03",
        title="Earnings Growth",
        question=spec["hypothesis_plain"],
        result=result,
        extra_blocks=[
            "## Protocol notes",
            "",
            "- Growth metric: YoY basic EPS from PIT fundamentals",
            f"- Rebalance every {D.REB_CS} sessions; hold {D.HOLD}",
            "",
        ],
    )
    return result


def run_exp_fund_04(panel: D.FundPanel) -> dict:
    spec = panel.frozen["experiments"]["EXP-FUND-04"]
    val = panel.valuations.copy()
    val["symbol"] = val["symbol"].astype(str).str.upper()
    val["available_ts"] = pd.to_datetime(val["available_ts"])
    val = val[val["pe"].astype(float) > 0]
    val = val[val["pe"].astype(float) <= D.PE_CAP]
    days = D.trading_days(panel.closes)
    disc_days = D.period_mask(days, D.DISCOVERY_START, D.DISCOVERY_END)
    conf_days = D.period_mask(days, D.CONFIRM_START, D.CONFIRM_END)
    # Score = -PE so high score = cheap (low PE); long_short uses high scores as long
    val["neg_pe"] = -val["pe"].astype(float)
    scores = D.latest_asof_frame(
        val, value_col="neg_pe", asof_col="available_ts",
        dates=days, symbols=list(panel.closes.columns),
    )
    hid = prereg.preregister(
        experiment_id="EXP-FUND-04",
        hypothesis=spec["hypothesis"],
        null_hypothesis=spec["null_hypothesis"],
        success_criteria=spec["success_criteria"],
        data_window={
            "foundation_id": D.FOUNDATION_ID,
            "ohlcv_snapshot_id": D.OHLCV_ID,
            "discovery": f"{D.DISCOVERY_START}→{D.DISCOVERY_END}",
            "confirm": f"{D.CONFIRM_START}→{D.CONFIRM_END}",
            "metric": "trailing_pe",
            "pe_cap": D.PE_CAP,
            "reb": D.REB_CS,
            "hold": D.HOLD,
        },
        protocol={
            "type": "ALPHA_FUNDAMENTAL",
            "portfolio": "long low PE / short high PE",
            "costs": "CNC",
            "n_trials": D.N_TRIALS,
            "no_sector_neutral": True,
            "production_authority": False,
        },
        seed=42,
        code_hash="exp_fund_04_v1",
    )
    discovery = _eval_cs_partition(scores, panel.closes, disc_days, invert=False)
    confirmation = None
    if discovery["verdict"] == "PASS":
        confirmation = _eval_cs_partition(scores, panel.closes, conf_days, invert=False)
    final = E.final_after_confirm(
        discovery["verdict"], None if confirmation is None else confirmation["verdict"]
    )
    reg = prereg.record(hid, {
        "discovery_pass": 1 if discovery["verdict"] == "PASS" else 0,
        "confirm_pass": 1 if confirmation and confirmation["verdict"] == "PASS" else 0,
        "mean_net": discovery["pack"]["mean_net"],
        "live_behaviour_changed": 0,
    })
    D.remember(final, "EXP-FUND-04", hid, discovery, "value_trailing_pe")
    result = {
        "experiment_id": "EXP-FUND-04",
        "type": "ALPHA_FUNDAMENTAL",
        "hypothesis_id": hid,
        "registry_status": reg.get("status"),
        "discovery": discovery,
        "confirmation": confirmation,
        "final_verdict": final,
        "next_action": D.next_action_for(final),
        "git_sha": _git_sha(),
        "result_hash": E.result_hash({"d": discovery["verdict"], "f": final}),
    }
    _write_report(
        REPO_ROOT / "EXP_FUND_04_VALUE_PE.md",
        experiment_id="EXP-FUND-04",
        title="Value / Trailing PE",
        question=spec["hypothesis_plain"],
        result=result,
        extra_blocks=[
            "## Protocol notes",
            "",
            "- Valuation: trailing PE with available_ts <= formation",
            f"- Outlier rule: exclude PE > {D.PE_CAP}; PE<=0 excluded",
            "- No sector-neutral value design",
            "",
        ],
    )
    return result


def write_final(results: list[dict], panel: D.FundPanel) -> Path:
    # FDR across discovery p-values
    pmap = {
        r["experiment_id"]: float(r["discovery"]["pack"]["p_value"])
        for r in results
    }
    fdr = fdr_on_pvalues(pmap, alpha=0.05)
    confirmed = [r for r in results if r["final_verdict"] == "CONFIRMED"]
    failed = [r for r in results if r["final_verdict"] in {"FAIL", "FAILED_CONFIRMATION"}]
    incon = [r for r in results if r["final_verdict"] == "INCONCLUSIVE"]
    disc_only = [r for r in results if r["final_verdict"] == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION"]

    if confirmed:
        overall = "FOLLOWUP_RESEARCH_FOR_CONFIRMED_ONLY"
        overall_note = (
            "One or more hypotheses confirmed on this certified scope. "
            "Follow-up research only — no production authorization."
        )
    elif disc_only and not failed:
        overall = "WAIT_FOR_INDEPENDENT_EVIDENCE"
        overall_note = "Discovery passed without sufficient independent confirmation sample."
    elif all(r["final_verdict"] in {"FAIL", "FAILED_CONFIRMATION"} for r in results):
        overall = "STOP_ML_RESCUE_REASSESS_DATA_OR_PAUSE"
        overall_note = (
            "All four fundamentals/events hypotheses failed under frozen rules. "
            "Do not add ML. Next bottleneck: longer independent history and/or new "
            "PIT-safe data families (e.g. shareholding), or temporarily pause alpha research."
        )
    else:
        overall = "MIXED_HOLD_NO_TUNING"
        overall_note = (
            "Mixed FAIL/INCONCLUSIVE — do not escalate complexity; preserve evidence."
        )

    lines = [
        "# QuantTerm Fundamentals + Events Research — Final",
        "",
        "> End-to-end cycle for EXP-FUND-01..04. No Phase B. No AI/ML. Production unchanged.",
        "> Global trust `OPERATIONAL_ONLY`. Closed OHLCV branches not reopened.",
        "",
        "## Plain English",
        "",
        "QuantTerm tested four company-information ideas using only facts that were "
        "public at each historical date: post-earnings drift, quality, earnings growth, "
        "and cheap-vs-expensive (trailing PE). "
        + overall_note,
        "",
        "## Data scope",
        "",
        f"- Foundation package: `{D.FOUNDATION_ID}`",
        f"- Parent OHLCV: `{D.OHLCV_ID}`",
        f"- Events / fundamentals / valuations: package ledgers (AVAILABLE_AT enforced via PitContract)",
        f"- Partitions: discovery `{D.DISCOVERY_START}→{D.DISCOVERY_END}`; "
        f"confirm `{D.CONFIRM_START}→{D.CONFIRM_END}`",
        f"- Costs: CNC {D.cost_pct()} pct points; turnover one-way={D.TURNOVER}",
        f"- Multiple-testing: per-test DSR n_trials={D.N_TRIALS}; cycle BH-FDR α=0.05",
        f"- FDR rejected (discovery): `{fdr.get('rejected')}`",
        "",
        "## Results table",
        "",
        "| EXPERIMENT | TYPE | DISCOVERY | CONFIRMATION | NET ECONOMIC VALUE | FINAL VERDICT | NEXT ACTION |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        conf = r.get("confirmation")
        lines.append(
            f"| {r['experiment_id']} | {r['type']} | {r['discovery']['verdict']} | "
            f"{conf['verdict'] if conf else '—'} | "
            f"{r['discovery']['pack']['mean_net']} | **{r['final_verdict']}** | "
            f"{r['next_action']} |"
        )
    lines += [
        "",
        "## Positive evidence",
        "",
    ]
    if confirmed or disc_only:
        for r in confirmed + disc_only:
            lines.append(
                f"- `{r['experiment_id']}` → `{r['final_verdict']}` "
                f"(disc net={r['discovery']['pack']['mean_net']})"
            )
    else:
        lines.append("- None.")
    lines += ["", "## Negative evidence", ""]
    if failed:
        for r in failed:
            lines.append(
                f"- `{r['experiment_id']}` → `{r['final_verdict']}` "
                f"(disc gross={r['discovery']['pack'].get('mean_gross')}, "
                f"net={r['discovery']['pack']['mean_net']})"
            )
    else:
        lines.append("- None.")
    lines += ["", "## Inconclusive evidence", ""]
    if incon:
        for r in incon:
            lines.append(
                f"- `{r['experiment_id']}` → INCONCLUSIVE "
                f"(n={r['discovery']['pack']['n']}, net={r['discovery']['pack']['mean_net']})"
            )
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Scientific-memory updates",
        "",
        "- Each experiment recorded REJECT/WATCH beliefs with hypothesis ids.",
        "- Closed OHLCV branches remain closed.",
        "",
        "## Production unchanged",
        "",
        "| Surface | Status |",
        "|---------|--------|",
        "| Brain / ranking / risk / sizing / execution | Unchanged |",
        "| Autopilot / broker / alerts | Unchanged |",
        "| Any CONFIRMED result | `ELIGIBLE_FOR_FOLLOWUP_RESEARCH` only |",
        "",
        "## What NOT to build next",
        "",
        "- ML/AI to rescue failed fundamentals hypotheses",
        "- Generic mining across all announcement types",
        "- Sector-neutral redesigns without PIT sector history",
        "- Shareholding factors without AVAILABLE_AT ownership ledger",
        "- Reopening momentum / reversal / low-vol / network branches",
        "",
        "## Overall decision",
        "",
        f"**{overall}**",
        "",
        overall_note,
        "",
        "## Status card",
        "",
        f"| Field | Value |",
        f"|-------|--------|",
        f"| FOUNDATION | `{D.FOUNDATION_ID}` |",
        f"| OHLCV | `{D.OHLCV_ID}` |",
        f"| OVERALL | **{overall}** |",
        f"| CONFIRMED | {len(confirmed)} |",
        f"| FAIL | {len(failed)} |",
        f"| INCONCLUSIVE | {len(incon)} |",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        f"_git_sha `{_git_sha()}`_",
        "",
    ]
    path = REPO_ROOT / "QUANTTERM_FUNDAMENTALS_EVENTS_RESEARCH_FINAL.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_cycle() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("load panel…", flush=True)
    panel = D.load_panel()
    print("EXP-FUND-01…", flush=True)
    r1 = run_exp_fund_01(panel)
    print("  ", r1["final_verdict"], r1["discovery"]["verdict"], r1["discovery"]["pack"]["mean_net"], flush=True)
    print("EXP-FUND-02…", flush=True)
    r2 = run_exp_fund_02(panel)
    print("  ", r2["final_verdict"], r2["discovery"]["verdict"], r2["discovery"]["pack"]["mean_net"], flush=True)
    print("EXP-FUND-03…", flush=True)
    r3 = run_exp_fund_03(panel)
    print("  ", r3["final_verdict"], r3["discovery"]["verdict"], r3["discovery"]["pack"]["mean_net"], flush=True)
    print("EXP-FUND-04…", flush=True)
    r4 = run_exp_fund_04(panel)
    print("  ", r4["final_verdict"], r4["discovery"]["verdict"], r4["discovery"]["pack"]["mean_net"], flush=True)
    results = [r1, r2, r3, r4]
    # strip series for JSON
    slim = []
    for r in results:
        s = dict(r)
        for key in ("discovery", "confirmation"):
            if s.get(key):
                s[key] = {k: v for k, v in s[key].items() if k not in {"gross", "net"}}
        slim.append(s)
    final_path = write_final(results, panel)
    payload = {"results": slim, "final_report": str(final_path)}
    (OUT_DIR / "cycle_result.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    out = run_cycle()
    print(json.dumps({
        "final_report": out["final_report"],
        "verdicts": [
            {"id": r["experiment_id"], "final": r["final_verdict"],
             "disc": r["discovery"]["verdict"], "net": r["discovery"]["pack"]["mean_net"]}
            for r in out["results"]
        ],
    }, indent=2))
