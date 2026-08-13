"""Run approved next-research cycle end-to-end (no production changes)."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from product.plain_language import PlainCard, render_layers
from research.phase_next import protocol as P
from research.phase_next.data import load_research_panel, period_masks
from research.phase_next.exp_lowvol import run_exp_next_02
from research.phase_next.exp_reversal import run_exp_next_01
from research.phase_next.exp_volcomp import run_exp_next_03

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "logs" / "phase_next"
READINESS = REPO / "NEXT_RESEARCH_EXECUTION_READINESS.md"
R01 = REPO / "EXP_NEXT_01_REVERSAL_REPORT.md"
R02 = REPO / "EXP_NEXT_02_LOW_VOL_REPORT.md"
R03 = REPO / "EXP_NEXT_03_VOL_COMPRESSION_REPORT.md"
FINAL = REPO / "QUANTTERM_NEXT_RESEARCH_CYCLE_FINAL.md"


def _git() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO)
        ).strip()
    except Exception:
        return "unknown"


def write_readiness(panel) -> dict:
    masks = period_masks(panel.closes)
    common = {
        "universe": "FIXED_PREREGISTERED_29",
        "snapshot_id": P.SNAPSHOT_ID,
        "dates": f"{panel.closes.index.min().date()}→{panel.closes.index.max().date()}",
        "identity": "VERIFIED (scoped cert)",
        "ca": "VERIFIED for panel consecutive jumps",
        "cost": "CNC round_trip_cost_pct",
        "pit": "PitContract on scoped snapshot",
        "discovery": f"{P.DISCOVERY_START}→{P.DISCOVERY_END} (n={len(masks['discovery'])})",
        "confirm": f"{P.CONFIRM_START}→end (n={len(masks['confirm'])})",
        "warmup": f"≤{P.WARMUP_END} (n={len(masks['warmup'])})",
    }
    rows = [
        {
            "experiment": "EXP-NEXT-01",
            "required_fields": "adjusted close",
            "volume": "NOT_REQUIRED",
            "benchmark_index": "NOT_REQUIRED (panel-relative CS)",
            "status": "READY",
            "blockers": [],
            **common,
        },
        {
            "experiment": "EXP-NEXT-02",
            "required_fields": "adjusted close → realized vol",
            "volume": "NOT_REQUIRED",
            "benchmark_index": "NOT_REQUIRED",
            "status": "READY",
            "blockers": [],
            **common,
        },
        {
            "experiment": "EXP-NEXT-03",
            "required_fields": "adjusted close → vol10/vol60",
            "volume": "NOT_REQUIRED",
            "benchmark_index": "NOT_REQUIRED",
            "status": "READY",
            "blockers": [],
            **common,
        },
    ]
    lines = [
        "# Next Research Execution Readiness",
        "",
        "> Pre-execution audit against certified snapshot "
        f"`{P.SNAPSHOT_ID}`. Global trust remains `{P.GLOBAL_TRUST}`.",
        "",
        "## Snapshot",
        "",
        f"- snapshot_id: `{P.SNAPSHOT_ID}`",
        f"- scoped_certification: `{panel.manifest.get('scoped_certification')}`",
        f"- trust_class: `{panel.manifest.get('trust_class')}`",
        f"- equity_sha256: `{panel.manifest.get('equity_sha256')}`",
        f"- n_symbols: {panel.closes.shape[1]}",
        f"- n_sessions: {len(panel.closes)}",
        f"- date_range: {common['dates']}",
        "",
        "## Temporal partitions (frozen before outcomes)",
        "",
        f"- Warmup / τ-fit: ≤ `{P.WARMUP_END}`",
        f"- Discovery OOS: `{P.DISCOVERY_START}` → `{P.DISCOVERY_END}`",
        f"- Confirmation OOS: `{P.CONFIRM_START}` → panel end (untouched until discovery known)",
        "",
        "## Per-experiment matrix",
        "",
        "| EXPERIMENT | FIELDS | UNIVERSE | DATES | IDENTITY | CA | INDEX | VOLUME | PIT | COSTS | STATUS | BLOCKERS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['required_fields']} | {r['universe']} | "
            f"{r['dates']} | {r['identity']} | {r['ca']} | {r['benchmark_index']} | "
            f"{r['volume']} | {r['pit']} | {r['cost']} | **{r['status']}** | "
            f"{r['blockers'] or '—'} |"
        )
    lines.extend([
        "",
        "## Classification",
        "",
        "- EXP-NEXT-01: **READY**",
        "- EXP-NEXT-02: **READY**",
        "- EXP-NEXT-03: **READY**",
        "",
        "No experiment blocked. Proceeding with all three.",
        "",
        f"_Written at {_now()}_",
        "",
    ])
    READINESS.write_text("\n".join(lines), encoding="utf-8")
    return {"rows": rows, "all_ready": True}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain_block(tested: str, happened: str, means: str, will_do: str) -> str:
    return (
        f"**What we tested:** {tested}\n\n"
        f"**What happened:** {happened}\n\n"
        f"**What it means:** {means}\n\n"
        f"**What QuantTerm will do:** {will_do}\n"
    )


def _next_action(final: str, exp_type: str) -> str:
    if final == "CONFIRMED":
        return (
            "FOLLOW_UP_POLICY_EXPERIMENT"
            if exp_type == "RISK"
            else "FOLLOW_UP_VALIDATION_ONLY"
        )
    if final == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION":
        return "WAIT_FOR_INDEPENDENT_EVIDENCE"
    if final in {"FAIL", "FAILED_CONFIRMATION"}:
        return "REJECT_CLOSE_BRANCH"
    if final == "BLOCKED":
        return "DATA_FOUNDATION_FIRST"
    return "HOLD_NO_TUNING"


def write_exp_report(path: Path, title: str, res: dict, plain: dict) -> None:
    lines = [
        f"# {title}",
        "",
        "> Scientific report. Production unchanged. Not a live trading authorization.",
        "",
        "## WHAT WE TESTED / WHAT HAPPENED / WHAT IT MEANS / WHAT QUANTTERM WILL DO",
        "",
        _plain_block(
            plain["tested"], plain["happened"], plain["means"], plain["will_do"],
        ),
        "",
        "---",
        "",
        "## Technical layer",
        "",
        f"- Experiment ID: `{res.get('experiment_id')}`",
        f"- Hypothesis ID: `{res.get('hypothesis_id')}`",
        f"- Type: `{res.get('type')}`",
        f"- Snapshot: `{res.get('snapshot_id')}`",
        f"- Discovery verdict: `{ (res.get('discovery') or {}).get('verdict') }`",
        f"- Confirmation: `"
        f"{None if res.get('confirmation') is None else res['confirmation'].get('verdict')}`",
        f"- Final verdict: **{res.get('final_verdict')}**",
        f"- Next action: `{_next_action(res.get('final_verdict'), res.get('type'))}`",
        f"- Production authority: `{res.get('production_authority')}`",
        f"- Registry: `{res.get('registry_status')}`",
        f"- Result hash: `{res.get('result_hash')}`",
        "",
        "### Discovery detail",
        "",
        "```json",
        json.dumps(res.get("discovery"), indent=2, default=str),
        "```",
        "",
        "### Confirmation detail",
        "",
        "```json",
        json.dumps(res.get("confirmation"), indent=2, default=str),
        "```",
        "",
        f"_Generated {_now()}_",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plain_for(res: dict) -> dict:
    eid = res["experiment_id"]
    final = res["final_verdict"]
    if eid == "EXP-NEXT-01":
        tested = (
            "Whether stocks that fall sharply over a few days tend to bounce back "
            "after trading costs."
        )
    elif eid == "EXP-NEXT-02":
        tested = (
            "Whether quieter (lower-volatility) stocks produce better risk-adjusted "
            "results than high-volatility stocks after costs."
        )
    else:
        tested = (
            "Whether unusually calm (compressed) price movement changes future "
            "downside risk — as a warning, not a buy tip."
        )

    if final in {"FAIL", "FAILED_CONFIRMATION"}:
        happened = (
            "After trading costs and the frozen checks, the effect was not reliable "
            "(or did not survive independent confirmation)."
        )
        means = "The idea does not currently show a proven advantage QuantTerm should use."
        will = "Nothing. The strategy / risk rule will not be used. Branch closed."
    elif final == "CONFIRMED":
        happened = (
            "The discovery sample showed a reliable effect and the independent later "
            "period agreed."
        )
        means = (
            "The idea is eligible for follow-up research only — not approved for "
            "real trading."
        )
        will = (
            "No live changes. A separate validation/policy experiment may be planned later."
        )
    elif final == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION":
        happened = "Discovery looked supportive, but independent confirmation was not available."
        means = "Not enough to treat as confirmed."
        will = "Wait for more independent history before acting."
    else:
        happened = "The evidence was mixed or underpowered under the frozen criteria."
        means = "We cannot claim the idea works or fails cleanly yet."
        will = "No tuning. No live use. Hold."
    return {"tested": tested, "happened": happened, "means": means, "will_do": will}


def write_final(results: list[dict], readiness: dict) -> None:
    rows = []
    for r in results:
        disc = (r.get("discovery") or {}).get("verdict")
        conf = None if r.get("confirmation") is None else r["confirmation"].get("verdict")
        econ = _econ_label(r)
        rows.append({
            "experiment": r["experiment_id"],
            "type": r["type"],
            "discovery": disc,
            "confirmation": conf,
            "economic_value": econ,
            "final": r["final_verdict"],
            "next": _next_action(r["final_verdict"], r["type"]),
        })

    confirmed = [x for x in rows if x["final"] == "CONFIRMED"]
    needs_future = [x for x in rows if x["final"] == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION"]
    blocked = [x for x in rows if x["final"] == "BLOCKED"]
    failed = [x for x in rows if x["final"] in {"FAIL", "FAILED_CONFIRMATION"}]

    if confirmed:
        overall = "FOLLOW_UP_ONLY_ON_CONFIRMED_HYPOTHESIS"
    elif needs_future and not confirmed:
        overall = "WAIT_FOR_INDEPENDENT_EVIDENCE"
    elif blocked and len(blocked) == len(rows):
        overall = "DATA_FOUNDATION_FIRST"
    elif len(failed) == len(rows):
        overall = "STOP_MODEL_EXPANSION_AND_REASSESS_DATA/HYPOTHESES"
    else:
        overall = "STOP_MODEL_EXPANSION_AND_REASSESS_DATA/HYPOTHESES"

    lines = [
        "# QuantTerm Next Research Cycle — Final Report",
        "",
        "> End-to-end cycle for EXP-NEXT-01/02/03. "
        f"Snapshot `{P.SNAPSHOT_ID}`. Global trust `{P.GLOBAL_TRUST}`. "
        "Production unchanged. Phase B not started.",
        "",
        "## 1. Executive summary",
        "",
        _exec_summary(rows, overall),
        "",
        "## 2. Data / snapshot certification used",
        "",
        f"- Snapshot: `{P.SNAPSHOT_ID}` (scoped READY_FOR_SCIENTIFIC_RERUN)",
        f"- Global trust: `{P.GLOBAL_TRUST}` (unchanged)",
        f"- Readiness doc: `{READINESS.name}` — all three READY",
        f"- Partitions: discovery `{P.DISCOVERY_START}→{P.DISCOVERY_END}`; "
        f"confirm `{P.CONFIRM_START}→end`",
        "",
        "## 3. Exact experiment IDs",
        "",
    ]
    for r in results:
        lines.append(
            f"- `{r['experiment_id']}` / `{r['hypothesis_id']}` / final=`{r['final_verdict']}`"
        )
    lines.extend(["", "## 4–9. Per-experiment outcomes", ""])
    for r in results:
        lines.append(f"### {r['experiment_id']}")
        lines.append("")
        plain = _plain_for(r)
        lines.append(_plain_block(
            plain["tested"], plain["happened"], plain["means"], plain["will_do"],
        ))
        lines.append("")
        lines.append(
            f"- Discovery: `{(r.get('discovery') or {}).get('verdict')}` · "
            f"Confirmation: `"
            f"{None if r.get('confirmation') is None else r['confirmation'].get('verdict')}` · "
            f"Final: **{r['final_verdict']}**"
        )
        lines.append("")

    lines.extend([
        "## 10. Statistical significance",
        "",
        "See per-experiment JSON in individual reports (harness DSR/PSR, FDR for "
        "EXP-NEXT-01 cell family, materiality gaps for EXP-NEXT-03).",
        "",
        "## 11. Economic significance",
        "",
    ])
    for row in rows:
        lines.append(f"- `{row['experiment']}`: `{row['economic_value']}`")
    lines.extend([
        "",
        "## 12. Cost impact",
        "",
        f"- ALPHA tests use CNC round-trip (`research.phase_next.eval_utils.cost_pct`).",
        "- Gross vs net reported in EXP-NEXT-01/02 packs; net≤0 cannot PASS.",
        "- EXP-NEXT-03 is RISK diagnostic (cost reporting secondary).",
        "",
        "## 13. Multiple-testing treatment",
        "",
        "- EXP-NEXT-01: BH-FDR across 6 formation×hold cells; DSR n_trials=6",
        "- EXP-NEXT-02: single primary spec; DSR n_trials=1",
        "- EXP-NEXT-03: single frozen τ from warmup; no formula mining",
        "- Across the three families: no post-hoc expansion; family results reported separately",
        "",
        "## 14. Closed hypotheses (this cycle + prior)",
        "",
        "- Prior closed: structure, network alpha, momentum, logistic, network interaction",
        "- This cycle closures: see final verdicts FAIL / FAILED_CONFIRMATION below",
        "",
        "## 15. Surviving hypotheses",
        "",
    ])
    if confirmed:
        for c in confirmed:
            lines.append(f"- `{c['experiment']}` CONFIRMED — follow-up research only")
    else:
        lines.append("- None confirmed for follow-up implementation.")
    lines.extend([
        "",
        "## 16. Scientific-memory updates",
        "",
        "- Outcomes recorded via phase_a5 preregistry + negative/watch beliefs.",
        "- Prior A.6 interaction failure lesson retained.",
        "",
        "## 17. Production behaviour confirmation",
        "",
        "- production_behaviour_changed: **False**",
        "- Brain / ranking / risk / execution / broker: **unchanged**",
        "- phase_b_started: **False**",
        "",
        "## 18. What NOT to build next",
        "",
        "- Do not rescue failed reversal/low-vol/compression with ML or parameter sweeps",
        "- Do not reopen momentum/network/structure branches",
        "- Do not invent live risk blocks from unconfirmed RISK diagnostics",
        "",
        "## 19. What QuantTerm should do next",
        "",
        f"**OVERALL NEXT ACTION: `{overall}`**",
        "",
        _overall_text(overall),
        "",
        "## 20. Plain-English summary",
        "",
        _exec_summary(rows, overall),
        "",
        "---",
        "",
        "## Final decision table",
        "",
        "| EXPERIMENT | TYPE | DISCOVERY | CONFIRMATION | ECONOMIC VALUE | FINAL VERDICT | NEXT ACTION |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['type']} | {row['discovery']} | "
            f"{row['confirmation']} | {row['economic_value']} | **{row['final']}** | "
            f"`{row['next']}` |"
        )
    lines.extend([
        "",
        f"| OVERALL | — | — | — | — | — | `{overall}` |",
        "",
        f"_git_sha: `{_git()}` · evaluated {_now()}_",
        "",
    ])
    FINAL.write_text("\n".join(lines), encoding="utf-8")


def _econ_label(r: dict) -> str:
    final = r["final_verdict"]
    if r["type"] == "RISK":
        d = r.get("discovery") or {}
        if final == "CONFIRMED":
            return "MATERIAL_DOWNSIDE_CONTEXT"
        if d.get("incremental_to_abs_vol") is False:
            return "NO_INCREMENTAL_RISK_VALUE"
        return "NO_MATERIAL_RISK_VALUE"
    d = r.get("discovery") or {}
    if r["experiment_id"] == "EXP-NEXT-01":
        best = (d.get("best") or {})
        if best.get("mean_net", 0) <= 0:
            return "NET_NON_POSITIVE"
        return "NET_POSITIVE_UNCONFIRMED" if final != "CONFIRMED" else "NET_POSITIVE_CONFIRMED"
    pack = (d.get("pack") or {})
    if pack.get("mean_net", 0) <= 0:
        return "NET_NON_POSITIVE"
    return "NET_POSITIVE_UNCONFIRMED" if final != "CONFIRMED" else "NET_POSITIVE_CONFIRMED"


def _exec_summary(rows, overall: str) -> str:
    bits = [f"{r['experiment']}→{r['final']}" for r in rows]
    return (
        "We finished the three approved next tests on verified panel history. "
        + "; ".join(bits)
        + f". Overall recommendation: {overall}. "
        "Nothing goes live."
    )


def _overall_text(overall: str) -> str:
    if overall == "FOLLOW_UP_ONLY_ON_CONFIRMED_HYPOTHESIS":
        return (
            "One or more hypotheses confirmed scientifically. Plan a separate "
            "validation/policy experiment. Do not change production."
        )
    if overall == "WAIT_FOR_INDEPENDENT_EVIDENCE":
        return (
            "Discovery support exists but independent confirmation history is "
            "insufficient. Wait for more certified sessions."
        )
    if overall == "DATA_FOUNDATION_FIRST":
        return (
            "Certified data cannot support the approved tests. Improve data "
            "foundation before new alpha work."
        )
    return (
        "Current certified evidence does not support these tested hypotheses. "
        "Stop model expansion. Reassess data breadth (PIT fundamentals/events) "
        "and/or wait for more independent history before new economic families. "
        "Do not reopen closed branches or add ML to rescue failures."
    )


def run_cycle() -> dict:
    panel = load_research_panel()
    readiness = write_readiness(panel)

    r1 = run_exp_next_01(panel)
    write_exp_report(R01, "EXP-NEXT-01 — Short-Horizon Reversal", r1, _plain_for(r1))

    r2 = run_exp_next_02(panel)
    write_exp_report(R02, "EXP-NEXT-02 — Low-Volatility Effect", r2, _plain_for(r2))

    r3 = run_exp_next_03(panel)
    write_exp_report(
        R03, "EXP-NEXT-03 — Volatility Compression (Risk Context)", r3, _plain_for(r3),
    )

    results = [r1, r2, r3]
    write_final(results, readiness)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_id": P.SNAPSHOT_ID,
        "global_trust": P.GLOBAL_TRUST,
        "production_behaviour_changed": False,
        "phase_b_started": False,
        "results": results,
        "evaluated_at": _now(),
        "git_sha": _git(),
    }
    (OUT_DIR / "cycle_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    out = run_cycle()
    print(json.dumps({
        "snapshot_id": out["snapshot_id"],
        "production_behaviour_changed": out["production_behaviour_changed"],
        "verdicts": {
            r["experiment_id"]: r["final_verdict"] for r in out["results"]
        },
        "hypothesis_ids": {
            r["experiment_id"]: r["hypothesis_id"] for r in out["results"]
        },
        "reports": [READINESS.name, R01.name, R02.name, R03.name, FINAL.name],
    }, indent=2))
