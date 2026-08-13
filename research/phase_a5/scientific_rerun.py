"""Frozen Phase A.5 scientific rerun against scoped certified snapshot.

Uses ONLY snapshot ``a7a9828ec37e09e4``. Does not alter production behaviour.
Does not begin Phase B. Global trust remains OPERATIONAL_ONLY.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.costs import round_trip_cost_pct
from product.plain_language import PlainCard, render_layers
from research.phase_a5 import metrics as M
from research.phase_a5.dataset import (
    CERTIFIED_SNAPSHOT_ID,
    SCOPED_SNAPSHOT_ROOT,
    load_certified_snapshot,
    load_sectors,
)
from research.phase_a5.exp_challenger import run_exp_a3_01
from research.phase_a5.exp_horizons import run_exp_a2_01
from research.phase_a5.exp_interaction import run_exp_a5a6_01
from research.phase_a5.exp_network import run_exp_a6_01
from research.phase_a5.exp_structure import run_exp_a5_01

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "PHASE_A5_FROZEN_PROTOCOLS@2026-08-11"
OOS_START_DATE = "2025-09-19"  # frozen registered oos_start

FROZEN_IDS = {
    "EXP-A5-01": "81b8889792f53113",
    "EXP-A6-01": "590571a11ee06fc2",
    "EXP-A2-01": "775b4a0fce7d5b83",
    "EXP-A3-01": "7842a46ee335685a",
    "EXP-A5A6-01": "3734b8a0a9124a60",
}

OUT_JSON = REPO_ROOT / "logs" / "phase_a5" / "scientific_rerun_results.json"
OUT_MD = REPO_ROOT / "PHASE_A5_RESEARCH_GRADE_RERUN.md"
DISPLAY_RESULTS = REPO_ROOT / "logs" / "phase_a5" / "results.json"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _next_action(scientific: str) -> str:
    if scientific == "PASS":
        return "ADVANCE_TO_CONFIRMATION"
    if scientific == "FAIL":
        return "REJECT"
    return "HOLD"


def _plain_for(scientific: str, name: str) -> dict:
    if scientific == "FAIL":
        card = PlainCard(
            label=name,
            state="FAILED",
            explanation=(
                "We tested this idea again using verified historical NSE data. "
                "It still did not show a reliable advantage, so QuantTerm should not use it."
            ),
            implication="Do not promote into Brain, ranking, risk, or execution.",
            technical=f"scientific_verdict={scientific}",
        )
    elif scientific == "PASS":
        card = PlainCard(
            label=name,
            state="PROMISING",
            explanation=(
                "This idea showed a reliable advantage in the frozen historical test. "
                "It is worth further validation, but it is not approved for real trading yet."
            ),
            implication="Eligible for confirmation review only — production unchanged.",
            technical=f"scientific_verdict={scientific}",
        )
    else:
        card = PlainCard(
            label=name,
            state="NOT_ENOUGH_DATA",
            explanation=(
                "The data are now trustworthy, but the test still does not give enough "
                "evidence to say whether this idea works."
            ),
            implication="Hold — do not promote; do not invent new models to rescue it.",
            technical=f"scientific_verdict={scientific}",
        )
    return render_layers(card)


def load_display_only() -> dict:
    if not DISPLAY_RESULTS.exists():
        return {}
    return json.loads(DISPLAY_RESULTS.read_text())


def run_scientific_rerun(
    *,
    snapshot_id: str = CERTIFIED_SNAPSHOT_ID,
) -> dict[str, Any]:
    if snapshot_id != CERTIFIED_SNAPSHOT_ID:
        raise ValueError(
            f"refusing substitute snapshot {snapshot_id}; "
            f"required {CERTIFIED_SNAPSHOT_ID}"
        )

    sid, pit, manifest, closes = load_certified_snapshot(snapshot_id)
    sectors = load_sectors()
    # Align panel to frozen sector map / 29 names
    missing_sec = [s for s in closes.columns if s not in sectors]
    if missing_sec:
        raise ValueError(f"sector map missing symbols: {missing_sec}")
    # Keep column order matching frozen panel where possible
    from research.phase_a5.scoped_certification import FROZEN_PANEL
    ordered = [s for s in FROZEN_PANEL if s in closes.columns]
    if len(ordered) != 29:
        raise ValueError(f"certified panel size {len(ordered)} != 29")
    closes = closes[ordered]

    display = load_display_only()
    display_exps = (display.get("experiments") or {})

    common_kw = dict(
        oos_start_date=OOS_START_DATE,
    )

    experiments = {}
    experiments["EXP-A5-01"] = run_exp_a5_01(
        closes=closes, sectors=sectors, manifest=manifest,
        frozen_hypothesis_id=FROZEN_IDS["EXP-A5-01"], **common_kw,
    )
    experiments["EXP-A6-01"] = run_exp_a6_01(
        closes=closes, sectors=sectors, manifest=manifest,
        frozen_hypothesis_id=FROZEN_IDS["EXP-A6-01"], **common_kw,
    )
    experiments["EXP-A2-01"] = run_exp_a2_01(
        closes=closes, manifest=manifest,
        frozen_hypothesis_id=FROZEN_IDS["EXP-A2-01"], **common_kw,
    )
    experiments["EXP-A3-01"] = run_exp_a3_01(
        closes=closes, manifest=manifest,
        frozen_hypothesis_id=FROZEN_IDS["EXP-A3-01"],
    )
    experiments["EXP-A5A6-01"] = run_exp_a5a6_01(
        closes=closes, sectors=sectors, manifest=manifest,
        frozen_hypothesis_id=FROZEN_IDS["EXP-A5A6-01"], **common_kw,
    )

    # Attach plain language + comparison + next action
    summary_rows = []
    for eid, hid in FROZEN_IDS.items():
        res = experiments[eid]
        sci = res.get("scientific_verdict") or M.scientific_verdict(res.get("verdict"))
        res["scientific_verdict"] = sci
        res["plain"] = _plain_for(sci, eid)
        res["next_action"] = _next_action(sci)
        res["result_hash"] = _result_hash({
            "verdict": res.get("verdict"),
            "metrics": res.get("metrics"),
            "fdr": res.get("fdr"),
        })
        prev = display_exps.get(eid) or {}
        res["display_only_comparison"] = {
            "previous_verdict": prev.get("verdict"),
            "previous_reason": prev.get("reason"),
            "previous_metrics": prev.get("metrics"),
            "certified_verdict": res.get("verdict"),
            "certified_scientific_verdict": sci,
            "certified_reason": res.get("reason"),
            "direction_changed": (prev.get("verdict") != res.get("verdict")),
            "note": (
                "Disagreement with DISPLAY_ONLY is expected and not an error; "
                "this rerun asks whether exploratory evidence survives certified data."
            ),
        }
        assert res.get("hypothesis_id") == hid, (eid, res.get("hypothesis_id"), hid)
        assert res.get("production_authority") is False
        summary_rows.append({
            "capability": eid,
            "hypothesis_id": hid,
            "display_only": prev.get("verdict"),
            "certified_raw": res.get("verdict"),
            "statistical_verdict": sci,
            "economic_verdict": _economic_label(eid, res),
            "final_scientific_verdict": sci,
            "next_action": res["next_action"],
        })

    # PIT mid-panel smoke already done in loader; record again
    mid = closes.index[len(closes) // 2].strftime("%Y-%m-%d")
    sample = list(closes.columns)[0]
    read = pit.as_of("bars", when=mid, symbol=sample)

    payload = {
        "global_trust_class": "OPERATIONAL_ONLY",
        "scoped_certification": "READY_FOR_SCIENTIFIC_RERUN",
        "snapshot_id": sid,
        "snapshot_manifest_checksum": manifest.get("manifest_checksum"),
        "snapshot_equity_sha256": manifest.get("equity_sha256"),
        "protocol_version": PROTOCOL_VERSION,
        "frozen_protocols_sha256": _file_sha(REPO_ROOT / "PHASE_A5_FROZEN_PROTOCOLS.md"),
        "scoped_cert_sha256": _file_sha(
            REPO_ROOT / "PHASE_A5_SCOPED_DATA_CERTIFICATION.md"
        ),
        "adjustment_policy_version": manifest.get("adjustment_policy_version"),
        "n_symbols": int(closes.shape[1]),
        "n_sessions": int(len(closes)),
        "date_start": str(closes.index.min().date()),
        "date_end": str(closes.index.max().date()),
        "oos_start_date": OOS_START_DATE,
        "cost_model": "CNC round_trip_cost_pct",
        "cost_pct": round_trip_cost_pct("CNC"),
        "seed": 42,
        "git_sha": _git_sha(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "production_behaviour_changed": False,
        "phase_b_started": False,
        "pit_smoke": {
            "status": read.status,
            "symbol": sample,
            "as_of": mid,
            "n_bars": len(read.data),
            "snapshot_id": sid,
        },
        "manifest": {
            k: manifest.get(k)
            for k in (
                "snapshot_id", "trust_class", "research_grade",
                "scoped_certification", "scoped_eligible_for_scientific_rerun",
                "scope", "global_trust_class", "adjustment_policy_version",
                "manifest_checksum", "equity_sha256", "index_sha256",
                "date_range", "instrument_count",
            )
        },
        "experiments": experiments,
        "summary_matrix": summary_rows,
        "positive_evidence": [
            r for r in summary_rows if r["final_scientific_verdict"] == "PASS"
        ],
        "negative_evidence": [
            r for r in summary_rows if r["final_scientific_verdict"] == "FAIL"
        ],
        "inconclusive_evidence": [
            r for r in summary_rows if r["final_scientific_verdict"] == "INCONCLUSIVE"
        ],
    }
    payload["rerun_result_hash"] = _result_hash({
        k: experiments[k].get("result_hash") for k in FROZEN_IDS
    })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(render_rerun_markdown(payload), encoding="utf-8")
    payload["report_path"] = str(OUT_MD)
    payload["results_path"] = str(OUT_JSON)
    return payload


def _economic_label(eid: str, res: dict) -> str:
    sci = res.get("scientific_verdict")
    if eid == "EXP-A2-01":
        pos = [
            h for h, row in (res.get("horizons") or {}).items()
            if (row or {}).get("mean_net", 0) > 0
        ]
        return "NON_POSITIVE_FAMILY" if not pos else (
            "FDR_CLEARED" if sci == "PASS" else "POINT_POSITIVE_UNCONFIRMED"
        )
    if eid == "EXP-A3-01":
        delta = (res.get("metrics") or {}).get("economic_value_delta")
        return "NO_INCREMENTAL_VALUE" if (delta is not None and delta <= 0) else (
            "INCREMENTAL" if sci == "PASS" else "MIXED"
        )
    if eid == "EXP-A5-01":
        incr = (res.get("metrics") or {}).get("incremental_r2")
        return "NO_INCREMENT" if (incr is not None and incr <= 0) else "INCREMENT"
    if eid == "EXP-A6-01":
        incr = (res.get("metrics") or {}).get("conditioned_improvement")
        return "NO_CONDITIONED_IMPROVEMENT" if (incr is not None and incr <= 0) else "IMPROVEMENT"
    if eid == "EXP-A5A6-01":
        n = (res.get("metrics") or {}).get("n_fdr_interactions") or 0
        return "NO_FDR_INTERACTION" if n < 1 else "FDR_INTERACTION"
    return "N/A"


def render_rerun_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# Phase A.5 Research-Grade Scientific Rerun")
    lines.append("")
    lines.append(
        "> Scientific rerun of the **frozen** Phase A.5 protocols against scoped "
        "certified snapshot "
        f"`{payload.get('snapshot_id')}`. "
        "**Global trust remains `OPERATIONAL_ONLY`.** "
        "Production behaviour unchanged. Phase B not started."
    )
    lines.append("")
    lines.append("## 1. Executive summary")
    lines.append("")
    lines.append(
        "QuantTerm re-tested five frozen research ideas using verified NSE history "
        "for the exact 29-name panel. Global database quality is still not fully "
        "certified; this rerun only uses the scoped certified snapshot."
    )
    lines.append("")
    for row in payload.get("summary_matrix") or []:
        eid = row["capability"]
        sci = row["final_scientific_verdict"]
        plain = ((payload["experiments"][eid].get("plain") or {}).get("layer1") or {})
        lines.append(f"### {eid} — **{sci}**")
        lines.append("")
        lines.append(plain.get("explanation") or "")
        lines.append("")
        lines.append(f"- Next action: `{row['next_action']}`")
        lines.append("")
    lines.append("## 2. Scoped certification reference")
    lines.append("")
    lines.append("- Source: `PHASE_A5_SCOPED_DATA_CERTIFICATION.md`")
    lines.append(
        f"- Hash: `{payload.get('scoped_cert_sha256')}`"
    )
    lines.append("- Scoped status: `READY_FOR_SCIENTIFIC_RERUN`")
    lines.append("- Global trust: `OPERATIONAL_ONLY` (unchanged)")
    lines.append(
        "- Identity 29/29 VERIFIED · CA unresolved consecutive 0 · "
        "universe FIXED_PREREGISTERED_29 · sector static map · "
        "price unresolved rate 0.0"
    )
    lines.append("")
    lines.append("## 3. Snapshot / provenance")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| snapshot_id | `{payload.get('snapshot_id')}` |")
    lines.append(
        f"| manifest_checksum | `{payload.get('snapshot_manifest_checksum')}` |"
    )
    lines.append(
        f"| equity_sha256 | `{payload.get('snapshot_equity_sha256')}` |"
    )
    lines.append(f"| securities | {payload.get('n_symbols')} |")
    lines.append(
        f"| date range | {payload.get('date_start')} → {payload.get('date_end')} |"
    )
    lines.append(f"| sessions | {payload.get('n_sessions')} |")
    lines.append(f"| oos_start (frozen) | {payload.get('oos_start_date')} |")
    lines.append(f"| cost model | {payload.get('cost_model')} = {payload.get('cost_pct')}% |")
    lines.append(f"| seed | {payload.get('seed')} |")
    lines.append(f"| git_sha | `{payload.get('git_sha')}` |")
    lines.append(f"| protocol | `{payload.get('protocol_version')}` |")
    lines.append(
        f"| frozen_protocols_sha256 | `{payload.get('frozen_protocols_sha256')}` |"
    )
    lines.append(f"| rerun_result_hash | `{payload.get('rerun_result_hash')}` |")
    lines.append(f"| evaluated_at | {payload.get('evaluated_at')} |")
    lines.append("")

    # Per-experiment sections 4-8
    section_n = 4
    for eid in FROZEN_IDS:
        res = payload["experiments"][eid]
        lines.append(f"## {section_n}. {eid} result")
        section_n += 1
        lines.append("")
        plain = (res.get("plain") or {}).get("layer1") or {}
        lines.append(f"**Plain English:** {plain.get('explanation')}")
        lines.append("")
        lines.append(f"- Hypothesis ID: `{res.get('hypothesis_id')}`")
        lines.append(f"- Internal verdict tag: `{res.get('verdict')}`")
        lines.append(f"- Scientific verdict: **{res.get('scientific_verdict')}**")
        lines.append(f"- Reason: {res.get('reason')}")
        lines.append(f"- Registry status: `{res.get('registry_status')}`")
        lines.append(f"- Result hash: `{res.get('result_hash')}`")
        lines.append(f"- Production authority: `{res.get('production_authority')}`")
        lines.append(f"- Next action: `{res.get('next_action')}`")
        lines.append("")
        lines.append("### Technical evidence")
        lines.append("")
        lines.append("```json")
        tech = {
            "metrics": res.get("metrics"),
            "fdr": res.get("fdr"),
            "baselines": res.get("baselines"),
            "methods": res.get("methods"),
            "partial_incr_risk": res.get("partial_incr_risk"),
            "feature_stats": res.get("feature_stats"),
            "horizons": res.get("horizons"),
            "interactions": res.get("interactions"),
            "vs_naive_verdict": (res.get("metrics") or {}).get("vs_naive_verdict"),
            "vs_rank_verdict": (res.get("metrics") or {}).get("vs_rank_verdict"),
            "economic_value_delta": (res.get("metrics") or {}).get("economic_value_delta"),
        }
        # Drop nulls for readability
        tech = {k: v for k, v in tech.items() if v is not None}
        lines.append(json.dumps(tech, indent=2, default=str))
        lines.append("```")
        lines.append("")

    lines.append(f"## {section_n}. DISPLAY_ONLY vs certified comparison")
    section_n += 1
    lines.append("")
    lines.append(
        "| EXPERIMENT | DISPLAY_ONLY | CERTIFIED (raw) | SCIENTIFIC | "
        "DIRECTION CHANGED |"
    )
    lines.append("|---|---|---|---|---|")
    for eid in FROZEN_IDS:
        res = payload["experiments"][eid]
        cmp_ = res.get("display_only_comparison") or {}
        lines.append(
            f"| {eid} | {cmp_.get('previous_verdict')} | {cmp_.get('certified_verdict')} | "
            f"{cmp_.get('certified_scientific_verdict')} | {cmp_.get('direction_changed')} |"
        )
    lines.append("")
    lines.append(
        "Disagreement with the exploratory DISPLAY_ONLY result is not treated as an "
        "error — the point of this rerun is to see what survives certified data."
    )
    lines.append("")

    lines.append(f"## {section_n}. Statistical evidence")
    section_n += 1
    lines.append("")
    for eid in FROZEN_IDS:
        res = payload["experiments"][eid]
        fdr = res.get("fdr") or {}
        lines.append(
            f"- **{eid}**: scientific=`{res.get('scientific_verdict')}`; "
            f"FDR rejected=`{fdr.get('rejected')}`; "
            f"metrics=`{json.dumps(res.get('metrics'), default=str)}`"
        )
    lines.append("")

    lines.append(f"## {section_n}. Economic evidence")
    section_n += 1
    lines.append("")
    for row in payload.get("summary_matrix") or []:
        lines.append(
            f"- **{row['capability']}**: economic label `{row['economic_verdict']}`"
        )
    a2 = payload["experiments"]["EXP-A2-01"].get("horizons") or {}
    if a2:
        lines.append("")
        lines.append("Horizon family (cost-aware OOS):")
        lines.append("")
        lines.append(
            "| Horizon | n | n_eff | expectancy | CI95 | PF | Sharpe | "
            "DD | cost_drag | stat | econ |"
        )
        lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|")
        for h, row in a2.items():
            lines.append(
                f"| {h} | {row.get('n')} | {row.get('n_eff')} | {row.get('expectancy')} | "
                f"{row.get('ci_95')} | {row.get('profit_factor')} | {row.get('sharpe')} | "
                f"{row.get('max_drawdown_R')} | {row.get('cost_drag')} | "
                f"{row.get('statistical_verdict')} | {row.get('economic_verdict')} |"
            )
    lines.append("")

    lines.append(f"## {section_n}. Cost impact")
    section_n += 1
    lines.append("")
    lines.append(
        f"- Frozen cost model: CNC `round_trip_cost_pct` = **{payload.get('cost_pct')}%**"
    )
    lines.append(
        "- EXP-A2-01 applies conservative ~100% one-way turnover cost drag per rebalance."
    )
    lines.append(
        "- EXP-A3-01 bake-off uses the same cost-aware economic delta machinery as frozen."
    )
    lines.append("")

    lines.append(f"## {section_n}. Multiple-testing / FDR")
    section_n += 1
    lines.append("")
    for eid in FROZEN_IDS:
        fdr = payload["experiments"][eid].get("fdr") or {}
        lines.append(f"- **{eid}**: rejected={fdr.get('rejected')}; detail keys="
                     f"{list((fdr.get('detail') or {}).keys())}")
    lines.append("")

    lines.append(f"## {section_n}. Positive evidence")
    section_n += 1
    lines.append("")
    pos = payload.get("positive_evidence") or []
    if not pos:
        lines.append("None — no experiment earned PASS under frozen criteria.")
    else:
        for r in pos:
            lines.append(f"- `{r['capability']}` → PASS → `{r['next_action']}`")
    lines.append("")

    lines.append(f"## {section_n}. Negative evidence")
    section_n += 1
    lines.append("")
    neg = payload.get("negative_evidence") or []
    if not neg:
        lines.append("None.")
    else:
        for r in neg:
            lines.append(
                f"- `{r['capability']}` → FAIL → `{r['next_action']}` "
                "(recorded in scientific memory; do not escalate model complexity)"
            )
    lines.append("")

    lines.append(f"## {section_n}. Inconclusive evidence")
    section_n += 1
    lines.append("")
    inc = payload.get("inconclusive_evidence") or []
    if not inc:
        lines.append("None.")
    else:
        for r in inc:
            lines.append(f"- `{r['capability']}` → INCONCLUSIVE → `{r['next_action']}`")
    lines.append("")

    lines.append(f"## {section_n}. Scientific-memory updates")
    section_n += 1
    lines.append("")
    lines.append(
        "FAIL outcomes were written as negative evidence; INCONCLUSIVE as WATCH "
        "beliefs in the isolated Phase A.5 scientific memory DB "
        "(`logs/phase_a5/scientific_memory.db`). Registry results updated on the "
        "frozen hypothesis IDs in `logs/phase_a5/experiments.db`."
    )
    lines.append("")

    lines.append(f"## {section_n}. Reproducibility")
    section_n += 1
    lines.append("")
    lines.append("```")
    lines.append(f"snapshot_id={payload.get('snapshot_id')}")
    lines.append(f"protocol_version={payload.get('protocol_version')}")
    lines.append(f"git_sha={payload.get('git_sha')}")
    lines.append(f"seed={payload.get('seed')}")
    lines.append(f"cost_pct={payload.get('cost_pct')}")
    lines.append(f"oos_start={payload.get('oos_start_date')}")
    lines.append(f"rerun_result_hash={payload.get('rerun_result_hash')}")
    lines.append("runner=python -m research.phase_a5.scientific_rerun")
    lines.append("```")
    lines.append("")

    lines.append(f"## {section_n}. Production behaviour unchanged")
    section_n += 1
    lines.append("")
    lines.append(f"- `production_behaviour_changed`: `{payload.get('production_behaviour_changed')}`")
    lines.append(f"- `phase_b_started`: `{payload.get('phase_b_started')}`")
    lines.append(
        "- Brain / CycleContext / ranking / portfolio authority / risk limits / "
        "execution / broker / live signals: **not modified**."
    )
    lines.append(
        "- Even PASS only means eligible for confirmation review, not live use."
    )
    lines.append("")

    lines.append(f"## {section_n}. What QuantTerm learned (plain English)")
    section_n += 1
    lines.append("")
    lines.append(
        "We re-checked five frozen research ideas with cleaner, verified history "
        "for this specific test panel. The full market database is still not "
        "certified overall. For these five ideas:"
    )
    lines.append("")
    for row in payload.get("summary_matrix") or []:
        eid = row["capability"]
        sci = row["final_scientific_verdict"]
        if sci == "FAIL":
            lines.append(
                f"- **{eid}**: still no reliable advantage — do not use it."
            )
        elif sci == "PASS":
            lines.append(
                f"- **{eid}**: showed a reliable historical advantage — worth "
                "further confirmation, not live trading yet."
            )
        else:
            lines.append(
                f"- **{eid}**: data are trustworthy now, but evidence is still "
                "too weak to decide."
            )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Final matrix")
    lines.append("")
    lines.append(
        "| CAPABILITY | DISPLAY_ONLY RESULT | CERTIFIED RESULT | "
        "STATISTICAL VERDICT | ECONOMIC VERDICT | FINAL SCIENTIFIC VERDICT | "
        "NEXT ACTION |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in payload.get("summary_matrix") or []:
        lines.append(
            f"| {row['capability']} | {row['display_only']} | {row['certified_raw']} | "
            f"{row['statistical_verdict']} | {row['economic_verdict']} | "
            f"**{row['final_scientific_verdict']}** | `{row['next_action']}` |"
        )
    lines.append("")
    lines.append(
        "STOP. Do not begin Phase B. Do not implement production changes from PASS. "
        "Do not escalate models from FAIL."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out = run_scientific_rerun()
    print(json.dumps({
        "snapshot_id": out["snapshot_id"],
        "global_trust_class": out["global_trust_class"],
        "production_behaviour_changed": out["production_behaviour_changed"],
        "phase_b_started": out["phase_b_started"],
        "summary": out["summary_matrix"],
        "report_path": out["report_path"],
        "rerun_result_hash": out["rerun_result_hash"],
    }, indent=2, default=str))
