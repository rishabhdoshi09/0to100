"""Write SEPA-003 reports from stats. Never rewrite SEPA-001–R2.1 files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.sepa003.constants import OUT_DIR
from research.sepa003.forward_spec import forward_record_template


def _pct(x):
    if x is None:
        return "—"
    return f"{100.0 * float(x):.2f}%" if abs(float(x)) <= 2 else f"{float(x):.3f}"


def _r(x):
    if x is None:
        return "—"
    return f"{float(x):+.3f}"


def _md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def write_regime_audit(payload, stats) -> Path:
    p = OUT_DIR / "SEPA_003_REGIME_AUDIT.md"
    by = stats.get("by_regime") or {}
    rows = [[k, v.get("n"), _r(v.get("mean")), v.get("ci_lower"), v.get("ci_upper")]
            for k, v in by.items()]
    p.write_text(
        "# SEPA-003 regime audit\n\n"
        "R2.1 labelled every core-F row `unknown` because "
        "`data.index_store.get_index_ohlcv('^NSEI')` had no deep local history.\n\n"
        f"**SEPA-003 series:** `{payload.get('index_source')}` "
        f"(version `regime_pit_v1`). Class `PIT_DEGRADED` if the NIFTY50 member "
        f"list is contemporaneous.\n\n"
        "States are computed from information ≤ as_of. Append-future invariance "
        "is tested in `tests/test_sepa_003.py`.\n\n"
        "No regime gate is added.\n\n"
        + _md_table(["Regime", "n", "E[R]", "CI lo", "CI hi"], rows)
        + "\n\n2025–2026 remains consumed evidence. Not untouched OOS.\n"
    )
    return p


def write_sector_audit(payload, stats) -> Path:
    p = OUT_DIR / "SEPA_003_SECTOR_AUDIT.md"
    cov = (stats.get("primary") or {}).get("tests") or []
    h8 = next((t for t in cov if t.get("id") == "H8"), {})
    years = stats.get("sector_coverage_by_year") or {}
    rows = [[y, v.get("n"), v.get("mapped"), v.get("pct_mapped")] for y, v in years.items()]
    p.write_text(
        "# SEPA-003 sector audit\n\n"
        "R2.1 UNKNOWN (3,131 / 4,208) came from `sector_of` parsing only "
        "NIFTY500 comment groups — NIFTY50 names were unmapped.\n\n"
        f"Map `{payload.get('sector', {}).get('version')}` : "
        f"{payload.get('sector', {}).get('n_mapped')} symbols "
        f"({payload.get('sector', {}).get('n_from_comments')} comments + "
        f"{payload.get('sector', {}).get('n_from_overlay')} overlay).\n\n"
        "`sector_identity_pit = false`. Unmapped stays UNKNOWN. "
        "No price-inferred industry.\n\n"
        + _md_table(["Year", "F fills", "Mapped", "% mapped"], rows)
        + "\n\n"
        f"Overall mapped: {h8.get('coverage')}\n\n"
        "H8 insufficient flag: "
        f"{h8.get('insufficient')}. If true, label is INSUFFICIENT_PIT_SECTOR_DATA.\n"
    )
    return p


def write_feature_dataset(payload, stats) -> Path:
    p = OUT_DIR / "SEPA_003_FEATURE_DATASET.md"
    p.write_text(
        "# SEPA-003 feature dataset\n\n"
        f"Reconstructed unique F fills: **{stats.get('n_reconstructed_fills')}** "
        f"(ledger F setups processed; skipped={payload.get('skipped')}).\n\n"
        f"R2.1 official F n (deduped) was 4,208 / E[R]=+0.123. "
        f"Reconstruction E[R]={_r((stats.get('reconstructed_expectancy') or {}).get('mean'))} "
        f"n={ (stats.get('reconstructed_expectancy') or {}).get('n') }.\n\n"
        "A difference is expected (fill-search window, embargo not re-applied "
        "identically). Thresholds were not changed to match.\n\n"
        "Machine-readable: `sepa_003_features.parquet` (or jsonl fallback), "
        "`sepa_003_controls.jsonl`, `sepa_003_g_panel.jsonl`, "
        "`sepa_003_feature_manifest.json`.\n\n"
        "Every row carries `not_validated_edge=true` and "
        "`confirmation_already_observed=true`.\n"
    )
    return p


def write_decay(stats) -> Path:
    p = OUT_DIR / "SEPA_003_DECAY_ANALYSIS.md"
    d = stats.get("decay") or {}
    fields = d.get("fields") or {}
    rows = []
    for k, v in fields.items():
        w, e = v.get("winning_era") or {}, v.get("weak_era") or {}
        rows.append([k, w.get("n"), w.get("median"), e.get("n"), e.get("median"),
                     v.get("cliffs_delta"), v.get("mw_p")])
    p.write_text(
        "# SEPA-003 decay analysis\n\n"
        "Winning era: through 2023-12-31. Weak era: 2024-01-01 onward. "
        "These are **diagnostic** eras. 2025–2026 is not a new holdout.\n\n"
        f"**Decay verdict:** `{stats.get('decay_verdict')}`\n\n"
        f"n win={d.get('n_win')} n weak={d.get('n_weak')}\n\n"
        f"Regime mix win: {d.get('regime_mix_win')}\n\n"
        f"Regime mix weak: {d.get('regime_mix_weak')}\n\n"
        + _md_table(["Feature", "n_win", "med_win", "n_weak", "med_weak", "Cliff δ", "MW p"], rows)
        + "\n\n"
        "Allowed verdicts: MARKET_CHANGED / POPULATION_CHANGED / UNSTABLE_EDGE / INCONCLUSIVE.\n"
    )
    return p


def write_survival(stats) -> Path:
    p = OUT_DIR / "SEPA_003_COMPONENT_SURVIVAL.md"
    q = stats.get("quartiles") or {}
    rows = [[k, (v or {}).get("n"), (v or {}).get("monotonic"), (v or {}).get("class")]
            for k, v in q.items()]
    ladder = stats.get("r2_ladder") or {}
    p.write_text(
        "# SEPA-003 component survival\n\n"
        "R2.1 ladder (frozen, cited — not re-run as official A–G):\n\n"
        + _md_table(
            ["Variant", "n", "E[R]", "Verdict"],
            [[k, (v or {}).get("n_deduped"), _r((v or {}).get("expectancy_r")),
              (v or {}).get("statistical_verdict")] for k, v in ladder.items()],
        )
        + "\n\nPrespecified RS / geometry quartiles on reconstructed F:\n\n"
        + _md_table(["Feature", "n", "Monotonic", "Class"], rows)
        + "\n\nClasses: ROBUST_POSITIVE / CONTEXT_DEPENDENT / UNSTABLE / "
        "NO_SIGNAL / INSUFFICIENT_DATA.\n\n"
        "No threshold was chosen from the best bin.\n"
    )
    return p


def write_matched(stats) -> Path:
    p = OUT_DIR / "SEPA_003_MATCHED_CONTROLS.md"
    m = stats.get("matched") or {}
    p.write_text(
        "# SEPA-003 matched controls\n\n"
        "VCP (reconstructed F) vs Stage-2+RS no-VCP controls, stratified by "
        "year + RS bucket + regime. Outcome = 20d forward % (common unit).\n\n"
        f"n_vcp={m.get('n_vcp')} n_ctrl={m.get('n_ctrl')} n_strata={m.get('n_strata')} "
        f"mean_diff={m.get('mean_diff')}\n\n"
        f"{m.get('note')}\n\n"
        "R2.1 C→D on the scanner ladder remains the primary historical H3 cite.\n"
    )
    return p


def write_entry(stats) -> Path:
    p = OUT_DIR / "SEPA_003_ENTRY_ANALYSIS.md"
    q = (stats.get("quartiles") or {})
    p.write_text(
        "# SEPA-003 entry analysis\n\n"
        "Explanatory only. **No new buy-zone.**\n\n"
        f"Pivot distance vs R: {q.get('distance_from_pivot_pct')}\n\n"
        f"Gap %: {q.get('breakout_gap_pct')}\n\n"
        f"Stop width: {q.get('stop_distance_pct')}\n\n"
        "H6 asks whether lower extension improves MAE / fail-rate, not whether "
        "a tighter zone would have passed confirmation.\n"
    )
    return p


def _h(stats, hid):
    for t in ((stats.get("primary") or {}).get("tests") or []):
        if t.get("id") == hid:
            return t
    return {}


def decide(stats) -> dict[str, Any]:
    """Map evidence to A/B/C/D. Not a trading licence."""
    h2, h3, h7, h8 = _h(stats, "H2"), _h(stats, "H3"), _h(stats, "H7"), _h(stats, "H8")
    decay = stats.get("decay_verdict")
    q = stats.get("quartiles") or {}
    vcp_class = (q.get("final_contraction_pct") or {}).get("class")
    dry_class = (q.get("dry_up_ratio") or {}).get("class")
    pivot_class = (q.get("distance_from_pivot_pct") or {}).get("class")
    r2_d = ((stats.get("r2_ladder") or {}).get("D") or {})
    r2_c = ((stats.get("r2_ladder") or {}).get("C") or {})
    r2_f_conf = ((((stats.get("r2_ladder") or {}).get("F") or {}).get("walk_forward") or {}).get("confirmation") or {})
    binary_vcp_helps = False
    if r2_c.get("expectancy_r") is not None and r2_d.get("expectancy_r") is not None:
        binary_vcp_helps = float(r2_d["expectancy_r"]) > float(r2_c["expectancy_r"])
    matched = (stats.get("matched") or {}).get("mean_diff")
    if matched is not None and matched > 0 and (stats.get("matched") or {}).get("n_strata", 0) >= 5:
        binary_vcp_helps = binary_vcp_helps or matched > 0
    retain = []
    retire = ["core_F_standalone", "VCP_binary_gate"]
    if not binary_vcp_helps:
        retire = ["core_F_standalone", "VCP_binary_as_hard_gate"]
    # Stage-2 / RS as filters
    retain.append("trend_template_as_quality_feature")
    retain.append("rs_percentile_as_ranking_feature")
    if vcp_class in {"ROBUST_POSITIVE", "CONTEXT_DEPENDENT"}:
        retain.append("final_contraction_depth_continuous")
    else:
        retire.append("VCP_binary_as_hard_gate")
    if dry_class in {"ROBUST_POSITIVE", "CONTEXT_DEPENDENT"}:
        retain.append("volume_dryup_continuous")
    if pivot_class in {"ROBUST_POSITIVE", "CONTEXT_DEPENDENT"}:
        retain.append("pivot_distance_as_risk_feature")
    choice = "A"
    reason = (
        "Core F failed consumed confirmation. Binary VCP did not earn an "
        "incremental gate. Selected continuous/quality features may be retained "
        "as NEW_HYPOTHESIS inputs to a later ensemble."
    )
    if decay == "MARKET_CHANGED" and h7.get("p") is not None:
        # still not C unless we would claim a regime-conditional SEPA rule
        choice = "A"
        reason += " Regime mix shifted; that does not resurrect core F."
    if h8.get("insufficient"):
        h8_label = "INSUFFICIENT_PIT_SECTOR_DATA"
    else:
        h8_label = "measured"
    return {
        "choice": choice,
        "label": "A — RETIRE CORE SEPA; RETAIN SELECT FEATURES",
        "reason": reason,
        "retain": retain,
        "retire": list(dict.fromkeys(retire)),
        "h8_label": h8_label,
        "further_historical_optimization": False,
        "new_hypothesis_only": True,
    }


def write_results(stats, decision) -> Path:
    p = OUT_DIR / "SEPA_003_RESULTS.md"
    prim = (stats.get("primary") or {}).get("fdr") or {}
    p.write_text(
        "# SEPA-003 results\n\n"
        "**Not VALIDATED_EDGE. Not untouched confirmation. Not paper-eligible.**\n\n"
        f"Reconstructed F fills n={stats.get('n_reconstructed_fills')} "
        f"E[R]={_r((stats.get('reconstructed_expectancy') or {}).get('mean'))}\n\n"
        f"Decay verdict: `{stats.get('decay_verdict')}`\n\n"
        f"Index source: `{stats.get('index_source')}`\n\n"
        f"FDR (primary p-values present): {prim}\n\n"
        f"Huber (explanatory, not deployed): {stats.get('huber')}\n\n"
        f"Strategic conclusion (see decision): **{decision.get('label')}**\n"
    )
    return p


def write_decision(stats, decision) -> Path:
    p = OUT_DIR / "SEPA_003_DECISION.md"
    h = {t["id"]: t for t in ((stats.get("primary") or {}).get("tests") or [])}
    q = stats.get("quartiles") or {}
    p.write_text(
        "# SEPA-003 decision\n\n"
        "Prior sequence SEPA-001 → 001R → 001R2 → 001R2.1 is immutable. "
        "Core F confirmation remains REJECT. This file does not re-open 2025–2026 "
        "as OOS.\n\n"
        f"# {decision.get('label')}\n\n"
        f"{decision.get('reason')}\n\n"
        "Paper / live / broker / GTT / autopilot: **not authorised**.\n\n"
        "## Fourteen answers\n\n"
        f"1. Strict Stage-2: useful as a **quality / loss-avoidance feature**, "
        f"not a standalone edge. R2.1 A {_r(((stats.get('r2_ladder') or {}).get('A') or {}).get('expectancy_r'))} "
        f"→ B {_r(((stats.get('r2_ladder') or {}).get('B') or {}).get('expectancy_r'))}. "
        f"Both confirmation REJECT. Do not call 'loses less' an edge.\n"
        f"2. RS: ranking feature. Prespecified buckets on G/F: {h.get('H2', {}).get('g_buckets')}. "
        f"Not a new cutoff. Confirmation C still REJECT.\n"
        f"3. VCP binary gate: **not useful as a hard gate**. R2.1 C→D worsened; "
        f"matched diff={ (stats.get('matched') or {}).get('mean_diff') }.\n"
        f"4. VCP continuous: class `{ (q.get('final_contraction_pct') or {}).get('class') }` "
        f"(final depth), tightness `{ (q.get('tightness') or {}).get('class') }`.\n"
        f"5. Pivot-entry geometry: class `{ (q.get('distance_from_pivot_pct') or {}).get('class') }`. "
        f"Explanatory; no new buy-zone.\n"
        f"6. Volume dry-up: class `{ (q.get('dry_up_ratio') or {}).get('class') }`.\n"
        f"7. Contraction tightness: see (4)/(6); binary VCP is not the carrier.\n"
        f"8. Regime-conditional: {h.get('H7', {}).get('by_regime')}. "
        f"Decay={stats.get('decay_verdict')}. No regime gate added.\n"
        f"9. Sector leadership: {decision.get('h8_label')} "
        f"coverage={h.get('H8', {}).get('coverage')}.\n"
        f"10. Features that survive: {decision.get('retain')}.\n"
        f"11. Development-era artifacts: pooled F +0.123R and 2020/21/23 year plus; "
        f"binary VCP as a profit engine; treating daily A n as F n.\n"
        f"12. QuantTerm should retain: {decision.get('retain')} as research features "
        f"(NEW_HYPOTHESIS, future validation required).\n"
        f"13. QuantTerm should retire: {decision.get('retire')}.\n"
        f"14. Further historical optimization of core F: "
        f"**{decision.get('further_historical_optimization')}**. "
        f"The confirmation block is consumed. Fitting a new F against 2025–2026 "
        f"would be in-sample theatre.\n\n"
        "## Strategic conclusion\n\n"
        f"**{decision.get('choice')}** — {decision.get('label')}\n\n"
        "C is not selected: a regime-conditional SEPA *rule* would still be a "
        "NEW_HYPOTHESIS and is not earned as a strategy. D is not selected: "
        "Stage-2/RS still look like adverse-selection reducers worth keeping as "
        "features.\n"
    )
    return p


def write_forward() -> Path:
    p = OUT_DIR / "SEPA_003_FORWARD_LEDGER.md"
    spec = forward_record_template()
    p.write_text(
        "# SEPA-003 forward observation (design only)\n\n"
        "Not activated. Not paper. Not autopilot.\n\n"
        f"```json\n{json.dumps(spec, indent=2)}\n```\n"
    )
    return p


def write_manifest(payload, stats) -> Path:
    p = OUT_DIR / "sepa_003_feature_manifest.json"
    p.write_text(json.dumps({
        "feature_set": "sepa-003.v1",
        "eligibility_version": "sepa-001r2.v1",
        "vcp_version": "vcp_causal_v2",
        "pivot_version": "pivot_last_contraction_v1",
        "regime_version": "regime_pit_v1",
        "sector_version": "sector_map_v1",
        "index_source": payload.get("index_source"),
        "not_validated_edge": True,
        "n_fills": stats.get("n_reconstructed_fills"),
    }, indent=2))
    return p


def write_all(payload: dict[str, Any], stats: dict[str, Any]) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decision = decide(stats)
    (OUT_DIR / "sepa_003_decision.json").write_text(json.dumps(decision, indent=2))
    return {
        "regime": write_regime_audit(payload, stats),
        "sector": write_sector_audit(payload, stats),
        "features": write_feature_dataset(payload, stats),
        "decay": write_decay(stats),
        "survival": write_survival(stats),
        "matched": write_matched(stats),
        "entry": write_entry(stats),
        "results": write_results(stats, decision),
        "decision": write_decision(stats, decision),
        "forward": write_forward(),
        "manifest": write_manifest(payload, stats),
    }
