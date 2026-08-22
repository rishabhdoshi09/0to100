"""Write FEATURE-001 markdown reports from frozen stats. Explanatory only."""
from __future__ import annotations

import json
from typing import Any

from research.feature001.constants import (
    FAMILY_KEYS,
    FDR_Q,
    MIN_N,
    OUT_DIR,
    RS_BUCKETS,
    TREND_BUCKETS,
    YEARS,
)
from research.feature001.forward_spec import forward_record_template


def _pct(x) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _n(x) -> str:
    if x is None:
        return "—"
    return f"{float(x):.3f}"


def _ci(block) -> str:
    if not block:
        return "—"
    m = block.get("mean")
    if m is None:
        return "—"
    lo, hi = block.get("ci_lower"), block.get("ci_upper")
    if lo is None or hi is None:
        return f"{m:.3f}"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _base_line(name: str, b: dict[str, Any]) -> str:
    exp = _ci((b or {}).get("expectancy_r") or {})
    return (
        f"| {name} | {b.get('n', 0)} | {exp} | {_pct(b.get('win_rate'))} | "
        f"{_n(b.get('pf'))} | {_n(b.get('mae'))} | {_n(b.get('mfe'))} | "
        f"{_pct(b.get('stop_before_1r'))} | {_pct(b.get('hit_1r'))} | "
        f"{_pct(b.get('hit_2r'))} | {_n(b.get('avg_winner'))} | "
        f"{_n(b.get('avg_loser'))} | {_n(b.get('drawdown_proxy_r'))} | "
        f"{_n(b.get('frequency_per_year'))} |"
    )


BASE_HDR = (
    "| Family | n | E[R] (CI) | Win rate | PF | MAE | MFE | "
    "Stop<1R | +1R | +2R | Avg win | Avg loss | DD proxy | Freq/yr |"
)
BASE_SEP = "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def _write(name: str, body: str):
    path = OUT_DIR / name
    path.write_text(body.rstrip() + "\n")
    return path


def write_baselines(stats: dict, meta: dict) -> None:
    lines = [
        "# FEATURE-001 — Family baselines",
        "",
        "**Claim class:** EXPLANATORY. Already-consumed history. Not VALIDATED_EDGE.",
        f"**Events:** {stats.get('n_events')} filled fires · "
        f"**Family rows:** {stats.get('n_family_rows')} · "
        f"**Grid:** every {meta.get('sample_step')} sessions, horizon {meta.get('horizon')}.",
        "",
        "Families are reported separately. Do not read a blended average as a strategy result.",
        "",
        BASE_HDR, BASE_SEP,
    ]
    for fam in FAMILY_KEYS:
        lines.append(_base_line(fam, stats["baselines"].get(fam) or {}))
    lines += ["", "## Category rollups (after family-level, not a substitute)", "",
              BASE_HDR, BASE_SEP]
    for cat, b in (stats.get("category_baselines") or {}).items():
        lines.append(_base_line(cat, b))
    lines += [
        "",
        "## Sampling note",
        "",
        meta.get("note") or "",
        f"First date `{meta.get('first_date')}` · last date `{meta.get('last_date')}` · "
        f"identity calibration = {meta.get('identity_calibration')}.",
    ]
    _write("FEATURE_001_BASELINES.md", "\n".join(lines))


def write_trend(stats: dict) -> None:
    lines = [
        "# FEATURE-001 — Trend contribution",
        "",
        "**Question:** conditional on a family firing, does stronger trend structure improve that family's outcome?",
        "Not: «is Stage-2 profitable?»",
        "",
        "Prespecified buckets: `strict` = 7/7 structure_pass; `near` = 5–6 rules; `non` = <5.",
        "",
        "| Family | n | class | Δ E[R] strong−weak | Spearman n_passed vs R | residual after mom | tail strong | tail weak |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMILY_KEYS:
        t = stats["trend_study"][fam]
        lines.append(
            f"| {fam} | {t['n']} | {t['classification']} | {_n(t.get('strong_vs_weak_delta_r'))} | "
            f"{_n((t.get('spearman') or {}).get('rho'))} | "
            f"{_n((t.get('residual_after_mom') or {}).get('rho'))} | "
            f"{_pct(t.get('tail_rate_strong'))} | {_pct(t.get('tail_rate_weak'))} |"
        )
    lines += ["", "## Bucket baselines by family", ""]
    for fam in FAMILY_KEYS:
        t = stats["trend_study"][fam]
        if t["n"] < 10:
            continue
        lines += [f"### {fam}", "", BASE_HDR, BASE_SEP]
        for b in TREND_BUCKETS:
            lines.append(_base_line(b, (t.get("buckets") or {}).get(b) or {}))
        lines.append("")
    _write("FEATURE_001_TREND_STUDY.md", "\n".join(lines))


def write_rs(stats: dict) -> None:
    lines = [
        "# FEATURE-001 — RS contribution (`rs_cs_v1`)",
        "",
        "Methodology unchanged. RS ≥ 70 is a **descriptive** flag. "
        "A better 95–99 bucket is not a licence to move the cutoff.",
        "",
        "Prespecified buckets: `<50`, `50–69`, `70–79`, `80–89`, `90–94`, `95–99`.",
        "",
        "| Family | n | class | Δ E[R] ≥70 vs <50 | Spearman RS vs R | residual after mom |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for fam in FAMILY_KEYS:
        t = stats["rs_study"][fam]
        lines.append(
            f"| {fam} | {t['n']} | {t['classification']} | {_n(t.get('strong_vs_weak_delta_r'))} | "
            f"{_n((t.get('spearman') or {}).get('rho'))} | "
            f"{_n((t.get('residual_after_mom') or {}).get('rho'))} |"
        )
    lines += ["", "## Bucket baselines by family", ""]
    for fam in FAMILY_KEYS:
        t = stats["rs_study"][fam]
        if t["n"] < 10:
            continue
        lines += [f"### {fam}", "", BASE_HDR, BASE_SEP]
        for b in RS_BUCKETS:
            lines.append(_base_line(b, (t.get("buckets") or {}).get(b) or {}))
        lines.append("")
    _write("FEATURE_001_RS_STUDY.md", "\n".join(lines))


def write_interactions(stats: dict) -> None:
    inc = stats.get("incremental") or {}
    lines = [
        "# FEATURE-001 — Strategy interactions and momentum redundancy",
        "",
        "**Claim class:** EXPLANATORY.",
        "",
        "## Joint A/B/C/D (attribution, not a rule)",
        "",
        "A = family alone · B = + structure_pass · C = + RS≥70 descriptive · D = both.",
        "",
    ]
    for fam in FAMILY_KEYS:
        j = stats["joint"][fam]
        if (j["A_strategy_alone"] or {}).get("n", 0) < 10:
            continue
        lines += [f"### {fam}", "", BASE_HDR, BASE_SEP]
        for k, lab in (
            ("A_strategy_alone", "A alone"),
            ("B_plus_strong_trend", "B + trend"),
            ("C_plus_strong_rs", "C + RS"),
            ("D_trend_and_rs", "D both"),
        ):
            lines.append(_base_line(lab, j.get(k) or {}))
        lines.append("")
    lines += [
        "## Momentum overlap",
        "",
        f"- corr(RS, mom_score) = `{inc.get('corr_rs_mom_score')}`",
        f"- corr(RS, momentum_5d) = `{inc.get('corr_rs_momentum_5d')}`",
        f"- corr(Trend n_passed, mom_score) = `{inc.get('corr_trend_mom_score')}`",
        f"- corr(Trend, RS) = `{inc.get('corr_trend_rs')}`",
        f"- Trend vs R after mom residual Spearman = `{inc.get('trend_after_mom')}`",
        f"- RS vs R after mom residual Spearman = `{inc.get('rs_after_mom')}`",
        f"- Trend vs R after production score = `{inc.get('trend_after_score')}`",
        f"- RS vs R after production score = `{inc.get('rs_after_score')}`",
        "",
        "High correlation with mom_score plus a near-zero residual means **REDUNDANT**, not a new edge.",
        "",
        "## Per-family policy (evidence-filled)",
        "",
        "| Family | n | Trend | RS |",
        "|---|---:|---|---|",
    ]
    for fam in FAMILY_KEYS:
        p = stats["policy"][fam]
        lines.append(f"| {fam} | {p['n']} | {p['trend']} | {p['rs']} |")
    _write("FEATURE_001_STRATEGY_INTERACTIONS.md", "\n".join(lines))


def write_ranking(stats: dict) -> None:
    r = stats.get("ranking") or {}
    lines = [
        "# FEATURE-001 — Ranking study",
        "",
        "Research only. Production ranking is unchanged.",
        "",
        f"Days with ≥8 simultaneous fires: **{r.get('n_rank_days')}**.",
        "",
        "| Rank key | Top−bottom E[R] |",
        "|---|---|",
    ]
    for k, v in (r.get("top_minus_bottom") or {}).items():
        lines.append(f"| {k} | {_ci(v)} |")
    lines += ["", "Precision (share of top-quintile names with net_R > 0):", ""]
    for k, v in (r.get("precision_top_quintile") or {}).items():
        lines.append(f"- `{k}`: {_pct(v)}")
    lines += ["", r.get("note") or ""]
    _write("FEATURE_001_RANKING_STUDY.md", "\n".join(lines))
    (OUT_DIR / "feature_001_ranking.json").write_text(json.dumps(r, indent=2, default=str))


def write_risk(stats: dict) -> None:
    lines = [
        "# FEATURE-001 — Loss-avoidance / RISK_FILTER_VALUE",
        "",
        "«Loses less» is not automatically an edge. Tail improvement without a stable "
        "expectancy lift is labelled **RISK_FILTER_VALUE**, not ALPHA.",
        "",
        "| Family | Trend tail strong | Trend tail weak | Trend stop<1R strong | weak | RS class | Trend class |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for fam in FAMILY_KEYS:
        t = stats["trend_study"][fam]
        s = stats["rs_study"][fam]
        lines.append(
            f"| {fam} | {_pct(t.get('tail_rate_strong'))} | {_pct(t.get('tail_rate_weak'))} | "
            f"{_pct(t.get('stop_before_1r_strong'))} | {_pct(t.get('stop_before_1r_weak'))} | "
            f"{s['classification']} | {t['classification']} |"
        )
    _write("FEATURE_001_RISK_FILTER_STUDY.md", "\n".join(lines))


def write_temporal(stats: dict) -> None:
    te = stats.get("temporal_events") or {}
    lines = [
        "# FEATURE-001 — Temporal stability",
        "",
        "This history has already been used by SEPA-001…003. Year splits are **explanatory**, "
        "not a new confirmation sample. No cell may be labelled VALIDATED_EDGE.",
        "",
        "## Event-level by year",
        "", BASE_HDR, BASE_SEP,
    ]
    for y in YEARS:
        lines.append(_base_line(y, (te.get("by_year") or {}).get(y) or {}))
    lines += ["", "## Rolling 2-year blocks", "", BASE_HDR, BASE_SEP]
    for name, b in (te.get("rolling_2y") or {}).items():
        lines.append(_base_line(name, b))
    lines += ["", "## Family Trend Δ by year (strong−weak)", "",
              "| Family | " + " | ".join(YEARS) + " |",
              "|---|" + "|".join(["---:" ] * len(YEARS)) + "|"]
    for fam in FAMILY_KEYS:
        d = stats["trend_study"][fam].get("year_deltas") or {}
        cells = " | ".join(_n(d.get(y)) for y in YEARS)
        lines.append(f"| {fam} | {cells} |")
    lines += ["", "## Family RS Δ by year (high−low)", "",
              "| Family | " + " | ".join(YEARS) + " |",
              "|---|" + "|".join(["---:" ] * len(YEARS)) + "|"]
    for fam in FAMILY_KEYS:
        d = stats["rs_study"][fam].get("year_deltas") or {}
        cells = " | ".join(_n(d.get(y)) for y in YEARS)
        lines.append(f"| {fam} | {cells} |")
    _write("FEATURE_001_TEMPORAL_STABILITY.md", "\n".join(lines))


def write_deprecation() -> None:
    body = """# FEATURE-001 — SEPA deprecation map

Core SEPA remains `RETIRED_RESEARCH_BENCHMARK`. This milestone does **not** delete research code.

## Keep

- Canonical Trend Template arithmetic (`research/sepa/trend.py`) and `trend_features_v1`
- `rs_cs_v1` (`research/sepa/rs.py`) and `rs_features_v1`
- SEPA-001 / 001R / 001R2.1 / 003 experiment documents, configs, and result files
- Generic structural helpers (PIT universe, FastRS, frames) used outside SEPA
- Ideas 7-rule point scorer as **Trend Quality context** (keys `sepa_*` retained)

## Research-only (do not call from production money paths)

- Core F eligibility (`research/sepa/engine.py`)
- VCP hard-gate and causal VCP state machines used as SEPA gates
- Pivot / buy-zone experiments
- Any future autopilot plan that requires Core F

## Deprecate from production semantics

- Headlines that read as a trade licence: `SEPA Ready`, `SEPA BUY`, `MEETS SEPA`
- Desk copy that says Ready Stage-2 **is** Minervini SEPA approved for money
- Treating `sepa_score >= 40` as SEPA eligibility (it never was Core F)
- VCP page language that implies a validated Minervini system

## Do not delete in this milestone

`research/sepa/**`, `docs/overhaul/experiments/SEPA-*/**`, `scan/setup_engine.py` VCP archetype, `screener/vcp_scanner.py`. Dead-code deletion needs its own tested cleanup.
"""
    _write("FEATURE_001_SEPA_DEPRECATION.md", body)


def write_results(stats: dict, meta: dict) -> None:
    lines = [
        "# FEATURE-001 — Results (explanatory)",
        "",
        f"**n events** = {stats.get('n_events')} · **family rows** = {stats.get('n_family_rows')}",
        f"**Primary hypotheses predeclared:** {stats.get('n_primary_hypotheses')} · "
        f"**Tests recorded:** {stats.get('n_tested')} · FDR q = {FDR_Q}.",
        "",
        "No result in this file is VALIDATED_EDGE.",
        "",
        "## Hypotheses",
        "",
        "| ID | stat | n | p | q | FDR reject |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for t in stats.get("hypotheses") or []:
        lines.append(
            f"| {t.get('id')} | {_n(t.get('stat'))} | {t.get('n')} | "
            f"{_n(t.get('p'))} | {_n(t.get('q'))} | {t.get('fdr_reject')} |"
        )
    lines += [
        "",
        f"## Final feature status",
        "",
        f"- Trend (`trend_features_v1`): **{stats.get('final_trend')}**",
        f"- RS (`rs_features_v1` / `rs_cs_v1`): **{stats.get('final_rs')}**",
        "",
        "See companion study notes for family cells and year splits.",
        "",
        f"Dataset first/last: `{meta.get('first_date')}` → `{meta.get('last_date')}`; "
        f"Nifty bench available = {meta.get('nifty_bench_available')}.",
    ]
    _write("FEATURE_001_RESULTS.md", "\n".join(lines))


def _fam_list(stats, feat, cls) -> list[str]:
    return [f for f, p in stats["policy"].items() if p[feat] == cls and p["n"] >= MIN_N]


def write_decision(stats: dict, meta: dict) -> None:
    inc = stats.get("incremental") or {}
    rk = stats.get("ranking") or {}
    lines = [
        "# FEATURE-001 — Decision",
        "",
        "**Claim class:** EXPLANATORY. Production trading behaviour is unchanged.",
        f"**Core SEPA:** RETIRED_RESEARCH_BENCHMARK.",
        "",
        "## Final classification (exactly one each)",
        "",
        f"| Feature | Status |",
        f"|---|---|",
        f"| Trend (`trend_features_v1`) | {stats.get('final_trend')} |",
        f"| RS (`rs_features_v1`) | {stats.get('final_rs')} |",
        "",
        "## Answers",
        "",
        f"1. **Strategies that benefit from Trend:** {_fam_list(stats, 'trend', 'POSITIVE_RANK_FEATURE') or 'none at POSITIVE_RANK_FEATURE'}; "
        f"risk-filter cells: {_fam_list(stats, 'trend', 'RISK_FILTER_VALUE') or 'none'}.",
        f"2. **Strategies that benefit from RS:** {_fam_list(stats, 'rs', 'POSITIVE_RANK_FEATURE') or 'none at POSITIVE_RANK_FEATURE'}; "
        f"risk-filter cells: {_fam_list(stats, 'rs', 'RISK_FILTER_VALUE') or 'none'}.",
        f"3. **Harmed:** Trend {_fam_list(stats, 'trend', 'NEGATIVE') or 'none'}; "
        f"RS {_fam_list(stats, 'rs', 'NEGATIVE') or 'none'}.",
        f"4. **Trend useful mainly for:** {_use_mode(stats, 'trend')}.",
        f"5. **RS useful mainly for:** {_use_mode(stats, 'rs')}.",
        f"6. **Redundant with momentum?** corr(RS, mom_score)={inc.get('corr_rs_mom_score')}; "
        f"corr(Trend, mom_score)={inc.get('corr_trend_mom_score')}; "
        f"RS after mom={inc.get('rs_after_mom')}; Trend after mom={inc.get('trend_after_mom')}.",
        f"7. **Trend+RS incremental?** See joint D vs A in `FEATURE_001_STRATEGY_INTERACTIONS.md`. "
        f"Not a production AND-gate.",
        f"8. **Stable across years:** only families whose year-delta signs do not flip "
        f"(see `FEATURE_001_TEMPORAL_STABILITY.md`). Unstable cells stay UNSTABLE.",
        f"9. **Development-era artifacts:** 2020–2023 vs 2024–2026 event baselines "
        f"in the temporal note. Do not promote a 2020–21-only lift.",
        f"10. **Hard gate now?** No. Neither feature becomes a production hard gate.",
        f"11. **Ranking feature now?** No production ranking change. "
        f"Forward-validation status is `{stats.get('final_trend')}` / `{stats.get('final_rs')}`. "
        f"Within-day rank spreads: `{rk.get('top_minus_bottom')}`.",
        f"12. **Remain research-only?** Any feature not labelled FORWARD-VALIDATE stays research-only. "
        f"Even FORWARD-VALIDATE is a *future-data* ticket, not paper/live.",
        f"13. **SEPA semantics to deprecate:** `MEETS SEPA`, Ready copy that Stage-2 **is** "
        f"Minervini SEPA for money, Ideas SEPA-as-eligibility, any autopilot plan requiring Core F. "
        f"See `FEATURE_001_SEPA_DEPRECATION.md`.",
        "",
        "## What this is not",
        "",
        "- Not a Core F resurrection",
        "- Not a VCP retune",
        "- Not a new RS cutoff",
        "- Not paper, not live, not VALIDATED_EDGE",
        "",
        "## Forward observation",
        "",
        "Shadow feature logging only, dates strictly after this freeze. "
        "Not activated in production in this milestone.",
        "",
        "```json",
        json.dumps(forward_record_template(), indent=2),
        "```",
    ]
    _write("FEATURE_001_DECISION.md", "\n".join(lines))
    _write("FEATURE_001_FORWARD_LEDGER.md",
           "# FEATURE-001 forward ledger\n\n"
           "Documented only. Do not wire into `app.py` in this milestone.\n\n"
           "```json\n" + json.dumps(forward_record_template(), indent=2) + "\n```\n")


def _use_mode(stats: dict, feat: str) -> str:
    classes = [p[feat] for p in stats["policy"].values() if p["n"] >= MIN_N]
    if not classes:
        return "insufficient data — research-only"
    if classes.count("POSITIVE_RANK_FEATURE") >= 3:
        return "ranking (explanatory); forward-validate later"
    if classes.count("RISK_FILTER_VALUE") >= 3:
        return "tail-risk / RISK_FILTER_VALUE, not ALPHA"
    if classes.count("REDUNDANT") >= 3:
        return "mostly redundant with existing momentum/score"
    if classes.count("NEGATIVE") >= 3:
        return "harmful on this history — do not gate"
    return "unstable or mixed — keep research-only"


def write_manifest(meta: dict, stats: dict) -> None:
    manifest = {
        "experiment": "FEATURE-001",
        "claim_class": "EXPLANATORY",
        "core_sepa_status": "RETIRED_RESEARCH_BENCHMARK",
        "trend_version": "trend_features_v1",
        "rs_version": "rs_features_v1",
        "rs_source": "rs_cs_v1",
        "meta": meta,
        "final_trend": stats.get("final_trend"),
        "final_rs": stats.get("final_rs"),
        "n_events": stats.get("n_events"),
        "n_family_rows": stats.get("n_family_rows"),
        "n_tested": stats.get("n_tested"),
    }
    (OUT_DIR / "feature_001_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    (OUT_DIR / "feature_001_per_strategy.json").write_text(
        json.dumps({"baselines": stats.get("baselines"), "policy": stats.get("policy")},
                   indent=2, default=str)
    )


def write_all(stats: dict, meta: dict) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_baselines(stats, meta)
    write_trend(stats)
    write_rs(stats)
    write_interactions(stats)
    write_ranking(stats)
    write_risk(stats)
    write_temporal(stats)
    write_deprecation()
    write_results(stats, meta)
    write_decision(stats, meta)
    write_manifest(meta, stats)
    return {p.name: str(p) for p in OUT_DIR.glob("FEATURE_001_*.md")}
