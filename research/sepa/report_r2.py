"""Write SEPA-001R2.1 markdown reports from a payload. Research only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

_OUT = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "SEPA-001R2"
_ROOT = Path(__file__).resolve().parents[2]


def _var_row(vid: str, stats: dict[str, Any]) -> str:
    if stats.get("not_sepa_r"):
        f20 = (stats.get("fwd_20d") or {})
        return (
            f"| {vid} | signal-day | {stats.get('n_raw_signal_days', 0)} | "
            f"{stats.get('n_deduped', stats.get('n', 0))} | — | "
            f"{f20.get('mean')} | {stats.get('hit_5pct')} | {stats.get('hit_10pct')} | "
            f"— | — | — | {stats.get('statistical_verdict')} / "
            f"{(stats.get('deployment') or {}).get('label')} |"
        )
    ci = stats.get("block_ci") or {}
    if isinstance(ci, dict) and "ci_lower" in ci:
        ci_s = f"[{ci.get('ci_lower'):+.3f}, {ci.get('ci_upper'):+.3f}]" if ci.get("ci_lower") is not None else "—"
    else:
        ci_s = "—"
    dep = (stats.get("deployment") or {}).get("label") or ""
    return (
        f"| {vid} | {stats.get('statistical_unit', '—')[:24]} | "
        f"{stats.get('n_raw_signal_days', '—')} | {stats.get('n_deduped', stats.get('n', 0))} | "
        f"{stats.get('expectancy_r')} | {stats.get('profit_factor')} | {stats.get('win_rate')} | "
        f"{stats.get('avg_win')} | {stats.get('avg_loss')} | {stats.get('max_dd_r')} | "
        f"{ci_s} | {stats.get('statistical_verdict')} / {dep} |"
    )


def write_results(payload: dict[str, Any]) -> Path:
    sample = payload.get("sample") or {}
    diag = payload.get("diagnostics") or {}
    lines = [
        "# SEPA-001R2 Results",
        "",
        f"**Revision:** `{payload.get('revision') or 'SEPA-001R2'}`  ",
        f"**Eligibility:** `{payload.get('eligibility_version')}`  ",
        f"**VCP:** `{payload.get('vcp_version')}`  ",
        f"**Pivot:** `{payload.get('pivot_version')}`  ",
        f"**Config hash:** `{payload.get('config_hash')}`  ",
        f"**Data:** {payload.get('data_source')}  ",
        f"**Eval:** {sample.get('evaluation_start')} → {sample.get('evaluation_end')} "
        f"(warmup {sample.get('warmup_sessions')} sessions)  ",
        f"**Observation:** `date_step={sample.get('date_step')}` "
        f"`scanner_step={sample.get('scanner_step')}` "
        f"(canonical daily={diag.get('canonical_daily')})  ",
        f"**Universe:** as-of investable, `top_n={sample.get('top_n')}`  ",
        f"**Unique setups:** {sample.get('unique_setups')} "
        f"(left-censored unique {sample.get('left_censored_unique')})  ",
        "",
        "SEPA-001 and SEPA-001R files are immutable. Layer 1 = signal quality. "
        "A–D are scanner-path R studies (deduped by exchange-session embargo). "
        "G is a forward-% signal study, **not** SEPA R. Harness PROMOTE is never "
        "a deployment label. Paper shadow is not live trading.",
        "",
        "## Main comparison (deduplicated units)",
        "",
        "| Variant | Unit | Raw n | Deduped n | E[R] | PF | Win % | Avg Win | Avg Loss | Max DD R | CI | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for vid, stats in (payload.get("variants") or {}).items():
        lines.append(_var_row(vid, stats))
    lines += [
        "",
        "## Diagnostics (R2.1 vs prior runner bugs)",
        "",
        f"- Static future-CA false removals (symbol×date kept by causal segments): "
        f"**{diag.get('static_quarantine_false_removals')}**",
        f"- Scanner step-5 would have missed A signal-days: **{diag.get('scanner_step5_missed_a')}**",
        f"- Scanner step-5 would have missed E entry-ready sessions: "
        f"**{diag.get('scanner_step5_missed_e_entry_ready')}**",
        f"- Calendar-day vs session embargo disagreements: "
        f"**{diag.get('embargo_calendar_disagreements')}**",
        f"- CA-censored outcomes: **{diag.get('ca_censored_outcomes')}**",
        "",
        "## Yearly universe (mean as-of size)",
        "",
        "| Year | Mean candidates | Mean investable | As-of points |",
        "|---|---|---|---|",
    ]
    for y, rec in sorted((payload.get("yearly_universe") or {}).items()):
        lines.append(
            f"| {y} | {rec.get('mean_candidates')} | {rec.get('mean_investable')} | {rec.get('as_of_points')} |"
        )
    lines += [
        "",
        "## RS buckets (ungated 20d forward %, as-of universe, CA-uncensored)",
        "",
        "| Bucket | n | Mean 20d % | Median 20d % |",
        "|---|---|---|---|",
    ]
    for k, rec in sorted((payload.get("rs_buckets") or {}).items()):
        lines.append(f"| {k} | {rec.get('n')} | {rec.get('mean_fwd_20d')} | {rec.get('median_fwd_20d')} |")
    lines += ["", "## Sample warnings", ""]
    integ = payload.get("integrity") or {}
    lines.append(f"- PIT class: `{integ.get('overall')}` (as-of metadata, source=bhav_inferred)")
    ca = payload.get("ca_audit") or {}
    lines.append(
        f"- CA complete (global verifier): `{ca.get('ca_complete')}` ; "
        f"ca_research_acceptable: `{ca.get('ca_research_acceptable')}` ; "
        f"unresolved enumerated: {ca.get('n_unresolved')}"
    )
    cov = payload.get("coverage") or {}
    lines.append(f"- Coverage: {cov.get('first')} → {cov.get('last')} ({cov.get('n_symbols')} symbols)")
    text = "\n".join(lines) + "\n"
    path = _OUT / "SEPA_001R2_RESULTS.md"
    path.write_text(text)
    (_ROOT / "SEPA_001R2_RESULTS.md").write_text(text)
    return path


def write_funnel(payload: dict[str, Any]) -> Path:
    snap = payload.get("funnel_snapshots") or {}
    uniq = payload.get("funnel_unique") or {}
    inv = max(1, int(snap.get("investable") or 1))
    lines = [
        "# SEPA-001R2 funnel",
        "",
        "Two funnels. Snapshot rows are symbol×date. Unique rows are setups / "
        "opportunities. Do not add them together or compare 1,500 daily A rows "
        "to 50 unique F setups as if `n` meant the same thing.",
        "",
        "## Snapshot funnel (symbol × date)",
        "",
        "| Stage | Count | % of investable snapshots |",
        "|---|---|---|",
    ]
    for k in ("candidates", "investable", "stage2", "rs_pass", "vcp_detected",
              "pivot_defined", "entry_ready"):
        v = int(snap.get(k) or 0)
        lines.append(f"| {k} | {v} | {100.0 * v / inv:.4f}% |")
    lines += [
        "",
        "## Unique-opportunity funnel",
        "",
        "| Stage | Unique count |",
        "|---|---|",
    ]
    for k, v in uniq.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"Unique setups (ledger): {(payload.get('sample') or {}).get('unique_setups')}",
        f"Left-censored unique: {(payload.get('sample') or {}).get('left_censored_unique')}",
        f"CA-censored outcomes (path crossings): "
        f"{(payload.get('diagnostics') or {}).get('ca_censored_outcomes')}",
        "",
        "Core F fills only `valid_fill` on the unique funnel, after next-open "
        "classification (gap-through / extended / stop-too-wide / left-censored / "
        "CA-censored are refusals or exclusions, not fabricated trades).",
        "",
    ]
    path = _OUT / "SEPA_001R2_FUNNEL.md"
    path.write_text("\n".join(lines))
    return path


def write_walk_forward(payload: dict[str, Any]) -> Path:
    proto = payload.get("validation_protocol") or {}
    lines = [
        "# SEPA-001R2 walk-forward",
        "",
        "Split was **predeclared** in `SEPA_001R2_VALIDATION_PROTOCOL.md` before "
        "final performance was calculated. Blocks are assigned by signal `as_of`.",
        "",
        f"- Development: first eligible date → `{proto.get('development_end')}`",
        f"- Validation: `{proto.get('validation')}`",
        f"- Confirmation: `{proto.get('confirmation')}`",
        "",
        "| Variant | Block | n | E[R] or 20d mean % | Verdict |",
        "|---|---|---|---|---|",
    ]
    for vid, stats in (payload.get("variants") or {}).items():
        wf = stats.get("walk_forward") or {}
        for name in ("development", "validation", "confirmation"):
            rec = wf.get(name) or {}
            if stats.get("not_sepa_r"):
                metric = (rec.get("fwd_20d") or {}).get("mean")
            else:
                metric = rec.get("expectancy_r")
            lines.append(
                f"| {vid} | {name} | {rec.get('n', rec.get('n_deduped', 0))} | "
                f"{metric} | {rec.get('statistical_verdict', '')} |"
            )
    f = (payload.get("variants") or {}).get("F") or {}
    dep = f.get("deployment") or {}
    lines += [
        "",
        "## Confirmation block (deployment evidence)",
        "",
        f"Core F confirmation n = `{(f.get('walk_forward') or {}).get('confirmation', {}).get('n')}`  ",
        f"`has_unseen_block` for F uses this block, not a hardcoded False.  ",
        f"Deployment label: `{dep.get('label')}`  ",
        f"Reasons: {dep.get('reasons')}",
        "",
    ]
    path = _OUT / "SEPA_001R2_WALK_FORWARD.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_all(payload: dict[str, Any]) -> dict[str, Path]:
    _OUT.mkdir(parents=True, exist_ok=True)
    return {
        "results": write_results(payload),
        "funnel": write_funnel(payload),
        "walk_forward": write_walk_forward(payload),
    }
